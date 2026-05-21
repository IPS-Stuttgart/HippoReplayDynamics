from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import replace
import inspect

import numpy as np
import pytest

from hipporeplayimm import cli
from hipporeplayimm.benchmarks import BenchmarkConfig, _build_models, _cell_split_seed, _score_session_split, _split_cells
from hipporeplayimm.clusterless import ClusterlessMarkConfig, ClusterlessMarkEncoding
from hipporeplayimm.data import ReplaySession, SpikeMarkData
from hipporeplayimm.encoding import EncodingConfig, LogEmissionTensor
from hipporeplayimm.models import CandidateKinematicModel, DiffusionModel
from hipporeplayimm.pyrecest_models import PyRecEstGoalParticleModel
from hipporeplayimm.state_space import StateSpaceDecoderConfig, StateSpaceReplayModel


class _EncodingStub:
    def __init__(self, cell_ids: np.ndarray):
        self.cell_ids = np.asarray(cell_ids, dtype=int)
        self.bin_centers = np.array([[0.0, 0.0]])
        self.occupancy_s = np.array([1.0])

    def select_cells(self, cell_ids):
        selected = _EncodingStub(np.asarray(cell_ids, dtype=int))
        selected.bin_centers = self.bin_centers
        selected.occupancy_s = self.occupancy_s
        return selected


def _minimal_session_with_marks() -> ReplaySession:
    cell_ids = np.array([1, 2, 3, 4], dtype=int)
    marks = SpikeMarkData(
        times=np.arange(4, dtype=float),
        marks=np.arange(8, dtype=float).reshape(4, 2),
        source_file="Spike_Data.mat",
        source_variable="Marks",
        feature_names=("m0", "m1"),
        cell_ids=cell_ids,
        group_ids=cell_ids,
    )
    return ReplaySession(
        rat="RatX",
        name="OpenY",
        path=None,  # type: ignore[arg-type]
        position=np.zeros((2, 3), dtype=float),
        spikes=np.column_stack([np.arange(4, dtype=float), cell_ids.astype(float)]),
        tetrode_cell_ids=np.column_stack([cell_ids, cell_ids]),
        excitatory_neurons=cell_ids,
        inhibitory_neurons=np.array([], dtype=int),
        ripple_events=np.empty((0, 6), dtype=float),
        run_times=np.empty((0, 2), dtype=float),
        sleep_box_immobile_times=np.empty((0, 2), dtype=float),
        sleep_times=np.empty((0, 2), dtype=float),
        rem_times=np.empty((0, 2), dtype=float),
        well_sequence=None,
        metadata={},
        spike_marks=marks,
    )


def test_shared_encoding_cli_arguments_are_registered() -> None:
    parser = ArgumentParser()
    cli._add_encoding_arguments(parser)
    args = parser.parse_args([])

    assert hasattr(args, "min_occupancy_s")
    assert hasattr(args, "rate_floor_hz")

    config = cli._encoding_config_from_args(args)
    defaults = EncodingConfig()
    assert config.min_occupancy_s == defaults.min_occupancy_s
    assert config.rate_floor_hz == defaults.rate_floor_hz


def test_shared_clusterless_cli_arguments_include_mark_group_by() -> None:
    parser = ArgumentParser()
    cli._add_encoding_arguments(parser)
    cli._add_clusterless_arguments(parser)
    args = parser.parse_args(["--clusterless-mark-group-by", "tetrode"])

    assert cli._clusterless_scalar_kwargs(args)["clusterless_mark_group_by"] == "tetrode"
    assert cli._clusterless_mark_config_from_args(args).mark_group_by == "tetrode"


def test_build_models_rejects_unknown_model_with_clear_error() -> None:
    with pytest.raises(ValueError, match="Unknown model name"):
        _build_models(BenchmarkConfig(models=("random", "not-a-model")))


def test_clusterless_fit_uses_train_marks_even_when_all_cells_are_enabled(monkeypatch) -> None:
    import hipporeplayimm.benchmarks as benchmarks

    captured = []
    session = _minimal_session_with_marks()
    encoding = _EncodingStub(np.array([1, 2, 3, 4], dtype=int))
    config = BenchmarkConfig(
        encoding=replace(EncodingConfig(), use_excitatory=False),
        test_cell_fraction=0.5,
        random_seed=11,
        models=("clusterless-state-space-imm",),
    )

    monkeypatch.setattr(benchmarks, "_build_models", lambda config, session=None: {"dummy": object()})
    monkeypatch.setattr(benchmarks, "_is_clusterless_model", lambda model: True)
    monkeypatch.setattr(benchmarks, "_event_indices", lambda *args, **kwargs: np.array([], dtype=int))

    def fake_fit_clusterless_mark_encoding(fit_session, clusterless_config):
        captured.append(fit_session)
        return object()

    monkeypatch.setattr(benchmarks, "fit_clusterless_mark_encoding", fake_fit_clusterless_mark_encoding)

    _score_session_split(session, config, encoding, split_index=0)

    split_seed = _cell_split_seed(config.random_seed, 0)
    expected_train_cells, _ = _split_cells(encoding.cell_ids, config.test_cell_fraction, split_seed)
    fit_cell_ids = captured[0].spike_marks.cell_ids
    assert set(fit_cell_ids.astype(int)) == set(expected_train_cells.astype(int))


def test_clusterless_group_ids_reject_fractional_values() -> None:
    encoding = ClusterlessMarkEncoding(
        x_edges=np.array([0.0, 1.0]),
        y_edges=np.array([0.0, 1.0]),
        bin_centers=np.array([[0.5, 0.5]], dtype=float),
        rate_hz=np.array([1.0]),
        occupancy_s=np.array([1.0]),
        effective_spike_count=np.array([1.0]),
        mark_mean=np.zeros((1, 2), dtype=float),
        mark_variance=np.ones((1, 2), dtype=float),
        mark_feature_names=("m0", "m1"),
        spike_mark_source="unit-test",
        config=ClusterlessMarkConfig(mark_likelihood="diagonal-gaussian", mark_group_by="cell"),
        mark_likelihood="diagonal-gaussian",
        group_ids=np.array([1], dtype=int),
        group_mark_mean=np.zeros((1, 1, 2), dtype=float),
        group_mark_variance=np.ones((1, 1, 2), dtype=float),
    )

    with pytest.raises(ValueError, match="integer-valued"):
        encoding.log_mark_likelihood(np.array([[0.0, 0.0]]), group_ids=np.array([1.5]))


def test_log_emission_tensor_metadata_is_declared_field() -> None:
    emissions = LogEmissionTensor(
        log_likelihood=np.zeros((1, 1), dtype=float),
        spike_counts=np.zeros((1, 1), dtype=int),
        times=np.array([0.0]),
        dt=0.02,
        cell_ids=np.array([1]),
        n_spikes=0,
        metadata={"source": "unit-test"},
    )

    assert emissions.metadata["source"] == "unit-test"


def test_duration_state_space_keeps_duration_metadata_with_occupancy_mask() -> None:
    emissions = LogEmissionTensor(
        log_likelihood=np.zeros((3, 4), dtype=float),
        spike_counts=np.zeros((3, 1), dtype=int),
        times=np.array([0.0, 0.01, 0.04], dtype=float),
        dt=0.02,
        cell_ids=np.array([1], dtype=int),
        n_spikes=0,
    )
    emissions.transition_durations = np.array([0.01, 0.03], dtype=float)
    centers = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        dtype=float,
    )
    model = StateSpaceReplayModel(
        mode="diffusion",
        config=StateSpaceDecoderConfig(mode="diffusion", valid_occupancy_threshold_s=0.5),
    )

    score = model.score(emissions, centers, occupancy_s=np.array([1.0, 0.0, 1.0, 0.0]))

    assert score.diagnostics["state_space_transition_durations"] == "0.01,0.03"
    assert score.diagnostics["state_space_valid_bin_count"] == 2


def test_state_space_rejects_nonpositive_diffusion_sigma() -> None:
    emissions = LogEmissionTensor(
        log_likelihood=np.zeros((2, 2), dtype=float),
        spike_counts=np.zeros((2, 1), dtype=int),
        times=np.array([0.0, 0.02], dtype=float),
        dt=0.02,
        cell_ids=np.array([1], dtype=int),
        n_spikes=0,
    )
    centers = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)
    model = StateSpaceReplayModel(
        mode="diffusion",
        config=StateSpaceDecoderConfig(mode="diffusion", diffusion_sigma_cm_sqrt_s=0.0),
    )

    with pytest.raises(ValueError, match="sigma_cm_sqrt_s"):
        model.score(emissions, centers)


def test_model_parameter_validation_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="sigma_cm"):
        DiffusionModel(sigma_cm=0.0)

    with pytest.raises(ValueError, match="mode_stickiness"):
        CandidateKinematicModel(mode="imm", mode_stickiness=2.0)

    with pytest.raises(ValueError, match="n_particles"):
        PyRecEstGoalParticleModel(n_particles=0)

    with pytest.raises(ValueError, match="jump_probability"):
        PyRecEstGoalParticleModel(jump_probability=1.5)


def test_patched_benchmark_config_accepts_cli_state_space_kwargs() -> None:
    parser = ArgumentParser()
    cli._add_state_space_arguments(parser)
    args = parser.parse_args([])

    config = BenchmarkConfig(**cli._state_space_scalar_kwargs(args))

    defaults = StateSpaceDecoderConfig()
    assert config.state_space_valid_occupancy_threshold_s == defaults.valid_occupancy_threshold_s
    assert config.state_space_momentum_candidate_mass_threshold == defaults.momentum_candidate_mass_threshold
    assert config.state_space_momentum_candidate_min_k == defaults.momentum_candidate_min_k
    assert config.state_space_momentum_candidate_max_k == defaults.momentum_candidate_max_k
    assert config.state_space_momentum_predicted_candidate_top_k == defaults.momentum_predicted_candidate_top_k


def test_patched_state_space_model_builder_preserves_candidate_support_knobs() -> None:
    import hipporeplayimm.benchmarks as benchmarks

    config = BenchmarkConfig(
        models=("state-space-imm",),
        state_space_momentum_candidate_mass_threshold=0.95,
        state_space_momentum_candidate_min_k=3,
        state_space_momentum_candidate_max_k=21,
        state_space_momentum_predicted_candidate_top_k=5,
        state_space_valid_occupancy_threshold_s=0.25,
    )

    model = benchmarks._build_models(config)["state-space-imm"]

    assert model.config.momentum_candidate_mass_threshold == pytest.approx(0.95)
    assert model.config.momentum_candidate_min_k == 3
    assert model.config.momentum_candidate_max_k == 21
    assert model.config.momentum_predicted_candidate_top_k == 5
    assert model.config.valid_occupancy_threshold_s == pytest.approx(0.25)


def test_base_compare_ground_truth_accepts_cli_state_space_kwargs() -> None:
    import hipporeplayimm.ground_truth as ground_truth

    compare = ground_truth.compare_scores_to_ground_truth
    for cell in compare.__closure__ or ():
        value = cell.cell_contents
        if callable(value) and getattr(value, "__name__", "") == "compare_scores_to_ground_truth":
            compare = value
            break

    signature = inspect.signature(compare)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return

    parser = ArgumentParser()
    cli._add_state_space_arguments(parser)
    args = parser.parse_args([])
    for name in cli._state_space_scalar_kwargs(args):
        assert name in signature.parameters
