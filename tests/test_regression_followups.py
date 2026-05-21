from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import replace

import numpy as np
import pytest

from hipporeplayimm import cli
from hipporeplayimm.benchmarks import BenchmarkConfig, _cell_split_seed, _score_session_split, _split_cells
from hipporeplayimm.data import ReplaySession, SpikeMarkData
from hipporeplayimm.encoding import EncodingConfig, LogEmissionTensor
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
