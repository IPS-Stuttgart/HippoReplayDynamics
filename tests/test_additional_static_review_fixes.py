from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from hipporeplayimm.accuracy_upgrades import (
    ContinuousTimeEmissionConfig,
    build_continuous_time_emissions,
    restrict_emissions_to_mask,
)
from hipporeplayimm.benchmarks import (
    BenchmarkConfig,
    _candidate_indices_for_model as _benchmark_candidate_indices_for_model,
    _effective_state_space_imm_stickiness,
    _score_state_space_model as _benchmark_score_state_space_model,
    _state_space_decoder_config,
)
from hipporeplayimm.data import ReplaySession, _coerce_mark_matrix
from hipporeplayimm.duration_dynamics import attach_duration_metadata, transition_durations_s
from hipporeplayimm.duration_occupancy import _mode_transition_matrices
from hipporeplayimm.encoding import (
    EmissionConfig,
    EncodingConfig,
    EncodingModel,
    LogEmissionTensor,
    _poisson_log_emissions,
    build_emissions,
    fit_place_field_encoding,
)
from hipporeplayimm.ground_truth import _score_state_space_joint_for_ground_truth
from hipporeplayimm.kd_reference import KDEncodingConfig, build_kd_emissions, fit_kd_place_field_encoding
from hipporeplayimm.models import EventScore
from hipporeplayimm.result_improvement_extensions import (
    build_sorted_emissions_with_replay_calibration,
    score_replay_model_compat,
)
from hipporeplayimm.simulation_recovery import (
    _candidate_indices_for_model as _simulation_candidate_indices_for_model,
    simulate_replay_event,
)
from hipporeplayimm.state_space_displacement_momentum import (
    _displacement_transition_matrix,
    _duration_adjusted_decays as _displacement_duration_adjusted_decays,
    _shifted_gaussian_transition_matrix,
)
from hipporeplayimm.state_space_model import StateSpaceDecoderConfig, _momentum_velocity_decays
from hipporeplayimm.state_space_sparse_momentum import (
    _duration_adjusted_decays as _sparse_momentum_duration_adjusted_decays,
)


ROOT = Path(__file__).resolve().parents[1]


def _load_script_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _minimal_event_score(model_name: str = "dummy") -> EventScore:
    return EventScore(
        model_name=model_name,
        log_likelihood=0.0,
        n_time=1,
        n_spikes=0,
        terminal_log_posterior=np.array([0.0], dtype=float),
    )


def _empty_spike_session() -> ReplaySession:
    times = np.linspace(0.0, 2.0, 21)
    position = np.column_stack(
        [
            times,
            np.linspace(0.0, 20.0, times.shape[0]),
            np.linspace(0.0, 5.0, times.shape[0]),
        ]
    )
    return ReplaySession(
        rat="RatX",
        name="Open1",
        path=Path("RatX/Open1"),
        position=position,
        spikes=np.empty((0, 2), dtype=float),
        tetrode_cell_ids=np.empty((0, 2), dtype=float),
        excitatory_neurons=np.array([], dtype=int),
        inhibitory_neurons=np.array([], dtype=int),
        ripple_events=np.empty((0, 6), dtype=float),
        run_times=np.array([[0.0, 2.0]], dtype=float),
        sleep_box_immobile_times=np.empty((0, 2), dtype=float),
        sleep_times=np.empty((0, 2), dtype=float),
        rem_times=np.empty((0, 2), dtype=float),
        well_sequence=None,
        metadata={},
    )


def _single_ripple_session() -> ReplaySession:
    session = _empty_spike_session()
    session.ripple_events = np.array([[0.20, 0.30, 0.25, 1.0, 0.0, 0.0]], dtype=float)
    return session


def _single_bin_encoding(cell_ids: tuple[int, ...] = ()) -> EncodingModel:
    ids = np.asarray(cell_ids, dtype=int)
    return EncodingModel(
        x_edges=np.array([0.0, 1.0], dtype=float),
        y_edges=np.array([0.0, 1.0], dtype=float),
        bin_centers=np.array([[0.5, 0.5]], dtype=float),
        rates_hz=np.ones((ids.shape[0], 1), dtype=float),
        occupancy_s=np.array([1.0], dtype=float),
        cell_ids=ids,
        config=EncodingConfig(),
    )


def test_place_field_encoding_smoothing_handles_empty_cell_set():
    session = _empty_spike_session()

    encoding = fit_place_field_encoding(
        session,
        EncodingConfig(bin_size_cm=5.0, smoothing_sigma_bins=1.0, min_speed_cm_s=0.0),
    )

    assert encoding.n_cells == 0
    assert encoding.rates_hz.shape == (0, encoding.n_bins)


def test_kd_place_field_encoding_smoothing_handles_empty_cell_set():
    session = _empty_spike_session()

    encoding = fit_kd_place_field_encoding(
        session,
        KDEncodingConfig(
            bin_size_cm=5.0,
            n_bins_x=8,
            n_bins_y=8,
            smoothing_sigma_cm=5.0,
            min_speed_cm_s=0.0,
        ),
    )

    assert encoding.n_cells == 0
    assert encoding.rates_hz.shape == (0, encoding.n_bins)


def test_poisson_emissions_accept_explicit_empty_cell_weights_for_zero_cells():
    log_likelihood = _poisson_log_emissions(
        np.zeros((3, 0), dtype=int),
        np.zeros((0, 2), dtype=float),
        np.array([0.01, 0.02, 0.03], dtype=float),
        cell_weights=[],
    )

    np.testing.assert_allclose(log_likelihood, np.zeros((3, 2), dtype=float))


def test_single_spike_mark_matrix_preserves_feature_columns_and_drops_time_column():
    spike_times = np.array([0.125], dtype=float)

    marks = _coerce_mark_matrix(
        np.array([[0.125, 10.0, 20.0, 30.0]], dtype=float),
        spike_count=1,
        spike_times=spike_times,
    )
    assert marks is not None
    np.testing.assert_allclose(marks, np.array([[10.0, 20.0, 30.0]], dtype=float))

    transposed = _coerce_mark_matrix(
        np.array([[10.0], [20.0], [30.0]], dtype=float),
        spike_count=1,
        spike_times=spike_times,
    )
    assert transposed is not None
    np.testing.assert_allclose(transposed, np.array([[10.0, 20.0, 30.0]], dtype=float))


def test_single_spike_one_dimensional_mark_vector_preserves_features_and_drops_time_column():
    spike_times = np.array([0.125], dtype=float)

    marks = _coerce_mark_matrix(
        np.array([0.125, 10.0, 20.0, 30.0], dtype=float),
        spike_count=1,
        spike_times=spike_times,
    )
    assert marks is not None
    np.testing.assert_allclose(marks, np.array([[10.0, 20.0, 30.0]], dtype=float))

    feature_only = _coerce_mark_matrix(
        np.array([10.0, 20.0, 30.0], dtype=float),
        spike_count=1,
        spike_times=spike_times,
    )
    assert feature_only is not None
    np.testing.assert_allclose(feature_only, np.array([[10.0, 20.0, 30.0]], dtype=float))


def test_sorted_emission_builders_accept_numpy_integer_ripple_indices():
    session = _single_ripple_session()
    encoding = _single_bin_encoding()

    emissions = [
        build_emissions(session, encoding, np.int64(0), EmissionConfig(time_bin_s=0.05)),
        build_kd_emissions(session, encoding, np.int64(0), time_bin_s=0.05),
        build_sorted_emissions_with_replay_calibration(
            session,
            encoding,
            np.int64(0),
            EmissionConfig(time_bin_s=0.05),
        ),
    ]

    for current in emissions:
        assert current.n_time == 2
        assert current.n_spikes == 0
        assert current.log_likelihood.shape == (2, 1)


def test_continuous_time_emissions_accept_numpy_integer_and_preserve_interval_durations():
    session = _single_ripple_session()
    session.spikes = np.array([[0.22, 1.0]], dtype=float)
    encoding = _single_bin_encoding((1,))

    emissions = build_continuous_time_emissions(
        session,
        encoding,
        np.int64(0),
        ContinuousTimeEmissionConfig(min_interval_s=1e-6),
    )

    np.testing.assert_allclose(emissions.bin_durations, np.array([0.02, 0.08]), atol=1e-12)
    np.testing.assert_allclose(emissions.transition_durations, np.array([0.08]), atol=1e-12)
    assert emissions.n_spikes == 1


def test_restrict_emissions_to_mask_preserves_duration_metadata():
    emissions = LogEmissionTensor(
        log_likelihood=np.array(
            [[0.0, -1.0, -2.0], [-3.0, -4.0, -5.0]],
            dtype=float,
        ),
        spike_counts=np.zeros((2, 1), dtype=int),
        times=np.array([0.02, 0.10], dtype=float),
        dt=0.05,
        cell_ids=np.array([1], dtype=int),
        n_spikes=0,
        bin_durations=np.array([0.02, 0.08], dtype=float),
        transition_durations=np.array([0.08], dtype=float),
        metadata={"emission_model": "variable-duration"},
    )

    restricted = restrict_emissions_to_mask(
        emissions,
        np.array([True, False, True], dtype=bool),
    )

    np.testing.assert_allclose(
        restricted.log_likelihood,
        emissions.log_likelihood[:, [0, 2]],
    )
    np.testing.assert_allclose(
        restricted.bin_durations,
        np.array([0.02, 0.08], dtype=float),
    )
    np.testing.assert_allclose(
        restricted.transition_durations,
        np.array([0.08], dtype=float),
    )
    assert restricted.metadata == {"emission_model": "variable-duration"}
    assert restricted.metadata is not emissions.metadata


def test_momentum_decay_helpers_reject_nonfinite_tau_and_decay():
    durations = np.array([0.01, 0.02], dtype=float)

    bad_tau = StateSpaceDecoderConfig(momentum_velocity_decay_tau_s=float("nan"))
    with pytest.raises(ValueError, match="momentum_velocity_decay_tau_s"):
        _momentum_velocity_decays(bad_tau, durations)
    with pytest.raises(ValueError, match="momentum_velocity_decay_tau_s"):
        _sparse_momentum_duration_adjusted_decays(bad_tau, durations, 0.01)
    with pytest.raises(ValueError, match="momentum_velocity_decay_tau_s"):
        _displacement_duration_adjusted_decays(bad_tau, durations, 0.01)

    bad_decay = StateSpaceDecoderConfig(momentum_velocity_decay=float("nan"))
    with pytest.raises(ValueError, match="momentum_velocity_decay"):
        _momentum_velocity_decays(bad_decay, durations)
    with pytest.raises(ValueError, match="momentum_velocity_decay"):
        _sparse_momentum_duration_adjusted_decays(bad_decay, durations, 0.01)
    with pytest.raises(ValueError, match="momentum_velocity_decay"):
        _displacement_duration_adjusted_decays(bad_decay, durations, 0.01)
    with pytest.raises(ValueError, match="reference dt"):
        _displacement_duration_adjusted_decays(StateSpaceDecoderConfig(), durations, float("nan"))


def test_displacement_transition_helpers_validate_and_normalize_degenerate_columns():
    centers = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)
    with pytest.raises(ValueError, match="max_step_sigma"):
        _shifted_gaussian_transition_matrix(
            centers,
            displacement=np.array([0.0, 0.0]),
            sigma_cm=1.0,
            max_step_sigma=float("nan"),
        )

    vectors = np.array([[0.0], [1.0], [2.0]], dtype=float)
    transition = _displacement_transition_matrix(vectors, sigma_cm=1e-150, decay=0.5)

    np.testing.assert_allclose(transition.sum(axis=0), np.ones(vectors.shape[0]))
    assert np.all(np.isfinite(transition))
    assert np.all(transition >= 0.0)
    assert transition[:, 1].max() == pytest.approx(1.0)


def test_simulation_recovery_candidate_helper_passes_bin_centers_when_supported():
    emissions = LogEmissionTensor(
        log_likelihood=np.zeros((1, 2), dtype=float),
        spike_counts=np.zeros((1, 1), dtype=int),
        times=np.array([0.0], dtype=float),
        dt=0.1,
        cell_ids=np.array([1], dtype=int),
        n_spikes=0,
    )
    bin_centers = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)

    class BinCenterAwareModel:
        def candidate_indices(self, observed, centers=None):
            assert observed is emissions
            assert centers is bin_centers
            return [np.array([0, 1], dtype=int)]

    class LegacyModel:
        def candidate_indices(self, observed):
            assert observed is emissions
            return [np.array([1], dtype=int)]

    assert _simulation_candidate_indices_for_model(BinCenterAwareModel(), emissions, bin_centers)[0].tolist() == [0, 1]
    assert _simulation_candidate_indices_for_model(LegacyModel(), emissions, bin_centers)[0].tolist() == [1]


def test_candidate_helpers_do_not_swallow_bin_center_aware_type_errors():
    emissions = LogEmissionTensor(
        log_likelihood=np.zeros((1, 2), dtype=float),
        spike_counts=np.zeros((1, 1), dtype=int),
        times=np.array([0.0], dtype=float),
        dt=0.1,
        cell_ids=np.array([1], dtype=int),
        n_spikes=0,
    )
    bin_centers = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)

    class BrokenBinCenterAwareModel:
        def candidate_indices(self, observed, centers=None):
            assert observed is emissions
            if centers is not None:
                raise TypeError("internal candidate support bug")
            return [np.array([1], dtype=int)]

    for helper in (_simulation_candidate_indices_for_model, _benchmark_candidate_indices_for_model):
        with pytest.raises(TypeError, match="internal candidate support bug"):
            helper(BrokenBinCenterAwareModel(), emissions, bin_centers)


def test_candidate_helpers_pass_keyword_only_bin_centers_when_supported():
    emissions = LogEmissionTensor(
        log_likelihood=np.zeros((1, 2), dtype=float),
        spike_counts=np.zeros((1, 1), dtype=int),
        times=np.array([0.0], dtype=float),
        dt=0.1,
        cell_ids=np.array([1], dtype=int),
        n_spikes=0,
    )
    bin_centers = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)

    class KeywordOnlyBinCenterModel:
        def candidate_indices(self, observed, *, centers=None):
            assert observed is emissions
            assert centers is bin_centers
            return [np.array([0], dtype=int)]

    for helper in (_simulation_candidate_indices_for_model, _benchmark_candidate_indices_for_model):
        assert helper(KeywordOnlyBinCenterModel(), emissions, bin_centers)[0].tolist() == [0]


def test_state_space_imm_switch_tau_is_preserved_for_duration_scorer():
    config = BenchmarkConfig(
        emissions=EmissionConfig(time_bin_s=0.02),
        state_space_imm_mode_stickiness=0.91,
        state_space_imm_switch_tau_s=0.5,
    )

    decoder_config = _state_space_decoder_config(config, "imm")

    assert decoder_config.imm_mode_stickiness == pytest.approx(np.exp(-0.02 / 0.5))
    assert decoder_config.imm_switch_tau_s == pytest.approx(0.5)
    assert _effective_state_space_imm_stickiness(config) == pytest.approx(np.exp(-0.02 / 0.5))


def test_imm_switch_tau_builds_duration_specific_transition_matrices():
    import hipporeplayimm.state_space as ss

    durations = np.array([0.02, 0.04, 0.08], dtype=float)

    matrices = _mode_transition_matrices(
        ss,
        n_modes=4,
        mode_stickiness=0.90,
        imm_switch_tau_s=0.5,
        durations=durations,
    )

    assert len(matrices) == durations.shape[0]
    for matrix, duration in zip(matrices, durations, strict=True):
        expected_stickiness = np.exp(-duration / 0.5)
        np.testing.assert_allclose(np.diag(matrix), expected_stickiness)
        off_diagonal = matrix[~np.eye(matrix.shape[0], dtype=bool)]
        np.testing.assert_allclose(off_diagonal, (1.0 - expected_stickiness) / 3.0)

    with pytest.raises(ValueError, match="imm_switch_tau_s"):
        _mode_transition_matrices(ss, 4, 0.90, float("inf"), durations)


def test_state_space_score_helpers_do_not_swallow_occupancy_type_errors():
    emissions = LogEmissionTensor(
        log_likelihood=np.zeros((1, 2), dtype=float),
        spike_counts=np.zeros((1, 1), dtype=int),
        times=np.array([0.0], dtype=float),
        dt=0.1,
        cell_ids=np.array([1], dtype=int),
        n_spikes=0,
    )
    bin_centers = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)
    occupancy_s = np.ones(2, dtype=float)

    class BrokenOccupancyAwareModel:
        mode = "diffusion"

        def score(self, observed, centers, *, candidate_indices=None, occupancy_s=None, return_trajectory=True):
            del candidate_indices, return_trajectory
            assert observed is emissions
            assert centers is bin_centers
            if occupancy_s is not None:
                raise TypeError("internal occupancy_s propagation bug")
            return _minimal_event_score()

    model = BrokenOccupancyAwareModel()
    with pytest.raises(TypeError, match="internal occupancy_s propagation bug"):
        _benchmark_score_state_space_model(model, emissions, bin_centers, None, occupancy_s)
    with pytest.raises(TypeError, match="internal occupancy_s propagation bug"):
        _score_state_space_joint_for_ground_truth(model, emissions, bin_centers, None, occupancy_s)


def test_state_space_score_helpers_omit_occupancy_for_legacy_scores():
    emissions = LogEmissionTensor(
        log_likelihood=np.zeros((1, 2), dtype=float),
        spike_counts=np.zeros((1, 1), dtype=int),
        times=np.array([0.0], dtype=float),
        dt=0.1,
        cell_ids=np.array([1], dtype=int),
        n_spikes=0,
    )
    bin_centers = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)
    occupancy_s = np.ones(2, dtype=float)
    candidates = [np.array([0, 1], dtype=int)]

    class LegacyScoreModel:
        mode = "diffusion"

        def __init__(self) -> None:
            self.calls = 0

        def score(self, observed, centers, *, candidate_indices=None):
            assert observed is emissions
            assert centers is bin_centers
            assert candidate_indices is candidates
            self.calls += 1
            return _minimal_event_score("legacy")

    benchmark_model = LegacyScoreModel()
    benchmark_score = _benchmark_score_state_space_model(
        benchmark_model,
        emissions,
        bin_centers,
        candidates,
        occupancy_s,
    )
    assert benchmark_score.model_name == "legacy"
    assert benchmark_model.calls == 1

    ground_truth_model = LegacyScoreModel()
    ground_truth_score = _score_state_space_joint_for_ground_truth(
        ground_truth_model,
        emissions,
        bin_centers,
        candidates,
        occupancy_s,
    )
    assert ground_truth_score.model_name == "legacy"
    assert ground_truth_model.calls == 1


def test_score_replay_model_compat_does_not_swallow_optional_argument_type_errors():
    emissions = LogEmissionTensor(
        log_likelihood=np.zeros((1, 2), dtype=float),
        spike_counts=np.zeros((1, 1), dtype=int),
        times=np.array([0.0], dtype=float),
        dt=0.1,
        cell_ids=np.array([1], dtype=int),
        n_spikes=0,
    )
    bin_centers = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)

    class BrokenCandidateModel:
        name = "broken-candidates"

        def candidate_indices(self, observed, centers=None):
            assert observed is emissions
            if centers is not None:
                raise TypeError("internal candidate support bug")
            return [np.array([0], dtype=int)]

        def score(self, observed, centers, *, candidate_indices=None):
            del candidate_indices
            assert observed is emissions
            assert centers is bin_centers
            return _minimal_event_score()

    class BrokenOccupancyModel:
        name = "broken-occupancy"

        def score(self, observed, centers, *, occupancy_s=None):
            assert observed is emissions
            assert centers is bin_centers
            if occupancy_s is not None:
                raise TypeError("internal occupancy_s scoring bug")
            return _minimal_event_score()

    with pytest.raises(TypeError, match="internal candidate support bug"):
        score_replay_model_compat(BrokenCandidateModel(), emissions, bin_centers)
    with pytest.raises(TypeError, match="internal occupancy_s scoring bug"):
        score_replay_model_compat(
            BrokenOccupancyModel(),
            emissions,
            bin_centers,
            occupancy_s=np.ones(2, dtype=float),
        )


def test_simulated_replay_rejects_nonfinite_sampling_scalars_before_poisson():
    encoding = EncodingModel(
        x_edges=np.array([0.0, 1.0]),
        y_edges=np.array([0.0, 1.0]),
        bin_centers=np.array([[0.5, 0.5]], dtype=float),
        rates_hz=np.ones((1, 1), dtype=float),
        occupancy_s=np.array([1.0], dtype=float),
        cell_ids=np.array([1], dtype=int),
        config=EncodingConfig(),
    )

    with pytest.raises(ValueError, match="dt"):
        simulate_replay_event(
            encoding,
            true_model="stationary",
            n_time=1,
            dt=float("nan"),
            rng=np.random.default_rng(1),
        )

    with pytest.raises(ValueError, match="spike_rate_scale"):
        simulate_replay_event(
            encoding,
            true_model="stationary",
            n_time=1,
            dt=0.02,
            rng=np.random.default_rng(1),
            spike_rate_scale=float("nan"),
        )


def test_track_event_prefix_emissions_slices_duration_metadata():
    track_event = _load_script_module("track_event_under_test", ROOT / "scripts" / "track_event.py")
    emissions = LogEmissionTensor(
        log_likelihood=np.zeros((5, 3), dtype=float),
        spike_counts=np.zeros((5, 1), dtype=int),
        times=np.array([0.05, 0.15, 0.25, 0.40, 0.55], dtype=float),
        dt=0.1,
        cell_ids=np.array([1], dtype=int),
        n_spikes=0,
    )
    emissions.transition_durations = np.array([0.1, 0.1, 0.15, 0.15], dtype=float)
    attach_duration_metadata(emissions)

    prefix = track_event._prefix_emissions(emissions, 3)

    np.testing.assert_allclose(transition_durations_s(prefix), np.array([0.1, 0.1]))
    assert tuple(prefix.dt.transition_durations) == (0.1, 0.1)
    assert prefix.n_time == 3
    assert prefix.n_spikes == 0
