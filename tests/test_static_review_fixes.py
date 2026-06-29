from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hipporeplayimm.benchmarks import BenchmarkConfig
from hipporeplayimm.clusterless import ClusterlessMarkConfig, _sqrt_optional, _validate_mark_config
from hipporeplayimm.clusterless_ground_truth import _clusterless_model_config_for_scores
from hipporeplayimm.duration_dynamics import DurationFloat
from hipporeplayimm.goal_state_space import _farthest_point_subset as goal_farthest_point_subset
from hipporeplayimm.pyrecest_models import (
    _farthest_point_subset as pyrecest_farthest_point_subset,
)
from hipporeplayimm.result_improvements import (
    posterior_calibration_summary,
    stratified_cell_split,
)


class _ProposalRecordingFilter:
    def __init__(self, position_particles: np.ndarray, proposal_positions: np.ndarray) -> None:
        self.position_particles = position_particles
        self.proposal_positions = proposal_positions
        self.filter_state = type(
            "_State",
            (),
            {"w": np.ones(self.position_particles.shape[0])},
        )()
        self.recorded_likelihoods: np.ndarray | None = None

    def update_position_likelihood_with_proposal(self, likelihood_fn, **kwargs):
        del kwargs
        self.recorded_likelihoods = np.asarray(likelihood_fn(self.proposal_positions), dtype=float)
        return 0.0


def test_pyrecest_proposal_callback_evaluates_requested_positions() -> None:
    filters = pytest.importorskip("pyrecest.filters")

    bin_centers = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    log_likelihood = np.array([-10.0, -5.0, 0.0])
    proposal_positions = np.array([[2.0, 0.0], [0.0, 0.0]])
    filter_ = _ProposalRecordingFilter(
        position_particles=np.array([[0.0, 0.0], [1.0, 0.0]]),
        proposal_positions=proposal_positions,
    )

    filters.update_position_grid_likelihood(
        filter_,
        log_likelihood,
        bin_centers,
        position_proposal_probability=1.0,
    )

    assert filter_.recorded_likelihoods is not None
    np.testing.assert_allclose(filter_.recorded_likelihoods, np.exp([0.0, -10.0]))


def test_duration_float_arithmetic_uses_scalar_base_dt() -> None:
    dt = DurationFloat(0.02, [0.001, 0.003])
    assert dt * 10.0 == pytest.approx(0.2)
    assert 10.0 * dt == pytest.approx(0.2)


def test_farthest_point_subset_accepts_one_dimensional_positions() -> None:
    points = np.arange(6.0)[:, None]
    assert goal_farthest_point_subset(points, max_points=3).shape == (3, 1)
    assert pyrecest_farthest_point_subset(points, max_points=3).shape == (3, 1)


def test_patched_benchmark_config_preserves_clusterless_fields() -> None:
    config = BenchmarkConfig()
    assert config.clusterless_mark_likelihood == "local-kde"
    assert config.clusterless_mark_kde_bandwidth is None
    assert config.clusterless_mark_kde_spatial_sigma_bins is None
    assert config.clusterless_mark_kde_max_neighbors == 256


def test_clusterless_mark_config_rejects_nonfinite_parameters() -> None:
    with pytest.raises(ValueError, match="mark_smoothing_sigma_bins"):
        _validate_mark_config(ClusterlessMarkConfig(mark_smoothing_sigma_bins=float("nan")))

    with pytest.raises(ValueError, match="mark_kde_spatial_sigma_bins"):
        _validate_mark_config(ClusterlessMarkConfig(mark_kde_spatial_sigma_bins=float("nan")))

    with pytest.raises(ValueError, match="mark_kde_max_neighbors"):
        _validate_mark_config(ClusterlessMarkConfig(mark_kde_max_neighbors=1.5))


def test_clusterless_ground_truth_recovers_mark_likelihood_metadata() -> None:
    scores = pd.DataFrame(
        {
            "clusterless_mark_likelihood": ["diagonal-gaussian"],
            "clusterless_mark_kde_bandwidth": [2.5],
            "clusterless_mark_kde_spatial_sigma_bins": [1.25],
            "clusterless_mark_kde_max_neighbors": [64],
        }
    )

    config = _clusterless_model_config_for_scores(
        scores,
        model_names=("clusterless-state-space-diffusion",),
        state_space_stationary_sigma_cm=2.0,
        state_space_diffusion_sigma_cm_sqrt_s=85.0,
        state_space_max_step_sigma=4.0,
        state_space_imm_mode_stickiness=0.95,
        state_space_momentum_sigma_cm_sqrt_s=85.0,
        state_space_momentum_initial_sigma_cm_sqrt_s=85.0,
        state_space_momentum_velocity_decay=0.95,
        state_space_momentum_candidate_top_k=128,
        clusterless_mark_smoothing_sigma_bins=1.0,
        clusterless_mark_prior_count=1.0,
        clusterless_mark_variance_floor=1.0,
        clusterless_rate_floor_hz=1e-4,
        clusterless_mark_likelihood="local-kde",
        clusterless_mark_kde_bandwidth=None,
        clusterless_mark_kde_spatial_sigma_bins=None,
        clusterless_mark_kde_max_neighbors=256,
    )

    assert config.clusterless_mark_likelihood == "diagonal-gaussian"
    assert config.clusterless_mark_kde_bandwidth == pytest.approx(2.5)
    assert config.clusterless_mark_kde_spatial_sigma_bins == pytest.approx(1.25)
    assert config.clusterless_mark_kde_max_neighbors == 64


def test_clusterless_kde_bandwidth_metadata_reports_bandwidth_not_variance() -> None:
    effective_variance = np.array([6.25, 9.0])
    np.testing.assert_allclose(_sqrt_optional(effective_variance), np.array([2.5, 3.0]))


def test_sweep_observation_cli_preserves_continue_on_error_default(monkeypatch, tmp_path) -> None:
    from hipporeplayimm import cli

    captured: dict[str, object] = {}

    class _Result:
        summary = pd.DataFrame()

    def fake_run(root, config):
        captured["root"] = root
        captured["config"] = config
        return _Result()

    def fake_write(result, output):
        captured["result"] = result
        captured["output"] = output

    monkeypatch.setattr(cli, "run_observation_parameter_sweep", fake_run)
    monkeypatch.setattr(cli, "write_observation_sweep_outputs", fake_write)

    assert (
        cli.main(
            [
                "sweep-observation",
                str(tmp_path),
                "--output",
                str(tmp_path / "out"),
                "--skip-simulation-recovery",
            ]
        )
        == 0
    )

    assert captured["root"] == str(tmp_path)
    assert captured["output"] == str(tmp_path / "out")
    assert getattr(captured["config"], "simulation_continue_on_error") is True


def test_stratified_cell_split_downsamples_overfull_strata_randomly() -> None:
    ids = np.arange(4)
    scores = np.arange(4.0)

    train, test = stratified_cell_split(
        ids,
        scores,
        test_fraction=0.3,
        random_seed=0,
        n_strata=2,
    )

    np.testing.assert_array_equal(test, np.array([2]))
    np.testing.assert_array_equal(train, np.array([0, 1, 3]))


def test_posterior_calibration_summary_coerces_probability_column() -> None:
    samples = pd.DataFrame(
        {
            "session": ["Rat1/Open1", "Rat1/Open1"],
            "true_bin_probability": ["0.25", "0.75"],
            "true_bin_rank": [1, 2],
            "n_position_bins": [4, 4],
        }
    )

    summary = posterior_calibration_summary(samples)

    assert summary.loc[0, "mean_true_probability"] == pytest.approx(0.5)
    assert summary.loc[0, "median_true_probability"] == pytest.approx(0.5)
    assert summary.loc[0, "mean_true_negative_log_probability"] == pytest.approx(
        float(np.mean(-np.log([0.25, 0.75])))
    )


def test_shuffle_p_values_keep_window_scopes_separate() -> None:
    from hipporeplayimm.shuffle_controls import add_shuffle_p_values

    real_scores = pd.DataFrame(
        {
            "session": ["Rat1/Open1", "Rat1/Open1"],
            "event_index": [7, 7],
            "model": ["sorted-spike-state-space-imm", "sorted-spike-state-space-imm"],
            "event_window_variant": ["core", "expanded"],
            "log_evidence": [5.0, 5.0],
        }
    )
    control_scores = pd.DataFrame(
        {
            "session": ["Rat1/Open1"] * 4,
            "event_index": [7] * 4,
            "model": ["sorted-spike-state-space-imm"] * 4,
            "event_window_variant": ["core", "core", "expanded", "expanded"],
            "log_evidence": [0.0, 1.0, 10.0, 11.0],
        }
    )

    scored = add_shuffle_p_values(real_scores, control_scores).set_index(
        "event_window_variant"
    )

    assert scored.loc["core", "shuffle_p_value"] == pytest.approx(1.0 / 3.0)
    assert scored.loc["expanded", "shuffle_p_value"] == pytest.approx(1.0)
    assert scored.loc["core", "shuffle_log_evidence_median"] == pytest.approx(0.5)
    assert scored.loc["expanded", "shuffle_log_evidence_median"] == pytest.approx(10.5)
