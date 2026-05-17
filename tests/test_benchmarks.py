import numpy as np
import pandas as pd

from hipporeplayimm.benchmarks import (
    BenchmarkConfig,
    BenchmarkResult,
    _add_relative_metrics,
    _build_models,
    _is_best_static_baseline_model,
    bootstrap_delta_ci,
)
from hipporeplayimm.evidence_reporting import TRUNCATED_EVIDENCE_SUPPORT


def test_benchmark_summary_and_bootstrap_ci():
    rows = pd.DataFrame(
        {
            "model": ["diffusion", "imm", "diffusion", "imm"],
            "heldout_log_likelihood": [-10.0, -9.0, -12.0, -10.0],
            "delta_vs_best_static": [0.0, 1.0, 0.0, 2.0],
            "bits_per_spike_vs_best_static": [0.0, 0.1, 0.0, 0.2],
        }
    )
    result = BenchmarkResult(rows)
    summary = result.summary()
    ci = bootstrap_delta_ci(rows, model="imm", n_bootstrap=100, random_seed=0)

    assert set(summary["model"]) == {"diffusion", "imm"}
    assert np.isfinite(ci[0])
    assert np.isfinite(ci[1])


def test_build_models_includes_opt_in_pyrecest_model():
    models = _build_models(
        BenchmarkConfig(
            models=("pyrecest-goal-particle",),
            pyrecest_particles=64,
            pyrecest_position_proposal_probability=0.5,
        )
    )

    assert set(models) == {"pyrecest-goal-particle"}
    assert models["pyrecest-goal-particle"].position_proposal_probability == 0.5


def test_build_models_includes_opt_in_pyrecest_imm_model():
    models = _build_models(
        BenchmarkConfig(models=("pyrecest-goal-particle-imm",), pyrecest_particles=64)
    )

    assert set(models) == {"pyrecest-goal-particle-imm"}


def test_state_space_aliases_canonicalize_sorted_spike_model_name():
    models = _build_models(BenchmarkConfig(models=("state-space-diffusion",)))

    assert models["state-space-diffusion"].name == "sorted-spike-state-space-diffusion"


def test_best_static_baseline_includes_state_space_single_mode_models():
    assert _is_best_static_baseline_model("random")
    assert _is_best_static_baseline_model("diffusion")
    assert _is_best_static_baseline_model("momentum")
    assert _is_best_static_baseline_model("sorted-spike-state-space-stationary")
    assert _is_best_static_baseline_model("sorted-spike-state-space-diffusion")
    assert _is_best_static_baseline_model("sorted-spike-state-space-fragmented")
    assert _is_best_static_baseline_model("sorted-spike-state-space-jump")
    assert _is_best_static_baseline_model("sorted-spike-state-space-momentum")
    assert _is_best_static_baseline_model("state-space-diffusion")
    assert _is_best_static_baseline_model("clusterless-state-space-stationary")
    assert _is_best_static_baseline_model("clusterless-state-space-diffusion")
    assert _is_best_static_baseline_model("clusterless-state-space-fragmented")
    assert _is_best_static_baseline_model("clusterless-state-space-jump")
    assert _is_best_static_baseline_model("clusterless-state-space-momentum")
    assert not _is_best_static_baseline_model("imm")
    assert not _is_best_static_baseline_model("sorted-spike-state-space-imm")
    assert not _is_best_static_baseline_model("clusterless-state-space-imm")
    assert not _is_best_static_baseline_model("pyrecest-goal-particle")
    assert not _is_best_static_baseline_model("pyrecest-goal-particle-imm")


def test_add_relative_metrics_uses_state_space_single_mode_baselines():
    rows = pd.DataFrame(
        {
            "session": ["s1", "s1", "s1", "s1"],
            "event_index": [7, 7, 7, 7],
            "model": [
                "sorted-spike-state-space-diffusion",
                "sorted-spike-state-space-momentum",
                "sorted-spike-state-space-jump",
                "sorted-spike-state-space-imm",
            ],
            "heldout_log_likelihood": [-8.0, -7.0, -9.0, -6.0],
            "test_spikes": [2, 2, 2, 2],
        }
    )

    result = _add_relative_metrics(rows)
    deltas = dict(zip(result["model"], result["delta_vs_best_static"]))

    assert deltas["sorted-spike-state-space-diffusion"] == -1.0
    assert deltas["sorted-spike-state-space-momentum"] == 0.0
    assert deltas["sorted-spike-state-space-jump"] == -2.0
    assert deltas["sorted-spike-state-space-imm"] == 1.0
    assert result["best_static_heldout_log_likelihood"].notna().all()


def test_add_relative_metrics_uses_clusterless_single_mode_baselines():
    rows = pd.DataFrame(
        {
            "session": ["s1", "s1", "s1", "s1"],
            "event_index": [7, 7, 7, 7],
            "model": [
                "clusterless-state-space-diffusion",
                "clusterless-state-space-momentum",
                "clusterless-state-space-jump",
                "clusterless-state-space-imm",
            ],
            "heldout_log_likelihood": [-8.0, -7.0, -9.0, -6.0],
            "test_spikes": [2, 2, 2, 2],
        }
    )

    result = _add_relative_metrics(rows)
    deltas = dict(zip(result["model"], result["delta_vs_best_static"]))

    assert deltas["clusterless-state-space-diffusion"] == -1.0
    assert deltas["clusterless-state-space-momentum"] == 0.0
    assert deltas["clusterless-state-space-jump"] == -2.0
    assert deltas["clusterless-state-space-imm"] == 1.0
    assert result["best_static_heldout_log_likelihood"].notna().all()


def test_add_relative_metrics_keeps_nan_when_no_static_baseline_is_present():
    rows = pd.DataFrame(
        {
            "session": ["s1"],
            "event_index": [7],
            "model": ["sorted-spike-state-space-imm"],
            "heldout_log_likelihood": [-6.0],
            "test_spikes": [2],
        }
    )

    result = _add_relative_metrics(rows)

    assert result["best_static_heldout_log_likelihood"].isna().all()
    assert result["delta_vs_best_static"].isna().all()


def test_add_relative_metrics_does_not_mix_exact_and_truncated_static_baselines():
    rows = pd.DataFrame(
        {
            "session": ["s1", "s1", "s1", "s1", "s1"],
            "event_index": [7, 7, 7, 7, 7],
            "model": ["random", "stationary", "diffusion", "momentum", "imm"],
            "heldout_log_likelihood": [-10.0, -9.0, -2.0, -4.0, -8.0],
            "test_spikes": [2, 2, 2, 2, 2],
            "diagnostic_candidate_evidence_support": [
                np.nan,
                np.nan,
                TRUNCATED_EVIDENCE_SUPPORT,
                TRUNCATED_EVIDENCE_SUPPORT,
                TRUNCATED_EVIDENCE_SUPPORT,
            ],
        }
    )

    result = _add_relative_metrics(rows)
    by_model = result.set_index("model")

    assert by_model["best_static_heldout_log_likelihood"].eq(-9.0).all()
    assert by_model.loc["random", "delta_vs_best_static"] == -1.0
    assert by_model.loc["stationary", "delta_vs_best_static"] == 0.0
    assert np.isnan(by_model.loc["diffusion", "delta_vs_best_static"])
    assert np.isnan(by_model.loc["momentum", "delta_vs_best_static"])
    assert np.isnan(by_model.loc["imm", "delta_vs_best_static"])
    assert by_model.loc["diffusion", "lower_bound_delta_vs_best_static"] == 7.0
    assert by_model.loc["momentum", "lower_bound_delta_vs_best_static"] == 5.0
    assert by_model.loc["imm", "lower_bound_delta_vs_best_static"] == 1.0
    assert by_model["best_static_truncated_lower_bound_heldout_log_likelihood"].eq(-2.0).all()
    assert by_model.loc["diffusion", "delta_vs_best_static_truncated_lower_bound"] == 0.0
    assert by_model.loc["momentum", "delta_vs_best_static_truncated_lower_bound"] == -2.0
    assert by_model.loc["imm", "delta_vs_best_static_truncated_lower_bound"] == -6.0


def test_benchmark_summary_separates_exact_and_truncated_support():
    rows = pd.DataFrame(
        {
            "session": ["s1", "s1", "s1"],
            "event_index": [7, 7, 7],
            "model": ["random", "stationary", "diffusion"],
            "heldout_log_likelihood": [-10.0, -9.0, -2.0],
            "test_spikes": [2, 2, 2],
            "diagnostic_candidate_evidence_support": [
                np.nan,
                np.nan,
                TRUNCATED_EVIDENCE_SUPPORT,
            ],
        }
    )

    summary = BenchmarkResult(_add_relative_metrics(rows)).summary().set_index("model")

    assert summary.loc["random", "evidence_comparable"]
    assert summary.loc["stationary", "evidence_comparable"]
    assert not summary.loc["diffusion", "evidence_comparable"]
    assert summary.loc["diffusion", "evidence_support"] == TRUNCATED_EVIDENCE_SUPPORT
    assert np.isnan(summary.loc["diffusion", "mean_delta_vs_best_static"])
    assert summary.loc["diffusion", "mean_lower_bound_delta_vs_best_static"] == 7.0
