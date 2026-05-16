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
    assert not _is_best_static_baseline_model("imm")
    assert not _is_best_static_baseline_model("sorted-spike-state-space-imm")
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
