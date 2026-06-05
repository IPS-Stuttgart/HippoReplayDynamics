import argparse
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_model_evidence.py"
_SPEC = importlib.util.spec_from_file_location("benchmark_model_evidence", _SCRIPT)
assert _SPEC is not None
benchmark_model_evidence = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(benchmark_model_evidence)

_events = benchmark_model_evidence._events
_clusterless_mark_config = benchmark_model_evidence._clusterless_mark_config
_add_evidence_columns = benchmark_model_evidence._add_evidence_columns
_family = benchmark_model_evidence._family
_models = benchmark_model_evidence._models
_score = benchmark_model_evidence._score


class _SessionStub:
    ripple_count = 10

    @staticmethod
    def ripple_indices_in_run():
        return np.array([2, 4, 6, 8], dtype=int)


def test_model_evidence_accepts_sorted_spike_state_space_models():
    args = argparse.Namespace(
        models=(
            "sorted-spike-state-space-diffusion "
            "sorted-spike-state-space-first-order-imm "
            "sorted-spike-state-space-momentum-exact-sparse "
            "sorted-spike-state-space-trajectory-imm-exact-sparse "
            "sorted-spike-state-space-momentum "
            "sorted-spike-state-space-imm"
        ),
        candidate_top_k=64,
        stationary_sigma_cm=2.0,
        diffusion_sigma_cm=12.0,
        momentum_sigma_cm=12.0,
        velocity_decay=0.95,
        mode_stickiness=0.94,
        state_space_stationary_sigma_cm=1.5,
        state_space_diffusion_sigma_cm_sqrt_s=42.0,
        state_space_max_step_sigma=3.0,
        state_space_imm_mode_stickiness=0.91,
        state_space_momentum_sigma_cm_sqrt_s=43.0,
        state_space_momentum_initial_sigma_cm_sqrt_s=44.0,
        state_space_momentum_velocity_decay=0.8,
        state_space_momentum_candidate_top_k=17,
        state_space_momentum_predicted_candidate_top_k=5,
    )

    models = _models(args)

    assert list(models) == [
        "sorted-spike-state-space-diffusion",
        "sorted-spike-state-space-first-order-imm",
        "sorted-spike-state-space-momentum-exact-sparse",
        "sorted-spike-state-space-trajectory-imm-exact-sparse",
        "sorted-spike-state-space-momentum",
        "sorted-spike-state-space-imm",
    ]
    assert models["sorted-spike-state-space-diffusion"].name == "sorted-spike-state-space-diffusion"
    assert models["sorted-spike-state-space-first-order-imm"].name == "sorted-spike-state-space-first-order-imm"
    assert models["sorted-spike-state-space-momentum-exact-sparse"].name == "sorted-spike-state-space-momentum-exact-sparse"
    assert models["sorted-spike-state-space-trajectory-imm-exact-sparse"].name == (
        "sorted-spike-state-space-trajectory-imm-exact-sparse"
    )
    assert models["sorted-spike-state-space-momentum"].name == "sorted-spike-state-space-momentum"
    assert models["sorted-spike-state-space-imm"].name == "sorted-spike-state-space-imm"
    assert models["sorted-spike-state-space-diffusion"].config.diffusion_sigma_cm_sqrt_s == 42.0
    assert models["sorted-spike-state-space-first-order-imm"].config.imm_mode_stickiness == 0.91
    assert models["sorted-spike-state-space-momentum"].config.momentum_sigma_cm_sqrt_s == 43.0
    assert models["sorted-spike-state-space-momentum"].config.momentum_initial_sigma_cm_sqrt_s == 44.0
    assert models["sorted-spike-state-space-momentum"].config.momentum_velocity_decay == 0.8
    assert models["sorted-spike-state-space-momentum"].config.momentum_candidate_top_k == 17
    assert models["sorted-spike-state-space-momentum"].config.momentum_predicted_candidate_top_k == 5
    assert models["sorted-spike-state-space-imm"].config.imm_mode_stickiness == 0.91


def test_model_evidence_accepts_clusterless_state_space_models():
    args = argparse.Namespace(
        models=(
            "clusterless-state-space-diffusion "
            "clusterless-state-space-first-order-imm "
            "clusterless-state-space-momentum-exact-sparse "
            "clusterless-state-space-trajectory-imm-exact-sparse "
            "clusterless-state-space-momentum "
            "clusterless-state-space-imm"
        ),
        candidate_top_k=64,
        stationary_sigma_cm=2.0,
        diffusion_sigma_cm=12.0,
        momentum_sigma_cm=12.0,
        velocity_decay=0.95,
        mode_stickiness=0.94,
        state_space_stationary_sigma_cm=1.5,
        state_space_diffusion_sigma_cm_sqrt_s=42.0,
        state_space_max_step_sigma=3.0,
        state_space_imm_mode_stickiness=0.91,
        state_space_momentum_sigma_cm_sqrt_s=43.0,
        state_space_momentum_initial_sigma_cm_sqrt_s=44.0,
        state_space_momentum_velocity_decay=0.8,
        state_space_momentum_candidate_top_k=17,
        state_space_momentum_predicted_candidate_top_k=5,
    )

    models = _models(args)

    assert list(models) == [
        "clusterless-state-space-diffusion",
        "clusterless-state-space-first-order-imm",
        "clusterless-state-space-momentum-exact-sparse",
        "clusterless-state-space-trajectory-imm-exact-sparse",
        "clusterless-state-space-momentum",
        "clusterless-state-space-imm",
    ]
    assert models["clusterless-state-space-diffusion"].name == "clusterless-state-space-diffusion"
    assert models["clusterless-state-space-first-order-imm"].name == "clusterless-state-space-first-order-imm"
    assert models["clusterless-state-space-momentum-exact-sparse"].name == (
        "clusterless-state-space-momentum-exact-sparse"
    )
    assert models["clusterless-state-space-trajectory-imm-exact-sparse"].name == (
        "clusterless-state-space-trajectory-imm-exact-sparse"
    )
    assert models["clusterless-state-space-momentum"].name == "clusterless-state-space-momentum"
    assert models["clusterless-state-space-imm"].name == "clusterless-state-space-imm"
    assert models["clusterless-state-space-diffusion"].config.diffusion_sigma_cm_sqrt_s == 42.0
    assert models["clusterless-state-space-first-order-imm"].config.imm_mode_stickiness == 0.91
    assert models["clusterless-state-space-momentum-exact-sparse"].config.momentum_sigma_cm_sqrt_s == 43.0
    assert models["clusterless-state-space-momentum"].config.momentum_sigma_cm_sqrt_s == 43.0
    assert models["clusterless-state-space-momentum"].config.momentum_predicted_candidate_top_k == 5


def test_model_evidence_clusterless_config_records_rate_floor():
    args = argparse.Namespace(
        bin_size_cm=6.0,
        smoothing_sigma_bins=2.0,
        min_speed_cm_s=5.0,
        clusterless_mark_smoothing_sigma_bins=1.5,
        clusterless_mark_prior_count=0.25,
        clusterless_mark_variance_floor=0.75,
        clusterless_rate_floor_hz=1e-3,
        clusterless_mark_likelihood="local-kde",
        clusterless_mark_kde_bandwidth=2.5,
        clusterless_mark_kde_spatial_sigma_bins=3.5,
        clusterless_mark_kde_max_neighbors=17,
    )

    config = _clusterless_mark_config(args)

    assert config.encoding is not None
    assert config.encoding.bin_size_cm == 6.0
    assert config.mark_smoothing_sigma_bins == 1.5
    assert config.mark_prior_count == 0.25
    assert config.mark_variance_floor == 0.75
    assert config.rate_floor_hz == 1e-3
    assert config.mark_likelihood == "local-kde"
    assert config.mark_kde_bandwidth == 2.5
    assert config.mark_kde_spatial_sigma_bins == 3.5
    assert config.mark_kde_max_neighbors == 17


def test_model_evidence_marks_clusterless_rows_unsupported_without_spike_marks(monkeypatch):
    args = argparse.Namespace(
        dataset_root="data",
        session="Rat1/Open1",
        events="0",
        max_events=None,
        models="clusterless-state-space-momentum-exact-sparse",
        bin_size_cm=6.0,
        smoothing_sigma_bins=2.0,
        min_speed_cm_s=5.0,
        time_bin_s=0.004,
        spike_rate_scale=2.0,
        emission_likelihood_temperature=0.3,
        emission_negative_binomial_overdispersion=0.0,
        clusterless_mark_smoothing_sigma_bins=1.0,
        clusterless_mark_prior_count=1.0,
        clusterless_mark_variance_floor=1.0,
        clusterless_rate_floor_hz=1e-4,
        clusterless_mark_likelihood="local-kde",
        clusterless_mark_kde_bandwidth=None,
        clusterless_mark_kde_spatial_sigma_bins=None,
        clusterless_mark_kde_max_neighbors=256,
        state_space_imm_mode_stickiness=0.95,
        state_space_common_support_top_k=0,
    )
    session = SimpleNamespace(session_id="Rat1/Open1", ripple_count=1)
    encoding = SimpleNamespace(
        bin_centers=np.zeros((2, 2), dtype=float),
        occupancy_s=np.ones(2, dtype=float),
    )
    sorted_emissions = SimpleNamespace(
        n_time=3,
        n_spikes=5,
        log_likelihood=np.zeros((3, 2), dtype=float),
    )

    monkeypatch.setattr(benchmark_model_evidence, "_session_path", lambda root, session: Path("."))
    monkeypatch.setattr(benchmark_model_evidence, "_check_session", lambda path: None)
    monkeypatch.setattr(benchmark_model_evidence, "load_replay_session", lambda path: session)
    monkeypatch.setattr(benchmark_model_evidence, "fit_place_field_encoding", lambda session, config: encoding)
    monkeypatch.setattr(
        benchmark_model_evidence,
        "_models",
        lambda args, session=None: {
            "clusterless-state-space-momentum-exact-sparse": (
                benchmark_model_evidence.ClusterlessStateSpaceReplayModel(
                    mode="momentum-exact-sparse"
                )
            )
        },
    )
    monkeypatch.setattr(
        benchmark_model_evidence,
        "fit_clusterless_mark_encoding",
        lambda session, config: (_ for _ in ()).throw(
            ValueError("Session does not contain spike marks for clusterless encoding.")
        ),
    )
    monkeypatch.setattr(
        benchmark_model_evidence,
        "build_emissions",
        lambda session, encoding, event_id, config: sorted_emissions,
    )

    scores = _score(args)

    assert scores["status"].tolist() == ["unsupported"]
    assert scores["model"].tolist() == ["clusterless-state-space-momentum-exact-sparse"]
    assert scores["evidence_support"].tolist() == ["not_scored"]
    assert scores["evidence_comparable"].tolist() == [False]
    assert "spike marks" in str(scores.loc[0, "error"])
    assert scores.loc[0, "n_time"] == 3
    assert scores.loc[0, "n_spikes"] == 5


def test_model_evidence_classifies_state_space_families():
    assert _family("sorted-spike-state-space-stationary") == "nontrajectory"
    assert _family("sorted-spike-state-space-diffusion") == "trajectory"
    assert _family("sorted-spike-state-space-fragmented") == "trajectory"
    assert _family("sorted-spike-state-space-momentum") == "trajectory"
    assert _family("sorted-spike-state-space-trajectory-imm-exact-sparse") == "trajectory"
    assert _family("sorted-spike-state-space-first-order-imm") == "trajectory"
    assert _family("sorted-spike-state-space-imm") == "trajectory"
    assert _family("clusterless-state-space-stationary") == "nontrajectory"
    assert _family("clusterless-state-space-diffusion") == "trajectory"
    assert _family("clusterless-state-space-fragmented") == "trajectory"
    assert _family("clusterless-state-space-momentum") == "trajectory"
    assert _family("clusterless-state-space-momentum-exact-sparse") == "trajectory"
    assert _family("clusterless-state-space-trajectory-imm-exact-sparse") == "trajectory"
    assert _family("clusterless-state-space-first-order-imm") == "trajectory"
    assert _family("clusterless-state-space-imm") == "trajectory"


def test_model_evidence_run_event_selection_uses_session_event_ids():
    assert _events("run", _SessionStub()) == [2, 4, 6, 8]
    assert _events("run:1-2", _SessionStub()) == [4, 6]


def test_model_evidence_excludes_truncated_lower_bounds_from_exact_probabilities():
    scored = _add_evidence_columns(
        pd.DataFrame(
            [
                _score_row("random", 0.0),
                _score_row("stationary", -2.0),
                _score_row(
                    "momentum",
                    100.0,
                    diagnostic_candidate_evidence_support="truncated_full_grid",
                ),
            ]
        )
    )

    random = scored[scored["model"] == "random"].iloc[0]
    momentum = scored[scored["model"] == "momentum"].iloc[0]

    assert bool(random["is_best_model"])
    assert random["best_model"] == "random"
    assert not bool(momentum["evidence_comparable"])
    assert not bool(momentum["is_best_model"])
    assert pd.isna(momentum["relative_log_evidence"])
    assert pd.isna(momentum["model_probability"])
    assert momentum["best_truncated_lower_bound_model"] == "momentum"
    assert bool(momentum["is_best_truncated_lower_bound"])
    assert momentum["truncated_relative_log_evidence"] == 0.0


def _score_row(model: str, log_evidence: float, **extra: object) -> dict[str, object]:
    row = {
        "status": "success",
        "session": "RatX/OpenY",
        "event_index": 0,
        "model": model,
        "requested_model": model,
        "model_family": _family(model),
        "log_evidence": log_evidence,
        "n_time": 3,
        "n_spikes": 5,
        "runtime_s": 0.0,
        "error": "",
    }
    row.update(extra)
    return row
