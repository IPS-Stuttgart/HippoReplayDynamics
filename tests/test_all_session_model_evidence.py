from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path("scripts").resolve()))
from aggregate_all_session_model_evidence import (
    _load_combined,
    random_effects_model_probabilities,
    session_best_model_counts,
    session_model_evidence_summary,
)


def test_all_session_model_evidence_workflow_exports_expected_outputs():
    workflow = Path(".github/workflows/model-evidence-all-sessions.yml").read_text(encoding="utf-8")

    assert "name: Benchmark replay model evidence all sessions" in workflow
    assert "Rat1/Open1 Rat1/Open2 Rat2/Open1 Rat2/Open2 Rat3/Open1 Rat3/Open2 Rat4/Open1 Rat4/Open2" in workflow
    assert "scripts/plan_model_evidence_event_shards.py" in workflow
    assert "scripts/aggregate_all_session_model_evidence.py" in workflow
    assert "spike_rate_scale:" in workflow
    assert "--spike-rate-scale" in workflow
    assert 'CLUSTERLESS_MARK_SMOOTHING_SIGMA_BINS: "1.0"' in workflow
    assert "--clusterless-mark-smoothing-sigma-bins" in workflow
    assert "all_sessions_model_evidence_summary.csv" in workflow
    assert "session_model_evidence_summary.csv" in workflow
    assert "random_effects_model_probabilities.csv" in workflow


def test_all_session_summary_helpers_group_by_session():
    rows = []
    for session, winner in (("Rat1/Open1", "diffusion"), ("Rat1/Open2", "momentum")):
        for event_index in (0, 1):
            for model in ("diffusion", "momentum"):
                rows.append(
                    {
                        "session": session,
                        "event_index": event_index,
                        "model": model,
                        "model_family": "trajectory",
                        "status": "success",
                        "log_evidence": 2.0 if model == winner else 1.0,
                        "relative_log_evidence": 0.0 if model == winner else -1.0,
                        "model_probability": 0.75 if model == winner else 0.25,
                        "is_best_model": model == winner,
                        "best_model": winner,
                        "best_trajectory_model": winner,
                        "best_nontrajectory_model": "",
                        "runtime_s": 0.1,
                    }
                )
    frame = pd.DataFrame(rows)

    session_summary = session_model_evidence_summary(frame)
    session_counts = session_best_model_counts(frame)
    random_effects = random_effects_model_probabilities(frame)

    assert set(session_summary["session"]) == {"Rat1/Open1", "Rat1/Open2"}
    assert set(session_counts["comparison"]) >= {"best_model", "best_trajectory_model"}
    assert set(random_effects["model"]) == {"diffusion", "momentum"}
    assert random_effects["random_effects_probability"].sum() == 1.0


def test_all_session_aggregation_rejects_mixed_run_settings(tmp_path: Path):
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    pd.DataFrame([_score_row(event_index=0, spike_rate_scale=1.0)]).to_csv(first, index=False)
    pd.DataFrame([_score_row(event_index=1, spike_rate_scale=2.0)]).to_csv(second, index=False)

    with pytest.raises(ValueError, match="spike_rate_scale"):
        _load_combined(str(tmp_path / "*.csv"))


def _score_row(*, event_index: int, spike_rate_scale: float = 1.0) -> dict[str, object]:
    return {
        "status": "success",
        "session": "RatX/OpenY",
        "event_index": event_index,
        "model": "diffusion",
        "requested_model": "diffusion",
        "model_family": "trajectory",
        "log_evidence": -1.0,
        "n_time": 3,
        "n_spikes": 5,
        "runtime_s": 0.0,
        "error": "",
        "bin_size_cm": 6.0,
        "smoothing_sigma_bins": 2.0,
        "min_speed_cm_s": 5.0,
        "time_bin_s": 0.003,
        "spike_rate_scale": spike_rate_scale,
        "clusterless_mark_smoothing_sigma_bins": 1.0,
        "clusterless_mark_prior_count": 1.0,
        "clusterless_mark_variance_floor": 1.0,
        "clusterless_rate_floor_hz": 1e-4,
    }
