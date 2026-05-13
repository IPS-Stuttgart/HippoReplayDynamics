from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path("scripts").resolve()))
from aggregate_all_session_model_evidence import (
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
