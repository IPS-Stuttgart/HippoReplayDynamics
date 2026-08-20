from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from candidate_support_convergence import (  # noqa: E402
    best_model_agreement,
    evidence_delta_summary,
    load_candidate_support_run,
)


def _run(label: str, rows: list[tuple[str, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "run_label": label,
                "session": "Rat1/Open1",
                "event_index": 0,
                "model": model,
                "log_evidence": log_evidence,
                "evidence_support": "exact_full_grid",
            }
            for model, log_evidence in rows
        ]
    )


def test_evidence_delta_summary_rejects_duplicate_event_model_rows():
    model = "sorted-spike-state-space-momentum"
    left = _run("k64", [(model, -8.0), (model, -80.0)])
    right = _run("k128", [(model, -7.0)])

    with pytest.raises(ValueError, match="duplicate rows"):
        evidence_delta_summary([left, right])


def test_best_model_agreement_rejects_duplicate_event_model_rows():
    diffusion = "sorted-spike-state-space-diffusion"
    momentum = "sorted-spike-state-space-momentum"
    left = _run("k64", [(diffusion, -8.0), (momentum, -7.0), (momentum, -70.0)])
    right = _run("k128", [(diffusion, -8.0), (momentum, -7.0)])

    with pytest.raises(ValueError, match="duplicate rows"):
        best_model_agreement([left, right])


def test_load_candidate_support_run_rejects_all_failed_rows(tmp_path):
    score_file = tmp_path / "event_model_evidence.csv"
    pd.DataFrame(
        [
            {
                "status": "failed",
                "session": "Rat1/Open1",
                "event_index": 0,
                "model": "sorted-spike-state-space-momentum",
                "requested_model": "sorted-spike-state-space-momentum",
                "model_family": "trajectory",
                "log_evidence": -8.0,
                "n_time": 4,
                "n_spikes": 9,
                "runtime_s": 0.1,
                "error": "synthetic failure",
            }
        ]
    ).to_csv(score_file, index=False)

    with pytest.raises(ValueError, match="no successful rows"):
        load_candidate_support_run(score_file)
