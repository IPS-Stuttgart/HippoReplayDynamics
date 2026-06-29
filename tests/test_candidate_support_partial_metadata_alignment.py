from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from candidate_support_convergence import best_model_agreement, evidence_delta_summary  # noqa: E402


def _single_model_run(label: str, seed_value: object, log_evidence: float) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "run_label": label,
                "session": "Rat1/Open1",
                "event_index": 0,
                "benchmark_random_seed": seed_value,
                "model": "sorted-spike-state-space-imm",
                "log_evidence": log_evidence,
                "evidence_support": "exact_full_grid",
            }
        ]
    )


def _two_model_run(label: str, seed_value: object) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "run_label": label,
                "session": "Rat1/Open1",
                "event_index": 0,
                "benchmark_random_seed": seed_value,
                "model": "sorted-spike-state-space-imm",
                "log_evidence": 5.0,
                "evidence_support": "exact_full_grid",
            },
            {
                "run_label": label,
                "session": "Rat1/Open1",
                "event_index": 0,
                "benchmark_random_seed": seed_value,
                "model": "sorted-spike-state-space-diffusion",
                "log_evidence": 0.0,
                "evidence_support": "exact_full_grid",
            },
        ]
    )


def test_evidence_delta_summary_falls_back_when_optional_metadata_missing_in_one_run():
    seeded = _single_model_run("seeded", 7, 0.0)
    legacy = _single_model_run("legacy", pd.NA, 2.0)

    summary = evidence_delta_summary([seeded, legacy])

    assert summary.shape[0] == 1
    row = summary.iloc[0]
    assert int(row["events"]) == 1
    assert float(row["mean_delta_b_minus_a"]) == pytest.approx(2.0)
    assert float(row["max_abs_delta"]) == pytest.approx(2.0)


def test_best_model_agreement_falls_back_when_optional_metadata_missing_in_one_run():
    seeded = _two_model_run("seeded", 7)
    legacy = _two_model_run("legacy", pd.NA)

    agreement = best_model_agreement([seeded, legacy])

    assert agreement.shape[0] == 1
    row = agreement.iloc[0]
    assert int(row["events"]) == 1
    assert int(row["best_model_agreements"]) == 1
    assert int(row["best_model_disagreements"]) == 0
    assert float(row["best_model_agreement_fraction"]) == pytest.approx(1.0)
