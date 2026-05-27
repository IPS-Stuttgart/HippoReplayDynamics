from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path("scripts").resolve()))
from select_momentum_confidence_threshold import main  # noqa: E402


POSITIVE_MODEL = "sorted-spike-state-space-momentum-exact-sparse"
REFERENCE_MODEL = "sorted-spike-state-space-diffusion"


def test_stratum_scoped_thresholds_can_recover_more_confident_claims(tmp_path, monkeypatch):
    recovery_dir = tmp_path / "recovery"
    evidence_dir = tmp_path / "evidence"
    global_out = tmp_path / "global"
    stratum_out = tmp_path / "stratum"
    recovery_dir.mkdir()
    evidence_dir.mkdir()

    pd.DataFrame(
        [
            *_paired_rows("rec-temp-low", 0.5, 0, "diffusion", 3.0),
            *_paired_rows("rec-temp-low", 0.5, 1, "momentum", 6.0),
            *_paired_rows("rec-temp-high", 1.0, 0, "diffusion", 5.0),
            *_paired_rows("rec-temp-high", 1.0, 1, "momentum", 8.0),
        ]
    ).to_csv(recovery_dir / "simulation_recovery_event_scores.csv", index=False)
    pd.DataFrame(
        [
            *_paired_rows("evidence-temp-low", 0.5, 0, None, 4.5),
            *_paired_rows("evidence-temp-high", 1.0, 0, None, 7.0),
        ]
    ).to_csv(evidence_dir / "event_model_evidence.csv", index=False)

    _run_selector(
        monkeypatch,
        recovery_dir=recovery_dir,
        evidence_dir=evidence_dir,
        output=global_out,
        extra_args=(),
    )
    _run_selector(
        monkeypatch,
        recovery_dir=recovery_dir,
        evidence_dir=evidence_dir,
        output=stratum_out,
        extra_args=("--threshold-scope", "stratum", "--stratify-columns", "emission_likelihood_temperature"),
    )

    global_summary = pd.read_csv(global_out / "momentum_confidence_threshold_evidence_summary.csv")
    stratum_summary = pd.read_csv(stratum_out / "momentum_confidence_threshold_evidence_summary.csv")
    stratum_rows = pd.read_csv(stratum_out / "momentum_confidence_threshold_evidence_by_stratum.csv")

    assert global_summary["positive_model_claims"].tolist() == [1]
    assert stratum_summary["positive_model_claims"].tolist() == [2]
    assert sorted(stratum_rows["margin_threshold"].tolist()) == [4.0, 6.0]
    assert sorted(stratum_rows["matrix_id"].tolist()) == ["evidence-temp-high", "evidence-temp-low"]


def _run_selector(
    monkeypatch,
    *,
    recovery_dir: Path,
    evidence_dir: Path,
    output: Path,
    extra_args: tuple[str, ...],
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "select_momentum_confidence_threshold.py",
            "--recovery-scores",
            str(recovery_dir),
            "--evidence-scores",
            str(evidence_dir),
            "--output",
            str(output),
            "--thresholds",
            "0 4 6",
            "--max-false-positive-claims",
            "0",
            *extra_args,
        ],
    )

    assert main() == 0


def _paired_rows(
    matrix_id: str,
    temperature: float,
    event_index: int,
    true_model: str | None,
    positive_minus_reference: float,
) -> list[dict[str, object]]:
    common = {
        "matrix_id": matrix_id,
        "session": "Rat1/Open1",
        "event_index": event_index,
        "emission_likelihood_temperature": temperature,
        "status": "success",
        "evidence_comparable": True,
    }
    if true_model is not None:
        common["true_model"] = true_model
    return [
        {
            **common,
            "model": REFERENCE_MODEL,
            "log_evidence": 0.0,
        },
        {
            **common,
            "model": POSITIVE_MODEL,
            "log_evidence": positive_minus_reference,
        },
    ]
