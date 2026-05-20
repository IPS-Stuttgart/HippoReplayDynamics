from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from candidate_support_convergence import infer_run_label, write_candidate_support_convergence  # noqa: E402


def _write_scores(root: Path, rows: list[dict[str, object]]) -> None:
    root.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(root / "event_model_evidence.csv", index=False)


def _row(
    event_index: int,
    model: str,
    log_evidence: float,
    top_k: int | None = None,
    predicted_top_k: int | None = None,
) -> dict[str, object]:
    row = {
        "status": "success",
        "session": "RatX/OpenY",
        "event_index": event_index,
        "model": model,
        "requested_model": model,
        "model_family": "trajectory",
        "log_evidence": log_evidence,
        "n_time": 4,
        "n_spikes": 9,
        "runtime_s": 0.1,
        "error": "",
    }
    if top_k is not None:
        row["diagnostic_state_space_momentum_candidate_top_k"] = top_k
        row["diagnostic_state_space_momentum_evidence_support"] = "truncated_full_grid"
    if predicted_top_k is not None:
        row["diagnostic_state_space_momentum_predicted_candidate_top_k"] = predicted_top_k
    return row


def test_candidate_support_convergence_infers_predicted_support_label():
    frame = pd.DataFrame([_row(1, "sorted-spike-state-space-momentum", -8.0, 128, 4)])
    assert infer_run_label(frame, "fallback") == "top_k=128,pred_k=4"


def test_candidate_support_convergence_reports_stable_runs(tmp_path):
    run64 = tmp_path / "run64"
    run128 = tmp_path / "run128"
    model = "sorted-spike-state-space-momentum"
    _write_scores(run64, [_row(1, model, -8.0, 64), _row(2, model, -6.0, 64)])
    _write_scores(run128, [_row(1, model, -8.05, 128), _row(2, model, -5.95, 128)])

    out = tmp_path / "out"
    result = write_candidate_support_convergence([run64, run128], out, labels=["k64", "k128"])

    delta = pd.read_csv(out / "candidate_support_delta_summary.csv")
    agreement = pd.read_csv(out / "candidate_support_best_model_agreement.csv")
    text = (out / "candidate_support_convergence_warnings.txt").read_text(encoding="utf-8")
    assert delta.loc[0, "events"] == 2
    assert float(delta.loc[0, "mean_abs_delta"]) == pytest.approx(0.05)
    assert float(agreement.loc[0, "best_model_agreement_fraction"]) == pytest.approx(1.0)
    assert "passed" in text
    assert result["warnings"] == text


def test_candidate_support_convergence_warns_when_best_model_changes(tmp_path):
    run64 = tmp_path / "run64"
    run128 = tmp_path / "run128"
    diffusion = "sorted-spike-state-space-diffusion"
    momentum = "sorted-spike-state-space-momentum"
    _write_scores(run64, [_row(1, diffusion, -8.0), _row(1, momentum, -9.0, 64)])
    _write_scores(run128, [_row(1, diffusion, -8.0), _row(1, momentum, -7.0, 128)])

    out = tmp_path / "out"
    write_candidate_support_convergence(
        [run64, run128], out, labels=["k64", "k128"], delta_tolerance=0.5
    )

    agreement = pd.read_csv(out / "candidate_support_best_model_agreement.csv")
    text = (out / "candidate_support_convergence_warnings.txt").read_text(encoding="utf-8")
    assert float(agreement.loc[0, "best_model_agreement_fraction"]) == pytest.approx(0.0)
    assert "potential" in text
