from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from candidate_support_convergence import write_candidate_support_convergence  # noqa: E402


def _write_scores(root: Path, rows: list[dict[str, object]]) -> None:
    root.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(root / "event_model_evidence.csv", index=False)


def _row(event_index: int, model: str, log_evidence: float, top_k: int | None = None) -> dict[str, object]:
    row = {
        "status": "success",
        "session": "RatX/OpenY",
        "event_index": event_index,
        "model": model,
        "requested_model": model,
        "model_family": "trajectory" if model.endswith(("diffusion", "momentum", "imm")) else "nontrajectory",
        "log_evidence": log_evidence,
        "n_time": 4,
        "n_spikes": 9,
        "runtime_s": 0.1,
        "error": "",
    }
    if top_k is not None and model.endswith("momentum"):
        row["diagnostic_state_space_momentum_candidate_top_k"] = top_k
        row["diagnostic_state_space_momentum_evidence_support"] = "truncated_full_grid"
    return row


def test_candidate_support_convergence_passes_stable_runs(tmp_path):
    run64 = tmp_path / "run64"
    run128 = tmp_path / "run128"
    _write_scores(
        run64,
        [
            _row(1, "sorted-spike-state-space-diffusion", -9.0),
            _row(1, "sorted-spike-state-space-momentum", -8.0, top_k=64),
            _row(2, "sorted-spike-state-space-diffusion", -7.0),
            _row(2, "sorted-spike-state-space-momentum", -6.0, top_k=64),
        ],
    )
    _write_scores(
        run128,
        [
            _row(1, "sorted-spike-state-space-diffusion", -9.0),
            _row(1, "sorted-spike-state-space-momentum", -8.05, top_k=128),
            _row(2, "sorted-spike-state-space-diffusion", -7.0),
            _row(2, "sorted-spike-state-space-momentum", -5.95, top_k=128),
        ],
    )

    out = tmp_path / "out"
    outputs = write_candidate_support_convergence(
        [run64, run128],
        out,
        labels=["k64", "k128"],
        delta_tolerance=1.0,
        agreement_threshold=0.95,
    )

    delta = pd.read_csv(out / "candidate_support_delta_summary.csv")
    agreement = pd.read_csv(out / "candidate_support_best_model_agreement.csv")
    warnings = (out / "candidate_support_convergence_warnings.txt").read_text(encoding="utf-8")

    momentum = delta[delta["model"].eq("sorted-spike-state-space-momentum")].iloc[0]
    assert momentum["events"] == 2
    assert float(momentum["mean_abs_delta"]) == pytest.approx(0.05)
    assert float(agreement.loc[0, "best_model_agreement_fraction"]) == pytest.approx(1.0)
    assert "passed" in warnings
    assert outputs["warnings"] == warnings


def test_candidate_support_convergence_warns_on_ranking_instability(tmp_path):
    run64 = tmp_path / "run64"
    run128 = tmp_path / "run128"
    _write_scores(
        run64,
        [
            _row(1, "sorted-spike-state-space-diffusion", -8.0),
            _row(1, "sorted-spike-state-space-momentum", -9.0, top_k=64),
        ],
    )
    _write_scores(
        run128,
        [
            _row(1, "sorted-spike-state-space-diffusion", -8.0),
            _row(1, "sorted-spike-state-space-momentum", -7.0, top_k=128),
        ],
    )

    out = tmp_path / "out"
    write_candidate_support_convergence(
        [run64, run128],
        out,
        labels=["k64", "k128"],
        delta_tolerance=0.5,
        agreement_threshold=0.95,
    )

    agreement = pd.read_csv(out / "candidate_support_best_model_agreement.csv")
    warnings = (out / "candidate_support_convergence_warnings.txt").read_text(encoding="utf-8")
    assert float(agreement.loc[0, "best_model_agreement_fraction"]) == pytest.approx(0.0)
    assert "potential instability" in warnings
    assert "best-model agreement" in warnings
