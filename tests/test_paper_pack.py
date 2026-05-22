from __future__ import annotations

import json

import pandas as pd
import pytest

from scripts.build_paper_pack import build_paper_pack
from scripts.make_paper_claims import PaperClaimConfig, build_paper_claim_tables


def test_make_paper_claims_requires_evidence_support_metadata():
    scores = pd.DataFrame(
        [
            {
                "session": "RatX/OpenY",
                "event_index": 0,
                "model": "sorted-spike-state-space-diffusion",
                "heldout_log_likelihood": 1.0,
            },
            {
                "session": "RatX/OpenY",
                "event_index": 0,
                "model": "sorted-spike-state-space-momentum",
                "heldout_log_likelihood": 2.0,
            },
        ]
    )

    with pytest.raises(KeyError, match="evidence-support metadata"):
        build_paper_claim_tables(scores, PaperClaimConfig())


def test_build_paper_pack_writes_claims_and_recovery_diagnostics(tmp_path):
    claim_scores = pd.DataFrame(
        [
            {
                "status": "success",
                "session": "RatX/OpenY",
                "event_index": 0,
                "model": "sorted-spike-state-space-diffusion",
                "heldout_log_likelihood": 1.0,
                "evidence_support": "exact_full_grid",
                "evidence_comparable": True,
            },
            {
                "status": "success",
                "session": "RatX/OpenY",
                "event_index": 0,
                "model": "sorted-spike-state-space-momentum",
                "heldout_log_likelihood": 2.0,
                "evidence_support": "exact_full_grid",
                "evidence_comparable": True,
            },
        ]
    )
    claim_path = tmp_path / "claim_scores.csv"
    claim_scores.to_csv(claim_path, index=False)
    recovery_scores = pd.DataFrame(
        [
            {
                "status": "success",
                "session": "RatX/OpenY",
                "event_index": 0,
                "true_model": "momentum",
                "expected_model": "sorted-spike-state-space-momentum",
                "model": "sorted-spike-state-space-diffusion",
                "log_evidence": 1.0,
                "evidence_support": "exact_full_grid",
                "evidence_comparable": True,
            },
            {
                "status": "success",
                "session": "RatX/OpenY",
                "event_index": 0,
                "true_model": "momentum",
                "expected_model": "sorted-spike-state-space-momentum",
                "model": "sorted-spike-state-space-momentum",
                "log_evidence": 2.0,
                "evidence_support": "truncated_full_grid",
                "evidence_comparable": False,
            },
        ]
    )
    recovery_dir = tmp_path / "recovery"
    recovery_dir.mkdir()
    recovery_scores.to_csv(recovery_dir / "simulation_recovery_event_scores.csv", index=False)

    output = tmp_path / "paper-pack"
    build_paper_pack(
        output=output,
        scores=[claim_path],
        simulation_recovery_scores=[recovery_dir],
        primary_model="momentum",
        baseline_model="diffusion",
        value_column="heldout_log_likelihood",
        n_bootstrap=10,
        random_seed=1,
    )

    assert (output / "model-claims" / "paper_claim_summary.csv").exists()
    assert (output / "simulation-recovery-diagnostics" / "simulation_recovery_diagnostic_summary.csv").exists()
    manifest = json.loads((output / "paper_pack_manifest.json").read_text(encoding="utf-8"))
    assert "model_claims" in manifest["outputs"]
    assert "simulation_recovery_diagnostics" in manifest["outputs"]
