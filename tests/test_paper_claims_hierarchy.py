import pandas as pd

from scripts.make_paper_claims import PaperClaimConfig, build_paper_claim_tables


def test_paper_claim_tables_include_leave_one_rat_out_summary(tmp_path):
    scores = pd.DataFrame(
        [
            _score("Rat1/Open1", 0, "sorted-spike-state-space-momentum", 4.0),
            _score("Rat1/Open1", 0, "sorted-spike-state-space-diffusion", 1.0),
            _score("Rat2/Open1", 1, "sorted-spike-state-space-momentum", 1.0),
            _score("Rat2/Open1", 1, "sorted-spike-state-space-diffusion", 2.0),
            _score("Rat3/Open1", 2, "sorted-spike-state-space-momentum", 5.0),
            _score("Rat3/Open1", 2, "sorted-spike-state-space-diffusion", 3.0),
        ]
    )

    tables = build_paper_claim_tables(
        scores,
        PaperClaimConfig(n_bootstrap=10, random_seed=1),
    )

    assert not tables.leave_one_rat_out_summary.empty
    assert set(tables.leave_one_rat_out_summary["left_out_rat"]) == {"Rat1", "Rat2", "Rat3"}

    output = tmp_path / "claims"
    tables.write(output)
    assert (output / "paper_claim_leave_one_rat_out_summary.csv").exists()


def _score(session: str, event_index: int, model: str, value: float) -> dict[str, object]:
    return {
        "status": "success",
        "session": session,
        "event_index": event_index,
        "model": model,
        "heldout_log_likelihood": value,
        "evidence_support": "exact_full_grid",
        "evidence_comparable": True,
    }
