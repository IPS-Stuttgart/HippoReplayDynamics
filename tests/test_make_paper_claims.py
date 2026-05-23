from __future__ import annotations

import pandas as pd

from scripts.make_paper_claims import (
    PaperClaimConfig,
    build_paper_claim_tables,
    load_score_tables,
)


def _row(session: str, event: int, model: str, value: float, support: str = "exact_full_grid") -> dict[str, object]:
    return {
        "session": session,
        "event_index": event,
        "model": model,
        "heldout_log_likelihood": value,
        "status": "success",
        "evidence_support": support,
    }


def test_paper_claim_tables_report_paired_session_heterogeneity(tmp_path):
    scores = pd.DataFrame(
        [
            _row("Rat1/Open1", 0, "sorted-spike-state-space-diffusion", 0.0),
            _row("Rat1/Open1", 0, "sorted-spike-state-space-momentum-exact-sparse", 2.0),
            _row("Rat1/Open1", 1, "sorted-spike-state-space-diffusion", 1.0),
            _row("Rat1/Open1", 1, "sorted-spike-state-space-momentum-exact-sparse", 0.0),
            _row("Rat2/Open1", 0, "sorted-spike-state-space-diffusion", 0.0),
            _row("Rat2/Open1", 0, "sorted-spike-state-space-momentum-exact-sparse", 3.0),
            _row("Rat2/Open1", 1, "sorted-spike-state-space-diffusion", 0.0),
            _row("Rat2/Open1", 1, "sorted-spike-state-space-momentum-exact-sparse", 4.0),
        ]
    )

    tables = build_paper_claim_tables(scores, PaperClaimConfig(n_bootstrap=100, random_seed=3))

    summary = tables.summary.iloc[0]
    assert summary["paired_events"] == 4
    assert summary["sessions"] == 2
    assert summary["apparent_primary_wins"] == 3
    assert summary["apparent_baseline_wins"] == 1
    assert summary["certified_primary_wins"] == 3
    assert summary["mean_delta_primary_minus_baseline"] == 2.0
    assert summary["primary_model"] == "sorted-spike-state-space-momentum-exact-sparse"
    assert summary["baseline_model"] == "sorted-spike-state-space-diffusion"

    by_session = tables.session_summary.set_index("group")
    assert by_session.loc["Rat1/Open1", "apparent_primary_wins"] == 1
    assert by_session.loc["Rat2/Open1", "apparent_primary_wins"] == 2

    output = tmp_path / "claims"
    tables.write(output)
    assert (output / "paper_claim_event_deltas.csv").exists()
    assert (output / "paper_claim_session_summary.csv").exists()
    assert (output / "paper_claim_summary.csv").exists()
    assert "Lower-bound-safe claim" in (output / "paper_claims.md").read_text(encoding="utf-8")


def test_lower_bound_primary_losses_are_nondecisive(tmp_path):
    scores = pd.DataFrame(
        [
            _row("Rat1/Open1", 0, "sorted-spike-state-space-diffusion", 0.0),
            _row("Rat1/Open1", 0, "sorted-spike-state-space-momentum", 1.0, "truncated_full_grid"),
            _row("Rat1/Open1", 1, "sorted-spike-state-space-diffusion", 2.0),
            _row("Rat1/Open1", 1, "sorted-spike-state-space-momentum", 1.0, "truncated_full_grid"),
        ]
    )

    tables = build_paper_claim_tables(
        scores,
        PaperClaimConfig(
            primary_model="sorted-spike-state-space-momentum",
            n_bootstrap=50,
            random_seed=4,
        ),
    )
    deltas = tables.event_deltas.sort_values("event_index")

    assert bool(deltas.iloc[0]["certified_primary_win"])
    assert deltas.iloc[0]["claim_category"] == "lower_bound_certified_primary_win"
    assert not bool(deltas.iloc[1]["strict_baseline_win"])
    assert bool(deltas.iloc[1]["nondecisive_due_to_primary_lower_bound"])
    assert deltas.iloc[1]["claim_category"] == "nondecisive_primary_lower_bound"

    summary = tables.summary.iloc[0]
    assert summary["certified_primary_wins"] == 1
    assert summary["strict_baseline_wins"] == 0
    assert summary["nondecisive_primary_lower_bound_events"] == 1


def test_load_score_tables_accepts_directory_with_event_scores(tmp_path):
    scores_dir = tmp_path / "run"
    scores_dir.mkdir()
    pd.DataFrame(
        [
            _row("Rat1/Open1", 0, "sorted-spike-state-space-diffusion", 0.0),
            _row("Rat1/Open1", 0, "sorted-spike-state-space-momentum", 1.0),
        ]
    ).to_csv(scores_dir / "event_scores.csv", index=False)

    loaded = load_score_tables([scores_dir])

    assert len(loaded) == 2
    assert loaded["source_score_file"].str.endswith("event_scores.csv").all()
