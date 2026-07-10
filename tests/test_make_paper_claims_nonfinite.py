from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.make_paper_claims import (
    PaperClaimConfig,
    build_paper_claim_tables,
    paired_event_deltas,
)


PRIMARY = "sorted-spike-state-space-momentum-exact-sparse"
BASELINE = "sorted-spike-state-space-diffusion"


def _row(event_index: int, model: str, value: object) -> dict[str, object]:
    return {
        "session": "Rat1/Open1",
        "event_index": event_index,
        "model": model,
        "heldout_log_likelihood": value,
        "status": "success",
        "evidence_support": "exact_full_grid",
        "evidence_comparable": True,
    }


def _scores_with_nonfinite_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _row(0, BASELINE, 1.0),
            _row(0, PRIMARY, 3.0),
            _row(1, BASELINE, 0.0),
            _row(1, PRIMARY, np.inf),
            _row(2, BASELINE, "-inf"),
            _row(2, PRIMARY, 4.0),
        ]
    )


def test_paired_event_deltas_excludes_nonfinite_score_rows() -> None:
    deltas = paired_event_deltas(
        _scores_with_nonfinite_rows(),
        primary_model=PRIMARY,
        baseline_model=BASELINE,
    )

    assert deltas["event_index"].tolist() == [0]
    assert deltas.loc[0, "delta_primary_minus_baseline"] == 2.0
    assert np.isfinite(deltas[["primary_value", "baseline_value", "delta_primary_minus_baseline"]]).all().all()


def test_paper_claim_summary_uses_only_finite_pairs() -> None:
    tables = build_paper_claim_tables(
        _scores_with_nonfinite_rows(),
        PaperClaimConfig(n_bootstrap=20, random_seed=7),
    )

    summary = tables.summary.iloc[0]
    assert summary["paired_events"] == 1
    assert summary["mean_delta_primary_minus_baseline"] == 2.0
    assert np.isfinite(summary["mean_delta_bootstrap_ci_low"])
    assert np.isfinite(summary["mean_delta_bootstrap_ci_high"])
