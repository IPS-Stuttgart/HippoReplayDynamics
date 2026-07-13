from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path("scripts").resolve()))
from model_evidence_support_audit import (  # noqa: E402
    EXACT_EVIDENCE_SUPPORT,
    TRUNCATED_EVIDENCE_SUPPORT,
    event_support_audit,
    pairwise_support_audit,
)


def test_support_audits_keep_random_seed_runs_independent():
    rows = []
    for random_seed, support, comparable in (
        (11, EXACT_EVIDENCE_SUPPORT, True),
        (12, TRUNCATED_EVIDENCE_SUPPORT, False),
    ):
        for model in ("diffusion", "momentum"):
            rows.append(
                {
                    "session": "Rat1/Open1",
                    "event_index": 3,
                    "random_seed": random_seed,
                    "model": model,
                    "model_family": "trajectory",
                    "status": "success",
                    "log_evidence": float(random_seed),
                    "evidence_support": support,
                    "evidence_comparable": comparable,
                }
            )
    scores = pd.DataFrame(rows)

    event_audit = event_support_audit(scores).sort_values("random_seed").reset_index(drop=True)
    assert event_audit["random_seed"].tolist() == [11, 12]
    assert not event_audit["has_mixed_exact_truncated"].any()
    assert event_audit["exact_rows"].tolist() == [2, 0]
    assert event_audit["truncated_rows"].tolist() == [0, 2]

    pairwise = pairwise_support_audit(scores).sort_values("comparison_support").reset_index(drop=True)
    assert set(pairwise["comparison_support"]) == {
        f"{EXACT_EVIDENCE_SUPPORT}_vs_{EXACT_EVIDENCE_SUPPORT}",
        f"{TRUNCATED_EVIDENCE_SUPPORT}_vs_{TRUNCATED_EVIDENCE_SUPPORT}",
    }
    assert pairwise["events"].tolist() == [1, 1]
    assert not pairwise["mixes_exact_and_truncated"].any()
