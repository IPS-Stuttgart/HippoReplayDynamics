import numpy as np
import pandas as pd

from hipporeplayimm.evidence_reporting import (
    EXACT_EVIDENCE_SUPPORT,
    ensure_evidence_support_columns,
    simulation_event_best_rows,
)


def test_generic_evidence_support_marks_nonfinite_exact_rows_noncomparable():
    rows = pd.DataFrame(
        [
            {
                "status": "success",
                "model": "finite",
                "log_evidence": 0.0,
                "evidence_support": EXACT_EVIDENCE_SUPPORT,
            },
            {
                "status": "success",
                "model": "nan",
                "log_evidence": np.nan,
                "evidence_support": EXACT_EVIDENCE_SUPPORT,
            },
            {
                "status": "success",
                "model": "positive-inf",
                "log_evidence": np.inf,
                "evidence_support": EXACT_EVIDENCE_SUPPORT,
            },
            {
                "status": "success",
                "model": "negative-inf",
                "log_evidence": -np.inf,
                "evidence_support": EXACT_EVIDENCE_SUPPORT,
            },
        ]
    )

    scored = ensure_evidence_support_columns(rows)
    comparable = scored.set_index("model")["evidence_comparable"]

    assert bool(comparable.loc["finite"])
    assert not bool(comparable.loc["nan"])
    assert not bool(comparable.loc["positive-inf"])
    assert not bool(comparable.loc["negative-inf"])


def test_simulation_event_best_rows_ignores_nonfinite_exact_rows_with_stale_best_flags():
    rows = pd.DataFrame(
        [
            {
                "status": "success",
                "session": "RatX/OpenY",
                "event_index": 0,
                "model": "bad-nonfinite",
                "log_evidence": np.inf,
                "evidence_support": EXACT_EVIDENCE_SUPPORT,
                "is_best_model": True,
            },
            {
                "status": "success",
                "session": "RatX/OpenY",
                "event_index": 0,
                "model": "good-finite",
                "log_evidence": 0.0,
                "evidence_support": EXACT_EVIDENCE_SUPPORT,
                "is_best_model": False,
            },
        ]
    )

    best = simulation_event_best_rows(rows)

    assert best["model"].tolist() == ["good-finite"]
