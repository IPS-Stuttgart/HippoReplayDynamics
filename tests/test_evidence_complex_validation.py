from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from hipporeplayimm.evidence_reporting import (
    ensure_evidence_support_columns,
    simulation_add_evidence_columns,
)
from hipporeplayimm.recovery_diagnostics import _successful_finite_scores


def _nested_complex_scalar() -> np.ndarray:
    inner = np.empty((), dtype=object)
    inner[()] = np.complex128(2.0 + 50.0j)
    outer = np.empty((), dtype=object)
    outer[()] = inner
    return outer


def test_complex_evidence_is_excluded_without_cast_warnings():
    frame = pd.DataFrame(
        {
            "session": ["RatX/Open1"] * 3,
            "event_index": [0] * 3,
            "model": ["direct-complex", "wrapped-complex", "real"],
            "log_evidence": pd.Series(
                [1.0 + 100.0j, _nested_complex_scalar(), "3.0"],
                dtype=object,
            ),
            "heldout_log_likelihood": pd.Series(
                [1.0 + 1.0j, _nested_complex_scalar(), "2.0"],
                dtype=object,
            ),
            "status": ["success"] * 3,
            "diagnostic_candidate_evidence_support": ["exact_full_grid"] * 3,
        }
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        supported = ensure_evidence_support_columns(frame)
        scored = simulation_add_evidence_columns(frame)
        successful = _successful_finite_scores(frame)

    assert supported["evidence_comparable"].tolist() == [False, False, True]
    assert successful["model"].tolist() == ["real"]

    by_model = scored.set_index("model")
    assert np.isnan(by_model.loc["direct-complex", "log_evidence"])
    assert np.isnan(by_model.loc["wrapped-complex", "log_evidence"])
    assert by_model.loc["real", "log_evidence"] == 3.0
    assert by_model.loc["real", "model_probability"] == 1.0
    assert bool(by_model.loc["real", "is_best_model"])
    assert scored["best_model"].eq("real").all()
