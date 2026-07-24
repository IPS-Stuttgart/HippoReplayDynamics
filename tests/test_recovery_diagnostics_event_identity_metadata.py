from __future__ import annotations

import numpy as np
import pandas as pd

from hipporeplayimm.recovery_diagnostics import build_recovery_diagnostic_tables


def _score_row(
    *,
    session: object,
    event_index: object,
    model: str,
    log_evidence: float,
) -> dict[str, object]:
    expected_model = "sorted-spike-state-space-momentum"
    return {
        "status": "success",
        "session": session,
        "event_index": event_index,
        "true_model": "momentum",
        "expected_model": expected_model,
        "model": model,
        "requested_model": model,
        "log_evidence": log_evidence,
        "n_time": 5,
        "n_spikes": 20,
        "evidence_support": "exact_full_grid",
        "evidence_comparable": True,
        "recovered_expected_model": model == expected_model,
    }


def test_recovery_diagnostics_normalize_persisted_event_identities() -> None:
    diffusion = "sorted-spike-state-space-diffusion"
    momentum = "sorted-spike-state-space-momentum"
    scores = pd.DataFrame(
        [
            _score_row(
                session=bytearray(b"RatX/OpenY"),
                event_index=[0],
                model=diffusion,
                log_evidence=1.0,
            ),
            _score_row(
                session=memoryview(b"RatX/OpenY"),
                event_index=(np.int64(0),),
                model=momentum,
                log_evidence=2.0,
            ),
            _score_row(
                session=np.bytes_("RatX/OpenY"),
                event_index=np.array([1]),
                model=diffusion,
                log_evidence=1.0,
            ),
            _score_row(
                session="RatX/OpenY",
                event_index=np.int64(1),
                model=momentum,
                log_evidence=2.0,
            ),
        ]
    )

    tables = build_recovery_diagnostic_tables(scores)
    events = tables.event_diagnostics.sort_values("event_index").reset_index(drop=True)

    assert events["session"].tolist() == ["RatX/OpenY", "RatX/OpenY"]
    assert events["event_index"].tolist() == [0, 1]
    assert events["successful_scores"].tolist() == [2, 2]
    assert events["comparable_scores"].tolist() == [2, 2]
    assert events["strict_recovered_expected_model"].tolist() == [True, True]
    assert events["certified_vs_exact_recovered_expected_model"].tolist() == [True, True]
