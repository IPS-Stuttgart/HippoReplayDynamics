from __future__ import annotations

from decimal import Decimal

import pandas as pd

from hipporeplayimm.recovery_diagnostics import _event_index_value, build_recovery_diagnostic_tables


def _score_row(event_index: object) -> dict[str, object]:
    model = "sorted-spike-state-space-diffusion"
    return {
        "status": "success",
        "session": "RatX/OpenY",
        "event_index": event_index,
        "true_model": "diffusion",
        "expected_model": model,
        "model": model,
        "requested_model": model,
        "log_evidence": 1.0,
        "n_time": 5,
        "n_spikes": 20,
        "evidence_support": "exact_full_grid",
        "evidence_comparable": True,
        "recovered_expected_model": True,
    }


def test_recovery_diagnostics_preserve_large_integral_event_index() -> None:
    event_index = 2**53 + 1

    tables = build_recovery_diagnostic_tables(pd.DataFrame([_score_row(event_index)]))

    assert tables.event_diagnostics.loc[0, "event_index"] == event_index


def test_event_index_normalization_handles_arbitrary_size_integers() -> None:
    event_index = 10**400

    assert _event_index_value(event_index) == event_index


def test_event_index_normalization_does_not_round_exact_decimals() -> None:
    integral = Decimal("9007199254740993")
    fractional = Decimal("9007199254740993.5")

    assert _event_index_value(integral) == 9007199254740993
    assert _event_index_value(fractional) == fractional
