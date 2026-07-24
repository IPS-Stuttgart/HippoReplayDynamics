from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hipporeplayimm import recovery_diagnostics as diagnostics
from hipporeplayimm.recovery_diagnostics import build_recovery_diagnostic_tables


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (b"true", True),
        (bytearray(b"1"), True),
        (memoryview(b"yes"), True),
        (np.bytes_("true"), True),
        (np.array([True]), True),
        ([np.bytes_("1")], True),
        ((bytearray(b"yes"),), True),
        (b"false", False),
        (bytearray(b"0"), False),
        (memoryview(b"no"), False),
        (np.bytes_("false"), False),
        (np.array([False]), False),
        ([np.bytes_("0")], False),
        ((bytearray(b"no"),), False),
    ],
)
def test_recovery_bool_coercion_decodes_persisted_scalar_metadata(value: object, expected: bool):
    assert diagnostics._coerce_bool(value) is expected


def test_recovery_diagnostics_keep_byte_backed_exact_scores():
    expected_model = "sorted-spike-state-space-momentum"
    scores = pd.DataFrame(
        [
            {
                "session": "RatX/OpenY",
                "event_index": 0,
                "true_model": "momentum",
                "expected_model": expected_model,
                "model": "sorted-spike-state-space-diffusion",
                "log_evidence": 1.0,
                "status": b"success",
                "evidence_support": "exact_full_grid",
                "evidence_comparable": memoryview(b"true"),
            },
            {
                "session": "RatX/OpenY",
                "event_index": 0,
                "true_model": "momentum",
                "expected_model": expected_model,
                "model": expected_model,
                "log_evidence": 2.0,
                "status": np.array([np.bytes_("success")]),
                "evidence_support": "exact_full_grid",
                "evidence_comparable": [np.bytes_("true")],
            },
        ]
    )

    tables = build_recovery_diagnostic_tables(scores)
    event = tables.event_diagnostics.iloc[0]

    assert event["successful_scores"] == 2
    assert event["comparable_scores"] == 2
    assert event["strict_best_model"] == expected_model
    assert bool(event["strict_recovered_expected_model"])
    assert bool(event["certified_vs_exact_recovered_expected_model"])
    assert event["failure_mode"] == "strict_recovered"
