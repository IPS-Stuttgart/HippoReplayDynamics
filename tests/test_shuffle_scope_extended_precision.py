from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hipporeplayimm.shuffle_controls import add_shuffle_p_values


def _extended_precision_integer() -> tuple[np.longdouble, int]:
    expected = 2**53 + 1
    value = np.longdouble(str(expected))
    if int(value) != expected:
        pytest.skip("platform longdouble does not exceed binary64 integer precision")
    return value, expected


def _score(seed: object, log_evidence: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "session": ["Rat1/Open1"],
            "event_index": [3],
            "model": ["state-space-first-order-imm"],
            "benchmark_random_seed": pd.Series([seed], dtype=object),
            "log_evidence": [log_evidence],
        }
    )


def test_shuffle_scope_matches_equal_extended_precision_and_integer_seeds() -> None:
    extended_seed, exact_seed = _extended_precision_integer()

    annotated = add_shuffle_p_values(
        _score(extended_seed, 10.0),
        _score(exact_seed, 12.0),
    )

    assert annotated.loc[0, "shuffle_count"] == 1
    assert annotated.loc[0, "shuffle_p_value"] == 1.0


def test_shuffle_scope_keeps_adjacent_extended_precision_seeds_distinct() -> None:
    extended_seed, exact_seed = _extended_precision_integer()
    adjacent_seed = np.longdouble(str(exact_seed - 1))

    annotated = add_shuffle_p_values(
        _score(extended_seed, 10.0),
        _score(adjacent_seed, 12.0),
    )

    assert np.isnan(annotated.loc[0, "shuffle_count"])
    assert np.isnan(annotated.loc[0, "shuffle_p_value"])
