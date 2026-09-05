import numpy as np
import pandas as pd
import pytest

from scripts.audit_replay_coverage_recovery import compare_tables, support_from_direct_counts


def test_direct_support_recount_has_no_gap_bridging():
    base = np.zeros((20, 2), int)
    base[:4] = 2
    base[8:] = 2
    result = support_from_direct_counts(base)
    assert result["windows"] == 17
    assert result["unfiltered_steps"] == 4
    assert result["filtered_steps"] == 2


def test_table_audit_rejects_missing_corrupt_and_duplicate_rows():
    expected = pd.DataFrame({"key": [1, 2], "count": [3, 4], "hash": ["a", "b"]})
    compare_tables(expected.iloc[::-1], expected, ["key"], ["count", "hash"])
    for actual in [expected.iloc[:1], pd.concat([expected, expected]), expected.assign(count=[3, 9]), expected.assign(hash=["a", "c"])]:
        with pytest.raises(AssertionError):
            compare_tables(actual, expected, ["key"], ["count", "hash"])
