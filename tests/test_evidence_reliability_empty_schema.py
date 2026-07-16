from __future__ import annotations

import pandas as pd

from hipporeplayimm.evidence_reliability import (
    RELIABILITY_FLAG_COLUMNS,
    add_event_reliability_flags,
)


def test_empty_reliability_annotation_preserves_standard_schema_on_rerun() -> None:
    scores = pd.DataFrame({"model": pd.Series(dtype=object)})

    once = add_event_reliability_flags(scores)
    twice = add_event_reliability_flags(once)

    assert once.empty
    assert twice.empty
    for column in RELIABILITY_FLAG_COLUMNS:
        assert column in once.columns
        assert list(twice.columns).count(column) == 1
    assert once["event_reliability_reasons"].dtype == object
    for column in RELIABILITY_FLAG_COLUMNS:
        if column != "event_reliability_reasons":
            assert once[column].dtype == bool
