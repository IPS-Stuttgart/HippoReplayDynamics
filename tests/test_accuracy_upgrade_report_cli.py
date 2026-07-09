from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts.accuracy_upgrade_report import _event_ids


def _session() -> SimpleNamespace:
    return SimpleNamespace(
        ripple_count=5,
        ripple_indices_in_run=lambda: [3, 1, 3],
    )


def test_event_ids_accept_comma_whitespace_ranges_and_deduplicate() -> None:
    assert _event_ids("1, 3 2-4", _session()) == [1, 2, 3, 4]


def test_event_ids_keep_run_and_all_shortcuts() -> None:
    session = _session()

    assert _event_ids("run", session) == [3, 1, 3]
    assert _event_ids("all", session) == [0, 1, 2, 3, 4]


@pytest.mark.parametrize("spec", ["", "   ", "1,,2", "1,", ",2"])
def test_event_ids_reject_empty_entries(spec: str) -> None:
    with pytest.raises(ValueError, match="--events"):
        _event_ids(spec, _session())


@pytest.mark.parametrize("spec", ["3-1", "1--2", "-1", "1.5", "abc"])
def test_event_ids_reject_malformed_indices_and_ranges(spec: str) -> None:
    with pytest.raises(ValueError, match="--events"):
        _event_ids(spec, _session())
