from __future__ import annotations

import pytest

from scripts.event_window_sensitivity_plan import _parse_paddings_s


def test_parse_paddings_accepts_comma_separated_values() -> None:
    assert _parse_paddings_s("0, 0.01,0.02") == (0.0, 0.01, 0.02)


@pytest.mark.parametrize(
    "value",
    ["", "0,,0.02", "0,", ",0.01", "nan", "inf", "-0.01", "0,abc"],
)
def test_parse_paddings_rejects_malformed_or_invalid_values(value: str) -> None:
    with pytest.raises(ValueError):
        _parse_paddings_s(value)
