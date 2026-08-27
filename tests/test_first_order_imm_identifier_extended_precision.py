from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))

from audit_first_order_imm_event_mean_mode_usage import _parse_integer_identifier


def _require_extended_precision() -> None:
    if np.finfo(np.longdouble).nmant <= np.finfo(float).nmant:
        pytest.skip("platform longdouble does not exceed binary64 precision")


def test_identifier_parser_rejects_fractional_extended_precision_value() -> None:
    _require_extended_precision()
    value = np.longdouble("9007199254740993.5")

    assert not bool(value.is_integer())
    assert float(value).is_integer()
    with pytest.raises(ValueError, match="event_index"):
        _parse_integer_identifier(value, name="event_index")


def test_identifier_parser_accepts_integral_extended_precision_value() -> None:
    _require_extended_precision()
    expected = 2**53 + 1
    value = np.longdouble(str(expected))

    assert _parse_integer_identifier(value, name="event_index") == expected
