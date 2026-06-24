from __future__ import annotations

import pytest

from hipporeplayimm import cli


def test_cli_float_value_grids_reject_nonfinite_values() -> None:
    for raw in ("nan", "1.0,nan", "inf", "-inf"):
        with pytest.raises(ValueError, match="finite"):
            cli._parse_float_values(raw)


def test_cli_float_value_grids_accept_finite_values() -> None:
    assert cli._parse_float_values("0.5, 1, 2.5") == (0.5, 1.0, 2.5)
