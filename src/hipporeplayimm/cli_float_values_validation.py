"""Runtime validation for comma-separated CLI float grids."""

from __future__ import annotations

import math


def apply_cli_float_values_validation_patch() -> None:
    """Reject NaN and infinite values in comma-separated CLI float lists."""

    from . import cli as _cli

    current = _cli._parse_float_values
    if getattr(current, "_hipporeplayimm_rejects_nonfinite", False):
        return

    def _parse_finite_float_values(value: str) -> tuple[float, ...]:
        parsed = current(value)
        if not all(math.isfinite(item) for item in parsed):
            raise ValueError(
                "comma-separated float value list must contain only finite values"
            )
        return parsed

    _parse_finite_float_values._hipporeplayimm_rejects_nonfinite = True  # type: ignore[attr-defined]
    _parse_finite_float_values._hipporeplayimm_original = current  # type: ignore[attr-defined]
    _cli._parse_float_values = _parse_finite_float_values
