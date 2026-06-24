"""Strict score-table boolean metadata parsing."""

from __future__ import annotations

import numpy as np


def _parse_strict_bool(value: object) -> bool:
    """Parse boolean-like metadata without accepting arbitrary numerics."""

    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer, float, np.floating)):
        return _parse_numeric_bool(float(value), value)

    text = str(value).strip().lower()
    if text in {"true", "yes", "on"}:
        return True
    if text in {"false", "no", "off"}:
        return False
    try:
        numeric = float(text)
    except ValueError:
        pass
    else:
        return _parse_numeric_bool(numeric, value)
    raise ValueError(f"cannot parse boolean value {value!r}")


def _parse_numeric_bool(numeric: float, original: object) -> bool:
    if not np.isfinite(numeric):
        raise ValueError(f"cannot parse boolean value {original!r}")
    if np.isclose(numeric, 0.0, rtol=0.0, atol=0.0):
        return False
    if np.isclose(numeric, 1.0, rtol=0.0, atol=0.0):
        return True
    raise ValueError(f"cannot parse boolean value {original!r}")


def apply_score_metadata_bool_validation_patch() -> None:
    """Install strict parsing for score metadata boolean fields."""

    from . import score_metadata as score_metadata_module

    if getattr(score_metadata_module, "_score_metadata_bool_validation_patch_applied", False):
        return

    score_metadata_module._parse_bool = _parse_strict_bool
    score_metadata_module._score_metadata_bool_validation_patch_applied = True
