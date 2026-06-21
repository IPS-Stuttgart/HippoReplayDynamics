"""Strict parsing for saved train/test cell-ID metadata."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


_PATCHED_FLAG = "_ground_truth_strict_cell_id_metadata_patch_applied"


def apply_ground_truth_cell_id_metadata_patch() -> None:
    """Reject malformed cell-ID metadata instead of silently truncating it."""

    from . import ground_truth as gt

    if getattr(gt, _PATCHED_FLAG, False):
        return

    def parse_cell_ids(value: object) -> np.ndarray | None:
        if value is None:
            return None
        if isinstance(value, np.ndarray):
            return _integer_array_from_values(value)
        if isinstance(value, (list, tuple, set)):
            return _integer_array_from_values(list(value))
        if pd.isna(value):
            return None
        text = str(value).strip()
        if not text:
            return None
        text = text.strip("[]()").replace(",", " ")
        if not text.strip():
            return None
        return _integer_array_from_values(text.split())

    parse_cell_ids.__name__ = gt._parse_cell_ids.__name__
    parse_cell_ids.__doc__ = gt._parse_cell_ids.__doc__
    gt._parse_cell_ids = parse_cell_ids
    setattr(gt, _PATCHED_FLAG, True)


def _integer_array_from_values(values: Any) -> np.ndarray:
    parsed = [_parse_cell_id_value(value) for value in np.asarray(values, dtype=object).reshape(-1)]
    return np.asarray(parsed, dtype=int)


def _parse_cell_id_value(value: Any) -> int:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("cell ID metadata must contain integer values") from exc
    if not np.isfinite(numeric):
        raise ValueError("cell ID metadata must contain finite integer values")
    integer = int(round(numeric))
    if not np.isclose(numeric, integer, rtol=0.0, atol=1e-9):
        raise ValueError("cell ID metadata must contain integer values")
    return int(integer)


__all__ = ["apply_ground_truth_cell_id_metadata_patch"]
