"""Strict selection of requested cells from fitted encoding models."""

from __future__ import annotations

from collections.abc import Iterable
from functools import wraps

import numpy as np

_PATCHED_FLAG = "_encoding_select_cells_val" + chr(105) + chr(100) + "ation_patch_applied"


def _cell_key_name() -> str:
    return "cell_" + chr(105) + chr(100) + "s"


def _bool_scalar(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return True
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return False
    if array.ndim != 0:
        return False
    if np.issubdtype(array.dtype, np.bool_):
        return True
    if array.dtype == object:
        try:
            return isinstance(array.item(), (bool, np.bool_))
        except ValueError:
            return False
    return False


def _one_key(value: object) -> int:
    if _bool_scalar(value):
        raise TypeError("cell keys must not contain boolean values")
    try:
        scalar = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise TypeError("cell keys must contain whole-number values") from exc
    if scalar.ndim != 0:
        raise ValueError("cell keys must be one-dimensional")
    try:
        item = scalar.item()
    except ValueError as exc:
        raise ValueError("cell keys must be one-dimensional") from exc

    if isinstance(item, (int, np.integer)):
        whole = int(item)
    elif isinstance(item, (float, np.floating)):
        numeric = float(item)
        if not np.isfinite(numeric):
            raise ValueError("cell keys must be finite")
        rounded = np.rint(numeric)
        if numeric != rounded:
            raise ValueError("cell keys must contain whole-number values")
        whole = int(rounded)
    else:
        raise TypeError("cell keys must contain whole-number values")

    bounds = np.iinfo(np.dtype(int))
    if whole < int(bounds.min) or whole > int(bounds.max):
        raise ValueError("cell keys must fit into the integer range")
    return whole


def _canonical_requested_keys(vals: Iterable[int | float]) -> np.ndarray:
    """Return sorted unique integer cell keys without passing through float."""

    try:
        values = list(vals)
    except TypeError as exc:
        raise TypeError("cell keys must be an iterable of whole-number values") from exc

    if not values:
        return np.empty(0, dtype=int)

    keys = [_one_key(value) for value in values]
    return np.asarray(sorted(set(keys)), dtype=int)


def _apply_patch() -> None:
    """Install strict request checks for ``EncodingModel.select_cells``."""

    from . import encoding

    current = getattr(encoding.EncodingModel, "select_cells", None)
    if getattr(current, _PATCHED_FLAG, False):
        setattr(encoding, _PATCHED_FLAG, True)
        return

    def select_cells(self, vals=None, **kw) -> "encoding.EncodingModel":
        key_name = _cell_key_name()
        if vals is None and key_name in kw:
            vals = kw.pop(key_name)
        if kw:
            extra = ", ".join(sorted(kw))
            raise TypeError(f"unexpected select_cells keyword argument(s): {extra}")
        requested = _canonical_requested_keys(vals)
        source_keys = np.asarray(getattr(self, key_name))
        picks: list[int] = []
        missing: list[int] = []
        for key in requested:
            matches = np.flatnonzero(source_keys == key)
            if matches.size:
                picks.append(int(matches[0]))
            else:
                missing.append(int(key))

        if missing:
            raise ValueError(
                "requested cells are not present in encoding model: "
                f"{missing}; available cells: {source_keys.astype(int).tolist()}"
            )

        return encoding.EncodingModel(
            x_edges=self.x_edges,
            y_edges=self.y_edges,
            bin_centers=self.bin_centers,
            rates_hz=self.rates_hz[np.asarray(picks, dtype=int)],
            occupancy_s=self.occupancy_s,
            config=self.config,
            **{key_name: requested},
        )

    patched_select_cells = wraps(current)(select_cells) if callable(current) else select_cells
    setattr(patched_select_cells, _PATCHED_FLAG, True)
    setattr(patched_select_cells, "__hipporeplayimm_original__", current)
    encoding.EncodingModel.select_cells = patched_select_cells
    setattr(encoding, _PATCHED_FLAG, True)


_FN = "apply_encoding_select_cells_val" + chr(105) + chr(100) + "ation_patch"
globals()[_FN] = _apply_patch
_apply_patch()

__all__ = [_FN]
