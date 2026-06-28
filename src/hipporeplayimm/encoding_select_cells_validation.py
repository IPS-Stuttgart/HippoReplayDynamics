"""Validate cell IDs requested from fitted encoding models.

``EncodingModel.select_cells`` historically coerced requested IDs with
``dtype=int`` before validation.  That could silently truncate non-integer inputs
such as ``1.9`` to ``1`` and select the wrong cell.  This patch validates the
requested identifiers before casting them to the integer dtype used internally,
while still accepting integer-valued numeric IDs commonly loaded from MATLAB
files as floats.
"""

from __future__ import annotations

from collections.abc import Iterable
from functools import wraps

import numpy as np

_PATCHED_FLAG = "_encoding_select_cells_validation_patch_applied"


def _canonical_requested_cell_ids(cell_ids: Iterable[int | float]) -> np.ndarray:
    """Return sorted unique integer cell IDs without lossy coercion."""

    try:
        values = list(cell_ids)
    except TypeError as exc:
        raise TypeError("cell_ids must be an iterable of integer-valued cell IDs") from exc

    if not values:
        return np.empty(0, dtype=int)

    for value in values:
        if isinstance(value, (bool, np.bool_)):
            raise TypeError("cell_ids must not contain boolean identifiers")
        if not isinstance(value, (int, np.integer, float, np.floating)):
            raise TypeError("cell_ids must contain integer-valued cell IDs")

    numeric = np.asarray(values, dtype=float)
    if numeric.ndim != 1:
        raise ValueError("cell_ids must be one-dimensional")
    if not np.all(np.isfinite(numeric)):
        raise ValueError("cell_ids must be finite")

    rounded = np.rint(numeric)
    if not np.all(numeric == rounded):
        raise ValueError("cell_ids must contain integer-valued cell IDs")

    integer_info = np.iinfo(np.dtype(int))
    if not np.all((rounded >= integer_info.min) & (rounded <= integer_info.max)):
        raise ValueError("cell_ids must fit into integer identifier range")
    return np.asarray(sorted(set(rounded.astype(int).tolist())), dtype=int)


def apply_encoding_select_cells_validation_patch() -> None:
    """Install strict request validation for ``EncodingModel.select_cells``."""

    from . import encoding

    current = getattr(encoding.EncodingModel, "select_cells", None)
    if getattr(current, _PATCHED_FLAG, False):
        setattr(encoding, _PATCHED_FLAG, True)
        return

    def select_cells(self, cell_ids: Iterable[int | float]) -> "encoding.EncodingModel":
        requested = _canonical_requested_cell_ids(cell_ids)
        indices: list[int] = []
        missing: list[int] = []
        for cell_id in requested:
            matches = np.flatnonzero(self.cell_ids == cell_id)
            if matches.size:
                indices.append(int(matches[0]))
            else:
                missing.append(int(cell_id))

        if missing:
            raise ValueError(
                "requested cell IDs are not present in encoding model: "
                f"{missing}; available cell IDs: {self.cell_ids.astype(int).tolist()}"
            )

        return encoding.EncodingModel(
            x_edges=self.x_edges,
            y_edges=self.y_edges,
            bin_centers=self.bin_centers,
            rates_hz=self.rates_hz[np.asarray(indices, dtype=int)],
            occupancy_s=self.occupancy_s,
            cell_ids=requested,
            config=self.config,
        )

    patched_select_cells = wraps(current)(select_cells) if callable(current) else select_cells
    setattr(patched_select_cells, _PATCHED_FLAG, True)
    setattr(patched_select_cells, "__hipporeplayimm_original__", current)
    encoding.EncodingModel.select_cells = patched_select_cells
    setattr(encoding, _PATCHED_FLAG, True)


__all__ = ["apply_encoding_select_cells_validation_patch"]
