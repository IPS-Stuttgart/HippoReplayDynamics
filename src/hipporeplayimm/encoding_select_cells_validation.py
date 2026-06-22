"""Validate cell IDs requested from fitted encoding models.

``EncodingModel.select_cells`` historically coerced requested IDs with
``dtype=int`` before validation.  That could silently truncate non-integer inputs
such as ``1.9`` to ``1`` and select the wrong cell.  This patch validates the
requested identifiers before casting them to the integer dtype used internally.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

_PATCHED_FLAG = "_encoding_select_cells_validation_patch_applied"


def _canonical_requested_cell_ids(cell_ids: Iterable[int]) -> np.ndarray:
    """Return sorted unique integer cell IDs without lossy coercion."""

    try:
        values = list(cell_ids)
    except TypeError as exc:
        raise TypeError("cell_ids must be an iterable of integer cell IDs") from exc

    ids: list[int] = []
    for value in values:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise TypeError("cell_ids must contain integer cell IDs")
        ids.append(int(value))
    return np.asarray(sorted(set(ids)), dtype=int)


def apply_encoding_select_cells_validation_patch() -> None:
    """Install strict request validation for ``EncodingModel.select_cells``."""

    from . import encoding

    if getattr(encoding, _PATCHED_FLAG, False):
        return

    def select_cells(self, cell_ids: Iterable[int]) -> "encoding.EncodingModel":
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

    encoding.EncodingModel.select_cells = select_cells
    setattr(encoding, _PATCHED_FLAG, True)


__all__ = ["apply_encoding_select_cells_validation_patch"]
