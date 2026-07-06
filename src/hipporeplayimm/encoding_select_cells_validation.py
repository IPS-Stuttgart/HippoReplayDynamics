"""Validate requested encoding cells and encoding configuration values.

``EncodingModel.select_cells`` historically coerced requested IDs with
``dtype=int`` before validation.  That could silently truncate non-integer inputs
such as ``1.9`` to ``1`` and select the wrong cell.  This patch validates the
requested identifiers before casting them to the integer dtype used internally,
while still accepting integer-valued numeric IDs commonly loaded from MATLAB
files as floats.  Integer identifiers are validated in their integer domain so
large cell IDs are not rounded through floating-point conversion.

The same imported runtime hook also validates ``EncodingConfig`` scalar fields
before the core encoder's historical ``float(...)`` coercion.  Without this
check, booleans, numeric strings, and array-shaped values can be accepted as
valid numeric encoder parameters or truthy boolean switches.
"""

from __future__ import annotations

from collections.abc import Iterable
from functools import wraps
import sys

import numpy as np

_PATCHED_FLAG = "_encoding_select_cells_validation_patch_applied"
_CONFIG_PATCHED_FLAG = "_encoding_config_validation_patch_applied"


def _coerce_requested_cell_id(value: object, integer_info: np.iinfo) -> int:
    """Coerce one requested cell ID without passing integer inputs through float."""

    try:
        scalar = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise TypeError("cell_ids must contain integer-valued cell IDs") from exc
    if scalar.ndim != 0:
        raise ValueError("cell_ids must be one-dimensional")

    item = scalar.item()
    if isinstance(item, (bool, np.bool_)):
        raise TypeError("cell_ids must not contain boolean identifiers")

    if isinstance(item, (int, np.integer)):
        cell_id = int(item)
    elif isinstance(item, (float, np.floating)):
        numeric = float(item)
        if not np.isfinite(numeric):
            raise ValueError("cell_ids must be finite")
        if not numeric.is_integer():
            raise ValueError("cell_ids must contain integer-valued cell IDs")
        cell_id = int(numeric)
    else:
        raise TypeError("cell_ids must contain integer-valued cell IDs")

    if cell_id < int(integer_info.min) or cell_id > int(integer_info.max):
        raise ValueError("cell_ids must fit into integer identifier range")
    return cell_id


def _canonical_requested_cell_ids(cell_ids: Iterable[int | float]) -> np.ndarray:
    """Return sorted unique integer cell IDs without lossy coercion."""

    try:
        values = list(cell_ids)
    except TypeError as exc:
        raise TypeError("cell_ids must be an iterable of integer-valued cell IDs") from exc

    if not values:
        return np.empty(0, dtype=int)

    integer_info = np.iinfo(np.dtype(int))
    requested = [_coerce_requested_cell_id(value, integer_info) for value in values]
    return np.asarray(sorted(set(requested)), dtype=int)


def _coerce_float_config_value(name: str, value: object) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a scalar float")
    try:
        scalar = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a scalar float") from exc
    if scalar.shape != ():
        raise TypeError(f"{name} must be a scalar float")
    if scalar.dtype.kind in {"S", "U", "c"}:
        raise TypeError(f"{name} must be a scalar float")
    try:
        item = scalar.item()
    except (AttributeError, IndexError, ValueError):
        item = value
    if isinstance(item, (bool, np.bool_, str, bytes, complex, np.complexfloating)):
        raise TypeError(f"{name} must be a scalar float")
    try:
        return float(item)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{name} must be a scalar float") from exc


def _validate_float_config_value(config: object, name: str, *, positive: bool) -> None:
    value = _coerce_float_config_value(name, getattr(config, name))
    if not np.isfinite(value) or value < 0.0 or (positive and value <= 0.0):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be finite and {qualifier}")


def _validate_bool_config_value(config: object, name: str) -> None:
    value = getattr(config, name)
    try:
        scalar = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a scalar boolean") from exc
    if scalar.shape != ():
        raise TypeError(f"{name} must be a scalar boolean")
    try:
        item = scalar.item()
    except (AttributeError, IndexError, ValueError):
        item = value
    if not isinstance(item, (bool, np.bool_)):
        raise TypeError(f"{name} must be a scalar boolean")


def _validate_encoding_config_scalar_types(config: object) -> None:
    for name in ("bin_size_cm", "min_occupancy_s", "rate_floor_hz"):
        _validate_float_config_value(config, name, positive=True)
    for name in ("smoothing_sigma_bins", "min_speed_cm_s", "arena_padding_cm"):
        _validate_float_config_value(config, name, positive=False)
    for name in ("use_excitatory", "exclude_ripple_intervals"):
        _validate_bool_config_value(config, name)


def apply_encoding_config_validation_patch() -> None:
    """Install strict scalar validation for ``EncodingConfig`` values."""

    from . import encoding

    current = getattr(encoding, "_validate_encoding_config", None)
    if getattr(current, _CONFIG_PATCHED_FLAG, False):
        setattr(encoding, _CONFIG_PATCHED_FLAG, True)
        _synchronize_encoding_config_aliases(getattr(current, "__hipporeplayimm_original__", None), current)
        return

    original_validate_encoding_config = current

    @wraps(original_validate_encoding_config)
    def _validate_encoding_config(config: "encoding.EncodingConfig") -> None:
        _validate_encoding_config_scalar_types(config)
        original_validate_encoding_config(config)

    setattr(_validate_encoding_config, _CONFIG_PATCHED_FLAG, True)
    setattr(_validate_encoding_config, "__hipporeplayimm_original__", original_validate_encoding_config)
    encoding._validate_encoding_config = _validate_encoding_config
    setattr(encoding, _CONFIG_PATCHED_FLAG, True)
    _synchronize_encoding_config_aliases(original_validate_encoding_config, _validate_encoding_config)


def _synchronize_encoding_config_aliases(previous: object | None, patched: object) -> None:
    if previous is None:
        return
    for module in list(sys.modules.values()):
        module_name = getattr(module, "__name__", "")
        if not module_name.startswith("hipporeplayimm"):
            continue
        if getattr(module, "_validate_encoding_config", None) is previous:
            module._validate_encoding_config = patched


def apply_encoding_select_cells_validation_patch() -> None:
    """Install strict request validation for ``EncodingModel.select_cells``."""

    apply_encoding_config_validation_patch()

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


# ``hipporeplayimm.__init__`` imports this module before re-exporting
# ``EncodingModel``.  Apply the patches on import so top-level package imports get
# the same strict validation that ``apply_runtime_patches()`` installs later.
apply_encoding_config_validation_patch()
apply_encoding_select_cells_validation_patch()


__all__ = ["apply_encoding_config_validation_patch", "apply_encoding_select_cells_validation_patch"]
