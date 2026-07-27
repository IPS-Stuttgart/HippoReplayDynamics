"""Runtime patches for shuffle-control ordering, integer values, and scope keys."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
import operator

import numpy as np

_PATCHED_FLAG = "_shuffle_spike_time_order_patch_applied"
_SCOPE_KEY_PATCHED_FLAG = "_shuffle_scope_numeric_key_patch_applied"
_GRID_SHAPE_PATCHED_FLAG = "_shuffle_grid_shape_validation_patch_applied"
_INTEGER_VALUE_PATCHED_FLAG = "_shuffle_integer_value_precision_patch_applied"
_PERMUTATION_PATCHED_FLAG = "_shuffle_nonidentity_permutation_patch_applied"
_CELL_IDENTITY_PATCHED_FLAG = "_shuffle_nonidentity_cell_identity_patch_applied"
_MARK_FEATURE_PATCHED_FLAG = "_shuffle_nonidentity_mark_feature_patch_applied"


def apply_shuffle_spike_time_order_patch() -> None:
    """Install sorted spike-time shuffling and strict shuffle-control validation."""

    from . import result_improvements as ri
    from . import shuffle_controls

    if not (
        getattr(ri, _PATCHED_FLAG, False)
        and getattr(ri, "shuffle_spike_times_session", None)
        is _shuffle_spike_times_session_sorted
    ):
        ri.shuffle_spike_times_session = _shuffle_spike_times_session_sorted
        setattr(ri, _PATCHED_FLAG, True)

    if not getattr(shuffle_controls, _SCOPE_KEY_PATCHED_FLAG, False):
        original_scope_label = shuffle_controls._scope_label

        def scope_label(value: object) -> str:
            if isinstance(value, Mapping):
                return _mapping_scope_label(value, scope_label)
            nonfinite_numeric = _nonfinite_numeric_scope_label(value)
            if nonfinite_numeric is not None:
                return repr(("scalar", nonfinite_numeric))
            numeric = _numeric_scope_label(value)
            if numeric is not None:
                return repr(("numeric", numeric))
            return original_scope_label(value)

        shuffle_controls._scope_label = scope_label
        setattr(shuffle_controls, _SCOPE_KEY_PATCHED_FLAG, True)

    if not getattr(shuffle_controls, _GRID_SHAPE_PATCHED_FLAG, False):
        original_validate_grid_shape = shuffle_controls._validate_grid_shape

        def validate_grid_shape(grid_shape: object) -> tuple[int, int]:
            return _validated_grid_shape(grid_shape, original_validate_grid_shape)

        shuffle_controls._validate_grid_shape = validate_grid_shape
        setattr(shuffle_controls, _GRID_SHAPE_PATCHED_FLAG, True)

    if not getattr(shuffle_controls, _INTEGER_VALUE_PATCHED_FLAG, False):
        shuffle_controls._nonnegative_integer_value = _nonnegative_integer_value
        setattr(shuffle_controls, _INTEGER_VALUE_PATCHED_FLAG, True)

    if not getattr(shuffle_controls, _PERMUTATION_PATCHED_FLAG, False):
        shuffle_controls.shuffled_encoding = _shuffled_encoding_nonidentity
        setattr(shuffle_controls, _PERMUTATION_PATCHED_FLAG, True)

    if not getattr(ri, _CELL_IDENTITY_PATCHED_FLAG, False):
        ri.shuffle_cell_identities_session = (
            _shuffle_cell_identities_session_nonidentity
        )
        setattr(ri, _CELL_IDENTITY_PATCHED_FLAG, True)

    if not (
        getattr(ri, _MARK_FEATURE_PATCHED_FLAG, False)
        and getattr(ri, "shuffle_mark_features_session", None)
        is _shuffle_mark_features_session_nonidentity
    ):
        ri.shuffle_mark_features_session = _shuffle_mark_features_session_nonidentity
        setattr(ri, _MARK_FEATURE_PATCHED_FLAG, True)


def _mapping_scope_label(value: Mapping[object, object], scope_label) -> str:
    items = sorted(
        ((scope_label(key), scope_label(item)) for key, item in value.items()),
        key=repr,
    )
    return repr(("mapping", items))


def _validated_grid_shape(
    grid_shape: object,
    original_validate_grid_shape,
) -> tuple[int, int]:
    try:
        values = tuple(grid_shape)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError(
            "grid_shape must contain exactly two integer dimensions"
        ) from exc
    if len(values) != 2:
        raise ValueError("grid_shape must contain exactly two integer dimensions")
    return tuple(  # type: ignore[return-value]
        _positive_integer_grid_dimension(value) for value in values
    )


def _positive_integer_grid_dimension(value: object) -> int:
    """Return one exact positive integer grid dimension."""

    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("grid_shape dimensions must be positive integers") from exc
    if array.ndim != 0:
        raise ValueError("grid_shape dimensions must be positive integers")
    scalar = array.item()
    if isinstance(scalar, (bool, np.bool_)):
        raise ValueError("grid_shape dimensions must be positive integers")
    if isinstance(scalar, (str, bytes, np.str_, np.bytes_)):
        raise ValueError("grid_shape dimensions must be positive integers")

    try:
        integer = operator.index(scalar)
    except TypeError:
        try:
            integer = int(scalar)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "grid_shape dimensions must be finite positive integers"
            ) from exc
        try:
            is_exact = bool(scalar == integer)
        except (TypeError, ValueError):
            is_exact = False
        if not is_exact:
            raise ValueError("grid_shape dimensions must be positive integers")

    if integer <= 0:
        raise ValueError("grid_shape dimensions must be positive integers")
    return int(integer)


def _shuffled_encoding_nonidentity(
    encoding,
    *,
    mode: str = "spatial-roll",
    random_seed: int = 1,
):
    """Return a shuffled encoding without identity permutations when avoidable."""

    from . import shuffle_controls

    mode = shuffle_controls._validated_shuffle_mode(mode)
    random_seed = shuffle_controls._nonnegative_integer_value(
        "random_seed",
        random_seed,
    )
    rng = np.random.default_rng(random_seed)
    rates = np.asarray(encoding.rates_hz, dtype=float).copy()
    if mode == "cell-permutation":
        rates = rates[_nonidentity_permutation(rates.shape[0], rng)]
    elif mode == "spatial-roll":
        rates = shuffle_controls._spatial_roll_rates(
            rates,
            encoding.grid_shape,
            rng,
        )
    elif mode == "spatial-permutation":
        rates = rates[:, _nonidentity_permutation(encoding.n_bins, rng)]
    elif mode == "independent-spatial-permutation":
        if rates.shape[0] > 0:
            rates = np.vstack(
                [
                    row[_nonidentity_permutation(encoding.n_bins, rng)]
                    for row in rates
                ]
            )
    else:  # pragma: no cover - guarded by _validated_shuffle_mode.
        raise AssertionError(f"Unhandled shuffle mode: {mode!r}")
    return shuffle_controls.EncodingModel(
        x_edges=encoding.x_edges.copy(),
        y_edges=encoding.y_edges.copy(),
        bin_centers=encoding.bin_centers.copy(),
        rates_hz=rates,
        occupancy_s=encoding.occupancy_s.copy(),
        cell_ids=encoding.cell_ids.copy(),
        config=encoding.config,
    )


def _nonidentity_permutation(
    size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw uniformly from nonidentity permutations when ``size`` exceeds one."""

    identity = np.arange(size)
    if size <= 1:
        return identity
    permutation = rng.permutation(size)
    while np.array_equal(permutation, identity):
        permutation = rng.permutation(size)
    return permutation


def _nonidentity_permuted_values(
    values: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Permute values until their sequence changes when a change is possible."""

    original = np.asarray(values)
    if original.size <= 1 or np.unique(original).size <= 1:
        return original.copy()
    while True:
        permuted = original[_nonidentity_permutation(original.size, rng)]
        if not np.array_equal(permuted, original, equal_nan=True):
            return permuted


def _shuffle_cell_identities_session_nonidentity(session, random_seed: int = 1):
    """Remap valid integral session cell identities without an avoidable identity draw."""

    from .data_cell_id_validation import _coerce_integral_ids

    rng = np.random.default_rng(_nonnegative_integer_seed(random_seed))
    raw_spikes = np.asarray(session.spikes)
    if raw_spikes.size == 0:
        return session
    if raw_spikes.ndim != 2 or raw_spikes.shape[1] < 2:
        raise ValueError(
            "session.spikes must be a two-dimensional array with time and cell-ID columns"
        )
    spike_cell_ids = _coerce_integral_ids(raw_spikes[:, 1], "spike cell IDs")
    try:
        spikes = np.asarray(session.spikes, dtype=float).copy()
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "session.spikes must contain numeric time and cell-ID values"
        ) from exc
    cells = np.unique(spike_cell_ids)
    shuffled = cells[_nonidentity_permutation(cells.size, rng)]
    mapping = {
        int(source): int(target)
        for source, target in zip(cells, shuffled, strict=True)
    }
    spikes[:, 1] = [mapping[int(cell)] for cell in spike_cell_ids]
    marks = session.spike_marks
    if marks is not None and marks.cell_ids is not None:
        mark_ids = _coerce_integral_ids(marks.cell_ids, "spike mark cell IDs")
        if mark_ids.ndim != 1:
            raise ValueError("spike mark cell IDs must be one-dimensional")
        mark_cell_ids = np.asarray(
            [mapping.get(int(cell), int(cell)) for cell in mark_ids],
            dtype=int,
        )
        marks = replace(marks, cell_ids=mark_cell_ids)
    return replace(session, spikes=spikes, spike_marks=marks)


def _shuffle_spike_times_session_sorted(session, random_seed: int = 1):
    from . import result_improvements as ri

    rng = np.random.default_rng(_nonnegative_integer_seed(random_seed))
    spikes = np.asarray(session.spikes, dtype=float).copy()
    if spikes.size == 0:
        return session
    if spikes.ndim != 2 or spikes.shape[1] < 1:
        raise ValueError(
            "session.spikes must be a two-dimensional array with a time column"
        )
    spikes[:, 0] = _nonidentity_permuted_values(spikes[:, 0], rng)
    order = np.argsort(spikes[:, 0], kind="mergesort")
    spikes = spikes[order]
    marks = session.spike_marks
    if marks is not None:
        mark_times = np.asarray(marks.times, dtype=float).copy()
        if mark_times.shape[0] == order.shape[0]:
            mark_times = spikes[:, 0].copy()
        elif mark_times.size:
            mark_times = rng.permutation(mark_times)
        marks = ri._replace_spike_mark_rows(marks, times=mark_times, order=order)
    return replace(session, spikes=spikes, spike_marks=marks)


def _shuffle_mark_features_session_nonidentity(session, random_seed: int = 1):
    """Shuffle every variable mark feature without avoidable no-op draws."""

    marks = session.spike_marks
    if marks is None or marks.n_features == 0:
        return session
    rng = np.random.default_rng(_nonnegative_integer_seed(random_seed))
    values = np.asarray(marks.marks, dtype=float).copy()
    for column in range(values.shape[1]):
        values[:, column] = _nonidentity_permuted_values(values[:, column], rng)
    return replace(session, spike_marks=replace(marks, marks=values))


def _numeric_scope_label(value: object) -> str | None:
    if isinstance(value, (bool, np.bool_)):
        return None
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if not isinstance(value, (float, np.floating)):
        return None
    if not bool(np.isfinite(value)):
        return None
    numerator, denominator = value.as_integer_ratio()
    if denominator == 1:
        return str(numerator)
    return f"{numerator}/{denominator}"


def _nonfinite_numeric_scope_label(value: object) -> str | None:
    if isinstance(value, (bool, np.bool_, int, np.integer)) or not isinstance(
        value,
        (float, np.floating),
    ):
        return None
    if bool(np.isfinite(value)):
        return None
    return str(value).strip()


def _nonnegative_integer_value(name: str, value: object) -> int:
    """Return an exact nonnegative integer without binary64 coercion."""

    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer scalar") from exc
    if array.ndim != 0:
        raise ValueError(f"{name} must be an integer scalar")
    scalar = array.item()
    if isinstance(scalar, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer, not boolean")
    if isinstance(scalar, (str, bytes, np.str_, np.bytes_)):
        raise ValueError(f"{name} must be an integer, not string")
    try:
        integer = operator.index(scalar)
    except TypeError:
        try:
            integer = int(scalar)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must be a finite integer") from exc
        try:
            is_exact = bool(scalar == integer)
        except (TypeError, ValueError):
            is_exact = False
        if not is_exact:
            raise ValueError(f"{name} must be an integer")
    if integer < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(integer)


def _nonnegative_integer_seed(value: object) -> int:
    return _nonnegative_integer_value("random_seed", value)
