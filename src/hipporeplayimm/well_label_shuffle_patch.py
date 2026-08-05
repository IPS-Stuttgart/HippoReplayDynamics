"""Keep shuffled behavioral well-label columns row-aligned."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .result_improvement_seed_validation import _nonnegative_integer_seed

_PATCHED_FLAG = "_well_label_shuffle_patch_applied"
_MISSING_WELL_LABELS = {"", "<na>", "na", "n/a", "nan", "none", "null", "missing"}
_BYTE_BACKED_SCALARS = (bytes, bytearray, memoryview, np.bytes_)
_INVALID_SCALAR = object()


def apply_well_label_shuffle_patch() -> None:
    """Install row-wise well-label shuffling."""

    from . import (
        result_improvement_seed_validation,
        result_improvement_split_validation,
        result_improvements,
    )

    result_improvement_seed_validation.apply_result_improvement_seed_validation_patch()
    result_improvement_split_validation.apply_result_improvement_split_validation_patch()
    if (
        getattr(result_improvements, _PATCHED_FLAG, False)
        and getattr(result_improvements, "shuffle_well_labels", None) is shuffle_well_labels
    ):
        return
    result_improvements.shuffle_well_labels = shuffle_well_labels
    setattr(result_improvements, _PATCHED_FLAG, True)


def _nonnegative_integer_value(name: str, value: object) -> int:
    """Return an exact nonnegative integer seed under the shared seed policy."""

    return _nonnegative_integer_seed(value, name)


def _row_sequences_equal(left: np.ndarray, right: np.ndarray) -> bool:
    """Compare label-row sequences with scalar missing-value semantics."""

    left_values = np.asarray(left, dtype=object)
    right_values = np.asarray(right, dtype=object)
    if left_values.shape != right_values.shape:
        return False
    return all(
        _scalar_values_equal(left_value, right_value)
        for left_value, right_value in zip(
            left_values.reshape(-1),
            right_values.reshape(-1),
            strict=True,
        )
    )


def _scalar_values_equal(left: object, right: object) -> bool:
    """Compare scalar labels while treating paired missing values as equal."""

    if left is right:
        return True
    try:
        left_missing = pd.isna(left)
        right_missing = pd.isna(right)
    except (TypeError, ValueError):
        left_missing = right_missing = False
    left_is_missing = isinstance(left_missing, (bool, np.bool_)) and bool(
        left_missing
    )
    right_is_missing = isinstance(right_missing, (bool, np.bool_)) and bool(
        right_missing
    )
    if left_is_missing or right_is_missing:
        return left_is_missing and right_is_missing
    try:
        equal = left == right
    except (TypeError, ValueError):
        return False
    return isinstance(equal, (bool, np.bool_)) and bool(equal)


def _nonidentity_permuted_rows(
    values: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Permute label rows until their observed sequence changes when possible."""

    original = np.asarray(values)
    if original.ndim != 2:
        raise ValueError("well-label values must be two-dimensional")
    if original.shape[0] <= 1:
        return original.copy()

    reference = original[:1]
    has_distinct_row = any(
        not _row_sequences_equal(original[index : index + 1], reference)
        for index in range(1, original.shape[0])
    )
    if not has_distinct_row:
        return original.copy()

    while True:
        permuted = original[rng.permutation(original.shape[0])]
        if not _row_sequences_equal(permuted, original):
            return permuted


def shuffle_well_labels(frame: pd.DataFrame, random_seed: int = 1) -> pd.DataFrame:
    """Shuffle complete label rows without breaking ID/coordinate links.

    Some score tables contain ``true_well_x``/``true_well_y`` columns but leave
    them missing when only the well identity is available.  Conversely, some
    coordinate-only rows may keep a missing or sentinel ``true_well_id`` column.
    Shuffle whichever label columns are present as row tuples so the null
    control is not a silent no-op for coordinate-backed labels.  When session
    metadata are available, keep permutations within each session so labels and
    coordinates from distinct environments cannot be mixed.
    """

    seed = _nonnegative_integer_value("random_seed", random_seed)
    if frame.empty:
        return frame.copy()
    out = frame.copy()
    label_columns = [column for column in ("true_well_id", "true_well_x", "true_well_y") if column in out]
    if not label_columns:
        return out

    labelled_rows = pd.Series(False, index=out.index)
    if "true_well_id" in out:
        labelled_rows |= _labelled_well_rows(out["true_well_id"])
    labelled_rows |= _coordinate_well_rows(out)
    malformed_rows = pd.Series(False, index=out.index)
    for column in label_columns:
        malformed_rows |= _malformed_scalar_rows(out[column])
    labelled_rows &= ~malformed_rows
    if not bool(labelled_rows.any()):
        return out

    label_values = out.loc[labelled_rows, label_columns].to_numpy(copy=True)
    rng = np.random.default_rng(seed)
    if "session" not in out:
        shuffled_values = _nonidentity_permuted_rows(label_values, rng)
    else:
        shuffled_values = label_values.copy()
        session_values = pd.DataFrame(
            {"session": out.loc[labelled_rows, "session"].to_numpy(copy=False)}
        )
        for positions in session_values.groupby(
            "session", sort=False, dropna=False
        ).indices.values():
            positions = np.asarray(positions, dtype=int)
            shuffled_values[positions] = _nonidentity_permuted_rows(
                label_values[positions],
                rng,
            )
    out.loc[labelled_rows, label_columns] = shuffled_values
    return out


def _unwrap_scalar_container(value: object) -> object:
    """Unwrap 0-D NumPy scalar containers without following cycles."""

    current = value
    seen_container_ids: set[int] = set()
    while isinstance(current, (np.ndarray, np.generic)):
        if isinstance(current, np.ndarray):
            if current.ndim != 0:
                return current
            container_id = id(current)
            if container_id in seen_container_ids:
                return _INVALID_SCALAR
            seen_container_ids.add(container_id)
        try:
            nested = current.item()
        except (TypeError, ValueError):
            return _INVALID_SCALAR
        if nested is current:
            return _INVALID_SCALAR
        current = nested
    return current


def _map_object_series(values: pd.Series, function) -> pd.Series:
    """Map values while preserving arbitrary-precision Python objects."""

    return pd.Series(
        [function(value) for value in values.to_numpy(dtype=object)],
        index=values.index,
        dtype=object,
    )


def _malformed_scalar_rows(values: pd.Series) -> pd.Series:
    """Return rows containing cyclic or complex scalar label values."""

    scalar_values = _map_object_series(values, _unwrap_scalar_container)
    return scalar_values.map(
        lambda value: value is _INVALID_SCALAR or _is_complex_scalar(value)
    ).astype(bool)


def _is_boolean_scalar(value: object) -> bool:
    """Return whether ``value`` is a scalar boolean, including 0-D arrays."""

    value = _unwrap_scalar_container(value)
    return isinstance(value, (bool, np.bool_))


def _is_complex_scalar(value: object) -> bool:
    """Return whether ``value`` is a scalar complex number."""

    value = _unwrap_scalar_container(value)
    if isinstance(value, (complex, np.complexfloating)):
        return True
    return isinstance(value, np.ndarray) and np.issubdtype(
        value.dtype,
        np.complexfloating,
    )


def _coerce_real_numeric_scalar(value: object) -> float:
    """Return one real numeric scalar or NaN for malformed coordinates."""

    value = _unwrap_scalar_container(value)
    if value is _INVALID_SCALAR or isinstance(
        value,
        (bool, np.bool_, complex, np.complexfloating),
    ):
        return float("nan")
    if isinstance(value, np.ndarray):
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return float("nan")


def _normalized_well_label(value: object) -> str:
    """Return normalized text, decoding byte-backed scalar containers first."""

    value = _unwrap_scalar_container(value)
    if value is _INVALID_SCALAR:
        return "missing"
    if isinstance(value, _BYTE_BACKED_SCALARS):
        value = bytes(value).decode("utf-8", errors="replace")
    return str(value).strip().lower()


def _labelled_well_rows(values: pd.Series) -> pd.Series:
    """Return rows whose well-ID field is an actual finite label."""

    scalar_values = _map_object_series(values, _unwrap_scalar_container)
    invalid_ids = scalar_values.map(lambda value: value is _INVALID_SCALAR)
    present = values.notna() & ~invalid_ids
    normalized = scalar_values.map(_normalized_well_label)
    boolean_ids = scalar_values.map(_is_boolean_scalar)
    complex_ids = scalar_values.map(_is_complex_scalar)
    exact_integer_ids = scalar_values.map(
        lambda value: isinstance(value, (int, np.integer))
        and not isinstance(value, (bool, np.bool_))
    )
    numeric = pd.Series(np.nan, index=values.index, dtype=float)
    numeric_input = ~exact_integer_ids & ~boolean_ids & ~complex_ids & ~invalid_ids
    numeric.loc[numeric_input] = pd.to_numeric(
        scalar_values.loc[numeric_input], errors="coerce"
    )
    numeric_present = exact_integer_ids | numeric.notna()
    numeric_values = numeric.fillna(0.0).to_numpy(dtype=float)
    finite_numeric = exact_integer_ids | pd.Series(
        np.isfinite(numeric_values), index=values.index
    )
    return (
        present
        & ~boolean_ids
        & ~complex_ids
        & ~normalized.isin(_MISSING_WELL_LABELS)
        & (~numeric_present | finite_numeric)
    )


def _coordinate_well_rows(frame: pd.DataFrame) -> pd.Series:
    """Return rows with a complete finite coordinate label."""

    coordinate_columns = ["true_well_x", "true_well_y"]
    if not all(column in frame for column in coordinate_columns):
        return pd.Series(False, index=frame.index)
    numeric = frame[coordinate_columns].apply(
        lambda values: values.map(_coerce_real_numeric_scalar)
    )
    finite = pd.DataFrame(
        np.isfinite(numeric.to_numpy(dtype=float)),
        index=frame.index,
        columns=coordinate_columns,
    )
    return finite.all(axis=1)


__all__ = ["apply_well_label_shuffle_patch", "shuffle_well_labels"]
