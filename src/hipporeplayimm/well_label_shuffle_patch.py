"""Keep shuffled behavioral well-label columns row-aligned."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .result_improvement_seed_validation import _nonnegative_integer_seed

_PATCHED_FLAG = "_well_label_shuffle_patch_applied"
_MISSING_WELL_LABELS = {"", "<na>", "na", "n/a", "nan", "none", "null", "missing"}
_BYTE_BACKED_SCALARS = (bytes, bytearray, memoryview, np.bytes_)


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
    """Compare label-row sequences with pandas missing-value semantics."""

    return pd.DataFrame(left).equals(pd.DataFrame(right))


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


def _is_boolean_scalar(value: object) -> bool:
    """Return whether ``value`` is a scalar boolean, including 0-D arrays."""

    if isinstance(value, (bool, np.bool_)):
        return True
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return False
    if array.ndim != 0:
        return False
    try:
        item = array.item()
    except ValueError:
        return False
    return isinstance(item, (bool, np.bool_))


def _normalized_well_label(value: object) -> str:
    """Return normalized text, decoding byte-backed scalar containers first."""

    if isinstance(value, np.ndarray) and value.ndim == 0:
        return _normalized_well_label(value.item())
    if isinstance(value, np.generic):
        return _normalized_well_label(value.item())
    if isinstance(value, _BYTE_BACKED_SCALARS):
        value = bytes(value).decode("utf-8", errors="replace")
    return str(value).strip().lower()


def _labelled_well_rows(values: pd.Series) -> pd.Series:
    """Return rows whose well-ID field is an actual finite label."""

    present = values.notna()
    normalized = values.map(_normalized_well_label)
    boolean_ids = values.map(_is_boolean_scalar)
    exact_integer_ids = values.map(
        lambda value: isinstance(value, (int, np.integer))
        and not isinstance(value, (bool, np.bool_))
    )
    numeric = pd.Series(np.nan, index=values.index, dtype=float)
    numeric.loc[~exact_integer_ids & ~boolean_ids] = pd.to_numeric(
        values.loc[~exact_integer_ids & ~boolean_ids], errors="coerce"
    )
    numeric_present = exact_integer_ids | numeric.notna()
    numeric_values = numeric.fillna(0.0).to_numpy(dtype=float)
    finite_numeric = exact_integer_ids | pd.Series(
        np.isfinite(numeric_values), index=values.index
    )
    return (
        present
        & ~boolean_ids
        & ~normalized.isin(_MISSING_WELL_LABELS)
        & (~numeric_present | finite_numeric)
    )


def _coordinate_well_rows(frame: pd.DataFrame) -> pd.Series:
    """Return rows with a complete finite coordinate label."""

    coordinate_columns = ["true_well_x", "true_well_y"]
    if not all(column in frame for column in coordinate_columns):
        return pd.Series(False, index=frame.index)
    boolean_coordinates = frame[coordinate_columns].apply(
        lambda values: values.map(_is_boolean_scalar)
    )
    numeric_input = frame[coordinate_columns].mask(boolean_coordinates)
    numeric = numeric_input.apply(pd.to_numeric, errors="coerce")
    finite = pd.DataFrame(
        np.isfinite(numeric.to_numpy(dtype=float)),
        index=frame.index,
        columns=coordinate_columns,
    )
    finite &= ~boolean_coordinates
    return finite.all(axis=1)


__all__ = ["apply_well_label_shuffle_patch", "shuffle_well_labels"]
