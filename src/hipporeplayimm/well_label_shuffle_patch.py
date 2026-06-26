"""Keep shuffled behavioral well-label columns row-aligned."""

from __future__ import annotations

import numpy as np
import pandas as pd

_PATCHED_FLAG = "_well_label_shuffle_patch_applied"
_MISSING_WELL_LABELS = {"", "<na>", "na", "n/a", "nan", "none", "null", "missing"}


def apply_well_label_shuffle_patch() -> None:
    """Install row-wise well-label shuffling."""

    from . import result_improvements

    if getattr(result_improvements, _PATCHED_FLAG, False):
        return
    result_improvements.shuffle_well_labels = shuffle_well_labels
    setattr(result_improvements, _PATCHED_FLAG, True)


def shuffle_well_labels(frame: pd.DataFrame, random_seed: int = 1) -> pd.DataFrame:
    """Shuffle complete label rows without breaking ID/coordinate links.

    Some score tables contain ``true_well_x``/``true_well_y`` columns but leave
    them missing when only the well identity is available.  Conversely, some
    coordinate-only rows may keep a missing or sentinel ``true_well_id`` column.
    Shuffle whichever label columns are present as row tuples so the null
    control is not a silent no-op for coordinate-backed labels.
    """

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
    rng = np.random.default_rng(random_seed)
    out.loc[labelled_rows, label_columns] = label_values[
        rng.permutation(label_values.shape[0])
    ]
    return out


def _labelled_well_rows(values: pd.Series) -> pd.Series:
    """Return rows whose well-ID field is an actual label, not a text sentinel."""

    present = values.notna()
    normalized = values.astype("string").str.strip().str.lower()
    return present & ~normalized.isin(_MISSING_WELL_LABELS)


def _coordinate_well_rows(frame: pd.DataFrame) -> pd.Series:
    """Return rows with a complete finite coordinate label."""

    coordinate_columns = [column for column in ("true_well_x", "true_well_y") if column in frame]
    if not coordinate_columns:
        return pd.Series(False, index=frame.index)
    numeric = frame[coordinate_columns].apply(pd.to_numeric, errors="coerce")
    finite = pd.DataFrame(
        np.isfinite(numeric.to_numpy(dtype=float)),
        index=frame.index,
        columns=coordinate_columns,
    )
    return finite.all(axis=1)


__all__ = ["apply_well_label_shuffle_patch", "shuffle_well_labels"]
