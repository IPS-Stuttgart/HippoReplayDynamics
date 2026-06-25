"""Keep shuffled behavioral well-label columns row-aligned."""

from __future__ import annotations

import numpy as np
import pandas as pd

_PATCHED_FLAG = "_well_label_shuffle_patch_applied"


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
    them missing when only the well identity is available.  The null control
    should still shuffle the available well identities in that case.  Rows
    without ``true_well_id`` remain untouched so padding/unlabelled events stay
    unlabelled.
    """

    if frame.empty or "true_well_id" not in frame:
        return frame.copy()
    out = frame.copy()
    label_columns = [column for column in ("true_well_id", "true_well_x", "true_well_y") if column in out]
    if not label_columns:
        return out

    labelled_rows = out["true_well_id"].notna()
    if not bool(labelled_rows.any()):
        return out

    label_values = out.loc[labelled_rows, label_columns].to_numpy(copy=True)
    rng = np.random.default_rng(random_seed)
    out.loc[labelled_rows, label_columns] = label_values[
        rng.permutation(label_values.shape[0])
    ]
    return out


__all__ = ["apply_well_label_shuffle_patch", "shuffle_well_labels"]
