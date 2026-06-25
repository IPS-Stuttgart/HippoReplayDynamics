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
    """Shuffle complete well-label tuples without breaking ID/coordinate links."""

    if frame.empty or "true_well_id" not in frame:
        return frame.copy()
    out = frame.copy()
    label_columns = [column for column in ("true_well_id", "true_well_x", "true_well_y") if column in out]
    if not label_columns:
        return out

    complete_labels = out[label_columns].notna().all(axis=1)
    if not bool(complete_labels.any()):
        return out

    label_values = out.loc[complete_labels, label_columns].to_numpy(copy=True)
    rng = np.random.default_rng(random_seed)
    out.loc[complete_labels, label_columns] = label_values[
        rng.permutation(label_values.shape[0])
    ]
    return out


__all__ = ["apply_well_label_shuffle_patch", "shuffle_well_labels"]
