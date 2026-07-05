"""Require explicit simulation best-row flags to match numeric evidence."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

_PATCHED_FLAG = "_simulation_best_row_flag_consistency_patch_applied"


def apply_simulation_best_row_flag_consistency_patch() -> None:
    """Make stale ``is_best_model`` flags defer to numeric log evidence.

    ``simulation_best_row_flags`` deliberately accepts an explicit best flag when
    exactly one row in an event is marked.  Concatenated or edited score tables can
    carry stale flags, however.  In that case a unique flag is still unsafe unless
    the flagged row has the maximum finite comparable ``log_evidence`` in its event.
    """

    from . import simulation_best_row_flags as best_flags

    if getattr(best_flags, _PATCHED_FLAG, False):
        return

    def best_rows_with_log_evidence_consistent_flags(
        frame: pd.DataFrame,
        reporting: Any,
    ) -> pd.DataFrame:
        group_columns = best_flags._event_group_columns(frame)
        if not group_columns:
            flags = reporting._coerce_bool_series(frame["is_best_model"])
            if _single_flag_is_log_evidence_best(frame, flags, best_flags):
                return frame.loc[flags].reset_index(drop=True)
            return best_flags._best_by_log_evidence(frame)

        pieces = []
        for _, group in frame.groupby(group_columns, sort=False, dropna=False):
            flags = reporting._coerce_bool_series(group["is_best_model"])
            if _single_flag_is_log_evidence_best(group, flags, best_flags):
                pieces.append(group.loc[flags])
            else:
                pieces.append(best_flags._best_by_log_evidence(group))
        if not pieces:
            return best_flags._empty_like(frame)
        return pd.concat(pieces, ignore_index=True, sort=False)

    best_flags._best_rows_with_guarded_flags = best_rows_with_log_evidence_consistent_flags
    setattr(best_flags, _PATCHED_FLAG, True)


def _single_flag_is_log_evidence_best(
    group: pd.DataFrame,
    flags: pd.Series,
    best_flags: Any,
) -> bool:
    if int(flags.sum()) != 1:
        return False

    working = best_flags._finite_log_evidence_rows(group)
    if working.empty:
        return False

    aligned_flags = flags.reindex(working.index, fill_value=False).astype(bool)
    if int(aligned_flags.sum()) != 1:
        return False

    log_evidence = pd.to_numeric(working["log_evidence"], errors="coerce")
    flagged_log_evidence = float(log_evidence.loc[aligned_flags].iloc[0])
    max_log_evidence = float(np.max(log_evidence.to_numpy(dtype=float)))
    return bool(flagged_log_evidence == max_log_evidence)


__all__ = ["apply_simulation_best_row_flag_consistency_patch"]
