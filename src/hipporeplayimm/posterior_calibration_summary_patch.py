"""Keep posterior-calibration summary denominators aligned."""

from __future__ import annotations

import numpy as np
import pandas as pd

_PATCHED_FLAG = "_posterior_calibration_summary_patch_applied"


def apply_posterior_calibration_summary_patch() -> None:
    """Patch calibration summaries to drop invalid probability rows consistently."""

    from . import result_improvements

    if getattr(result_improvements, _PATCHED_FLAG, False):
        return

    def posterior_calibration_summary(
        samples: pd.DataFrame,
        *,
        probability_column: str = "true_bin_probability",
        rank_column: str = "true_bin_rank",
        n_bins_column: str = "n_position_bins",
    ) -> pd.DataFrame:
        if samples.empty or probability_column not in samples:
            return pd.DataFrame()
        group_columns = [column for column in ("session", "model") if column in samples]
        if not group_columns:
            group_columns = ["_all"]
            samples = samples.copy()
            samples["_all"] = "all"

        frame = samples.copy()
        raw_probabilities = pd.to_numeric(frame[probability_column], errors="coerce")
        raw_probability_values = raw_probabilities.to_numpy(dtype=float)
        valid_probability = pd.Series(
            np.isfinite(raw_probability_values)
            & (raw_probability_values >= 0.0)
            & (raw_probability_values <= 1.0),
            index=frame.index,
        )
        frame = frame.loc[valid_probability].copy()
        if frame.empty:
            return pd.DataFrame()

        probabilities = pd.to_numeric(frame[probability_column], errors="coerce").clip(
            lower=np.finfo(float).tiny,
            upper=1.0,
        )
        frame[probability_column] = probabilities
        frame["true_negative_log_probability"] = -np.log(probabilities)
        if rank_column in frame and n_bins_column in frame:
            rank = pd.to_numeric(frame[rank_column], errors="coerce")
            n_bins = pd.to_numeric(frame[n_bins_column], errors="coerce")
            frame["rank_fraction"] = rank / n_bins
            frame["coverage_50_rank"] = frame["rank_fraction"] <= 0.50
            frame["coverage_80_rank"] = frame["rank_fraction"] <= 0.80
            frame["coverage_95_rank"] = frame["rank_fraction"] <= 0.95
        else:
            frame["rank_fraction"] = np.nan
            frame["coverage_50_rank"] = np.nan
            frame["coverage_80_rank"] = np.nan
            frame["coverage_95_rank"] = np.nan
        return (
            frame.groupby(group_columns, as_index=False)
            .agg(
                rows=(probability_column, "count"),
                mean_true_probability=(probability_column, "mean"),
                median_true_probability=(probability_column, "median"),
                mean_true_negative_log_probability=("true_negative_log_probability", "mean"),
                median_rank_fraction=("rank_fraction", "median"),
                coverage_50_rank=("coverage_50_rank", "mean"),
                coverage_80_rank=("coverage_80_rank", "mean"),
                coverage_95_rank=("coverage_95_rank", "mean"),
            )
            .reset_index(drop=True)
        )

    result_improvements.posterior_calibration_summary = posterior_calibration_summary
    setattr(result_improvements, _PATCHED_FLAG, True)


__all__ = ["apply_posterior_calibration_summary_patch"]
