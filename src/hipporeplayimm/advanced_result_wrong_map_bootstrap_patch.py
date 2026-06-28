"""Use statistic-specific rat clusters in wrong-map bootstrap summaries."""

from __future__ import annotations

from functools import wraps

import numpy as np
import pandas as pd

_PATCHED_FLAG = "_wrong_map_rat_bootstrap_patch_applied"


def _bootstrap_summary_columns() -> list[str]:
    return [
        "bootstrap_unit",
        "bootstrap_replicates",
        "random_seed",
        "statistic",
        "observed_events",
        "observed_rats",
        "observed_positive_delta_fraction",
        "positive_delta_fraction_ci95_low",
        "positive_delta_fraction_ci95_high",
        "observed_mean_delta_map_log_evidence",
        "mean_delta_ci95_low",
        "mean_delta_ci95_high",
        "probability_mean_delta_gt_0",
        "observed_median_delta_map_log_evidence",
        "median_delta_ci95_low",
        "median_delta_ci95_high",
        "probability_median_delta_gt_0",
        "most_common_selected_model",
    ]


def _positive_integer(value: object, name: str) -> int:
    """Return a positive integer argument or raise a clear validation error."""

    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if not np.isfinite(numeric) or numeric <= 0.0 or not numeric.is_integer():
        raise ValueError(f"{name} must be a positive integer")
    return int(numeric)


def _nonnegative_integer(value: object, name: str) -> int:
    """Return a nonnegative integer argument without boolean or float truncation."""

    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite nonnegative integer")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite nonnegative integer") from exc
    if not np.isfinite(numeric) or numeric < 0.0 or not numeric.is_integer():
        raise ValueError(f"{name} must be a finite nonnegative integer")
    return int(numeric)


def apply_wrong_map_rat_bootstrap_patch() -> None:
    """Patch rat-cluster bootstrap to sample only rats present for each statistic."""

    from . import advanced_result_diagnostics as diagnostics

    if getattr(diagnostics, _PATCHED_FLAG, False):
        return

    original = diagnostics.rat_bootstrap_wrong_map_absolute_evidence_summary

    @wraps(original)
    def rat_bootstrap_wrong_map_absolute_evidence_summary(
        deltas: pd.DataFrame,
        *,
        n_bootstrap: int = 2000,
        random_seed: int = 1,
    ) -> pd.DataFrame:
        """Rat-cluster bootstrap uncertainty for absolute map sensitivity."""

        replicates = _positive_integer(n_bootstrap, "n_bootstrap")
        seed = _nonnegative_integer(random_seed, "random_seed")
        columns = _bootstrap_summary_columns()
        if deltas.empty or "session" not in deltas.columns:
            return pd.DataFrame(columns=columns)
        frame = diagnostics._with_rat(deltas)
        rng = np.random.default_rng(seed)
        rows: list[dict[str, object]] = []
        for statistic, group in frame.groupby("statistic", sort=False):
            statistic_rats = sorted(group["rat"].dropna().astype(str).unique())
            if not statistic_rats:
                continue
            observed = diagnostics._wrong_map_delta_summary(group).iloc[0]
            positive_fractions: list[float] = []
            means: list[float] = []
            medians: list[float] = []
            by_rat = {
                rat: group[group["rat"].astype(str) == rat]
                for rat in statistic_rats
            }
            for _ in range(replicates):
                sampled = rng.choice(statistic_rats, size=len(statistic_rats), replace=True)
                sample = pd.concat([by_rat[rat] for rat in sampled], ignore_index=True)
                values = sample["delta_map_log_evidence"].to_numpy(float)
                positive_fractions.append(float(np.mean(values > 0.0)))
                means.append(float(np.mean(values)))
                medians.append(float(np.median(values)))
            rows.append(
                {
                    "bootstrap_unit": "rat",
                    "bootstrap_replicates": replicates,
                    "random_seed": seed,
                    "statistic": str(statistic),
                    "observed_events": int(observed["events"]),
                    "observed_rats": int(len(statistic_rats)),
                    "observed_positive_delta_fraction": float(observed["positive_delta_fraction"]),
                    "positive_delta_fraction_ci95_low": diagnostics._quantile(positive_fractions, 0.025),
                    "positive_delta_fraction_ci95_high": diagnostics._quantile(positive_fractions, 0.975),
                    "observed_mean_delta_map_log_evidence": float(observed["mean_delta_map_log_evidence"]),
                    "mean_delta_ci95_low": diagnostics._quantile(means, 0.025),
                    "mean_delta_ci95_high": diagnostics._quantile(means, 0.975),
                    "probability_mean_delta_gt_0": float(np.mean(np.asarray(means) > 0.0)),
                    "observed_median_delta_map_log_evidence": float(observed["median_delta_map_log_evidence"]),
                    "median_delta_ci95_low": diagnostics._quantile(medians, 0.025),
                    "median_delta_ci95_high": diagnostics._quantile(medians, 0.975),
                    "probability_median_delta_gt_0": float(np.mean(np.asarray(medians) > 0.0)),
                    "most_common_selected_model": str(observed["most_common_selected_model"]),
                }
            )
        return pd.DataFrame(rows, columns=columns)

    diagnostics.rat_bootstrap_wrong_map_absolute_evidence_summary = (
        rat_bootstrap_wrong_map_absolute_evidence_summary
    )
    setattr(diagnostics, _PATCHED_FLAG, True)


__all__ = ["apply_wrong_map_rat_bootstrap_patch"]
