#!/usr/bin/env python3
"""Build event-level replay-dynamics axes from full-core model evidence.

This is a post-hoc analysis layer, not a new decoder. It turns existing
all-session model-evidence artifacts into continuous event-level indices:

* diffusivity_index = P(diffusion) + P(fragmented) - P(stationary)
* momentum_index = P(momentum_exact_sparse) - P(diffusion)
* trajectory_family_index = P(trajectory_family) - P(stationary)
* switching_index = P(first_order_imm) - max(P(diffusion), P(momentum_exact_sparse))

The covariate model is intentionally simple and transparent: ordinary least
squares on standardized covariates with rat-cluster-robust standard errors,
plus leave-one-rat-out prediction and rat-cluster bootstrap summaries.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


STATIONARY = "sorted-spike-state-space-stationary"
DIFFUSION = "sorted-spike-state-space-diffusion"
FRAGMENTED = "sorted-spike-state-space-fragmented"
FIRST_ORDER_IMM = "sorted-spike-state-space-first-order-imm"
MOMENTUM_EXACT = "sorted-spike-state-space-momentum-exact-sparse"

REQUIRED_EXACT_CORE_MODELS: tuple[str, ...] = (
    STATIONARY,
    DIFFUSION,
    FRAGMENTED,
    FIRST_ORDER_IMM,
    MOMENTUM_EXACT,
)
TRAJECTORY_MODELS: tuple[str, ...] = (
    DIFFUSION,
    FRAGMENTED,
    FIRST_ORDER_IMM,
    MOMENTUM_EXACT,
)
DEFAULT_COVARIATES: tuple[str, ...] = (
    "pre_event_rate",
    "ripple_power",
    "theta_metric",
    "speed_pre",
    "event_duration",
    "spike_count",
)
DYNAMICS_INDICES: tuple[str, ...] = (
    "diffusivity_index",
    "momentum_index",
    "trajectory_family_index",
    "switching_index",
)

_MODEL_SHORT_NAMES = {
    STATIONARY: "stationary",
    DIFFUSION: "diffusion",
    FRAGMENTED: "fragmented",
    FIRST_ORDER_IMM: "first_order_imm",
    MOMENTUM_EXACT: "momentum_exact_sparse",
}


def _rat_from_session(session: object) -> str:
    return str(session).split("/", 1)[0]


def _parse_names(value: str | Iterable[str] | None, default: Sequence[str]) -> tuple[str, ...]:
    if value is None:
        return tuple(default)
    if isinstance(value, str):
        return tuple(part.strip() for part in value.replace(",", " ").split() if part.strip())
    return tuple(str(part) for part in value if str(part))


def _success_rows(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"session", "event_index", "model", "log_evidence"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"event-model evidence table is missing required columns: {missing}")

    out = frame.copy()
    if "status" in out.columns:
        out = out[out["status"].astype(str).eq("success")].copy()
    if "evidence_comparable" in out.columns:
        comparable = out["evidence_comparable"].map(_as_bool)
        out = out[comparable].copy()

    out["session"] = out["session"].astype(str)
    out["rat"] = out["session"].map(_rat_from_session)
    out["event_index"] = out["event_index"].astype(int)
    out["model"] = out["model"].astype(str)
    out["log_evidence"] = pd.to_numeric(out["log_evidence"], errors="coerce")
    return out.dropna(subset=["log_evidence"]).copy()


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _safe_softmax(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        return np.full(arr.shape, np.nan, dtype=float)
    shifted = arr - np.max(arr)
    exp_values = np.exp(shifted)
    total = exp_values.sum()
    if total <= 0 or not np.isfinite(total):
        return np.full(arr.shape, np.nan, dtype=float)
    return exp_values / total


def _first_numeric(group: pd.DataFrame, column: str) -> float:
    if column not in group.columns:
        return float("nan")
    values = pd.to_numeric(group[column], errors="coerce").dropna()
    if values.empty:
        return float("nan")
    return float(values.iloc[0])


def _available_covariates(frame: pd.DataFrame, covariates: Sequence[str]) -> tuple[str, ...]:
    available: list[str] = []
    for covariate in covariates:
        if covariate in frame.columns:
            values = pd.to_numeric(frame[covariate], errors="coerce")
            if values.notna().sum() >= 3 and values.nunique(dropna=True) > 1:
                available.append(covariate)
    return tuple(available)


def build_event_dynamics_axis(
    event_model_evidence: pd.DataFrame,
    *,
    covariates: Sequence[str] = DEFAULT_COVARIATES,
) -> pd.DataFrame:
    """Return one replay-dynamics-axis row per complete event."""

    evidence = _success_rows(event_model_evidence)
    rows: list[dict[str, object]] = []
    for (session, event_index), group in evidence.groupby(["session", "event_index"], sort=True):
        by_model = group.drop_duplicates("model", keep="last").set_index("model")
        missing = [model for model in REQUIRED_EXACT_CORE_MODELS if model not in by_model.index]
        logz_by_model = {
            model: (
                float(by_model.loc[model, "log_evidence"])
                if model in by_model.index
                else float("nan")
            )
            for model in REQUIRED_EXACT_CORE_MODELS
        }
        probabilities = _safe_softmax([logz_by_model[model] for model in REQUIRED_EXACT_CORE_MODELS])
        probability_by_model = dict(zip(REQUIRED_EXACT_CORE_MODELS, probabilities, strict=True))

        p_stationary = probability_by_model[STATIONARY]
        p_diffusion = probability_by_model[DIFFUSION]
        p_fragmented = probability_by_model[FRAGMENTED]
        p_first_order = probability_by_model[FIRST_ORDER_IMM]
        p_momentum = probability_by_model[MOMENTUM_EXACT]
        p_trajectory_family = float(
            p_diffusion + p_fragmented + p_first_order + p_momentum
        )

        best_trajectory_logz = np.nanmax(
            [logz_by_model[model] for model in TRAJECTORY_MODELS],
        )
        event_duration = _first_numeric(group, "event_duration")
        if not np.isfinite(event_duration):
            n_time = _first_numeric(group, "n_time")
            time_bin_s = _first_numeric(group, "time_bin_s")
            event_duration = float(n_time * time_bin_s) if np.isfinite(n_time * time_bin_s) else np.nan

        spike_count = _first_numeric(group, "spike_count")
        if not np.isfinite(spike_count):
            spike_count = _first_numeric(group, "n_spikes")
        event_spike_rate_hz = (
            float(spike_count / event_duration)
            if np.isfinite(spike_count) and np.isfinite(event_duration) and event_duration > 0
            else np.nan
        )

        row: dict[str, object] = {
            "session": session,
            "rat": _rat_from_session(session),
            "event_index": int(event_index),
            "exact_core_complete": not missing,
            "missing_exact_core_models": " ".join(missing),
            "logZ_stationary": logz_by_model[STATIONARY],
            "logZ_diffusion": logz_by_model[DIFFUSION],
            "logZ_fragmented": logz_by_model[FRAGMENTED],
            "logZ_first_order_imm": logz_by_model[FIRST_ORDER_IMM],
            "logZ_momentum_exact_sparse": logz_by_model[MOMENTUM_EXACT],
            "P_stationary": p_stationary,
            "P_diffusion": p_diffusion,
            "P_fragmented": p_fragmented,
            "P_first_order_imm": p_first_order,
            "P_momentum_exact_sparse": p_momentum,
            "P_trajectory_family": p_trajectory_family,
            "trajectory_family_margin": best_trajectory_logz - logz_by_model[STATIONARY],
            "momentum_minus_diffusion_margin": (
                logz_by_model[MOMENTUM_EXACT] - logz_by_model[DIFFUSION]
            ),
            "first_order_imm_minus_momentum_margin": (
                logz_by_model[FIRST_ORDER_IMM] - logz_by_model[MOMENTUM_EXACT]
            ),
            "diffusivity_index": p_diffusion + p_fragmented - p_stationary,
            "momentum_index": p_momentum - p_diffusion,
            "trajectory_family_index": p_trajectory_family - p_stationary,
            "switching_index": p_first_order - max(p_diffusion, p_momentum),
            "event_duration": event_duration,
            "spike_count": spike_count,
            "event_spike_rate_hz": event_spike_rate_hz,
        }
        for covariate in covariates:
            if covariate not in row:
                row[covariate] = _first_numeric(group, covariate)
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["session", "event_index"]).reset_index(drop=True)


def build_rat_dynamics_axis_summary(event_axis: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for rat, group in event_axis.groupby("rat", sort=True):
        row: dict[str, object] = {"rat": rat, "events": int(len(group))}
        for outcome in DYNAMICS_INDICES:
            values = pd.to_numeric(group[outcome], errors="coerce").dropna()
            row[f"{outcome}_mean"] = float(values.mean()) if not values.empty else np.nan
            row[f"{outcome}_median"] = float(values.median()) if not values.empty else np.nan
            row[f"{outcome}_positive_fraction"] = (
                float((values > 0).mean()) if not values.empty else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _design_matrix(
    frame: pd.DataFrame,
    covariates: Sequence[str],
    *,
    means: pd.Series | None = None,
    scales: pd.Series | None = None,
) -> tuple[np.ndarray, list[str], pd.Series, pd.Series]:
    if covariates:
        cov_frame = frame[list(covariates)].apply(pd.to_numeric, errors="coerce")
        if means is None:
            means = cov_frame.mean()
        if scales is None:
            scales = cov_frame.std(ddof=0).replace(0, 1.0)
        cov_frame = (cov_frame - means) / scales
        x = np.column_stack([np.ones(len(cov_frame)), cov_frame.to_numpy(dtype=float)])
        terms = ["intercept", *covariates]
    else:
        means = pd.Series(dtype=float)
        scales = pd.Series(dtype=float)
        x = np.ones((len(frame), 1), dtype=float)
        terms = ["intercept"]
    return x, terms, means, scales


def _fit_ols_cluster(
    frame: pd.DataFrame,
    outcome: str,
    covariates: Sequence[str],
    *,
    cluster_column: str = "rat",
) -> dict[str, object]:
    columns = [outcome, cluster_column, *covariates]
    data = frame[columns].copy()
    for column in [outcome, *covariates]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=[outcome, *covariates, cluster_column]).copy()
    if data.empty:
        return {
            "rows": pd.DataFrame(),
            "beta": np.asarray([], dtype=float),
            "terms": [],
            "means": pd.Series(dtype=float),
            "scales": pd.Series(dtype=float),
            "n_events": 0,
            "n_clusters": 0,
            "r_squared": np.nan,
        }

    y = data[outcome].to_numpy(dtype=float)
    x, terms, means, scales = _design_matrix(data, covariates)
    beta = np.linalg.pinv(x) @ y
    residuals = y - x @ beta
    rss = float(np.sum(residuals**2))
    tss = float(np.sum((y - y.mean()) ** 2))
    r_squared = float(1 - rss / tss) if tss > 0 else np.nan

    xtx_inv = np.linalg.pinv(x.T @ x)
    clusters = data[cluster_column].astype(str).to_numpy()
    unique_clusters = np.unique(clusters)
    meat = np.zeros((x.shape[1], x.shape[1]), dtype=float)
    for cluster in unique_clusters:
        mask = clusters == cluster
        x_g = x[mask]
        u_g = residuals[mask]
        score = x_g.T @ u_g
        meat += np.outer(score, score)

    n_events = len(data)
    n_terms = x.shape[1]
    n_clusters = len(unique_clusters)
    if n_clusters > 1 and n_events > n_terms:
        correction = (n_clusters / (n_clusters - 1)) * ((n_events - 1) / (n_events - n_terms))
        covariance = correction * (xtx_inv @ meat @ xtx_inv)
        standard_errors = np.sqrt(np.clip(np.diag(covariance), 0, np.inf))
    else:
        standard_errors = np.full(n_terms, np.nan, dtype=float)

    rows: list[dict[str, object]] = []
    df = max(n_clusters - 1, 1)
    for term, coefficient, standard_error in zip(terms, beta, standard_errors, strict=True):
        if np.isfinite(standard_error) and standard_error > 0:
            t_stat = float(coefficient / standard_error)
            p_value = float(2 * stats.t.sf(abs(t_stat), df=df))
        else:
            t_stat = np.nan
            p_value = np.nan
        rows.append(
            {
                "outcome": outcome,
                "term": term,
                "coefficient": float(coefficient),
                "cluster_robust_standard_error": float(standard_error),
                "t_statistic": t_stat,
                "p_value": p_value,
                "n_events": n_events,
                "n_clusters": n_clusters,
                "r_squared": r_squared,
                "standardized_covariates": term != "intercept",
            }
        )

    return {
        "rows": pd.DataFrame(rows),
        "beta": beta,
        "terms": terms,
        "means": means,
        "scales": scales,
        "n_events": n_events,
        "n_clusters": n_clusters,
        "r_squared": r_squared,
    }


def build_dynamics_axis_covariate_model(
    event_axis: pd.DataFrame,
    covariates: Sequence[str],
) -> pd.DataFrame:
    available = _available_covariates(event_axis, covariates)
    rows: list[pd.DataFrame] = []
    for outcome in DYNAMICS_INDICES:
        fit = _fit_ols_cluster(event_axis, outcome, available)
        frame = fit["rows"].copy()
        if not frame.empty:
            frame["covariates_used"] = " ".join(available)
            frame["covariates_missing"] = " ".join(
                covariate for covariate in covariates if covariate not in available
            )
        rows.append(frame)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_dynamics_axis_leave_one_rat_out(
    event_axis: pd.DataFrame,
    covariates: Sequence[str],
) -> pd.DataFrame:
    available = _available_covariates(event_axis, covariates)
    rows: list[dict[str, object]] = []
    for outcome in DYNAMICS_INDICES:
        for held_out_rat, test in event_axis.groupby("rat", sort=True):
            train = event_axis[event_axis["rat"].ne(held_out_rat)].copy()
            fit = _fit_ols_cluster(train, outcome, available)
            if fit["n_events"] == 0:
                rows.append(
                    {
                        "outcome": outcome,
                        "held_out_rat": held_out_rat,
                        "train_events": 0,
                        "test_events": int(len(test)),
                        "covariates_used": " ".join(available),
                        "train_r_squared": np.nan,
                        "test_correlation": np.nan,
                        "test_rmse": np.nan,
                        "test_mean_error": np.nan,
                    }
                )
                continue

            test_data = test[[outcome, *available]].copy()
            for column in [outcome, *available]:
                test_data[column] = pd.to_numeric(test_data[column], errors="coerce")
            test_data = test_data.dropna(subset=[outcome, *available]).copy()
            if test_data.empty:
                test_correlation = np.nan
                test_rmse = np.nan
                test_mean_error = np.nan
            else:
                x_test, _, _, _ = _design_matrix(
                    test_data,
                    available,
                    means=fit["means"],
                    scales=fit["scales"],
                )
                y_test = test_data[outcome].to_numpy(dtype=float)
                prediction = x_test @ fit["beta"]
                error = prediction - y_test
                if len(y_test) > 1 and np.std(prediction) > 0 and np.std(y_test) > 0:
                    test_correlation = float(np.corrcoef(prediction, y_test)[0, 1])
                else:
                    test_correlation = np.nan
                test_rmse = float(np.sqrt(np.mean(error**2)))
                test_mean_error = float(np.mean(error))

            rows.append(
                {
                    "outcome": outcome,
                    "held_out_rat": held_out_rat,
                    "train_events": int(fit["n_events"]),
                    "test_events": int(len(test_data)),
                    "covariates_used": " ".join(available),
                    "train_r_squared": fit["r_squared"],
                    "test_correlation": test_correlation,
                    "test_rmse": test_rmse,
                    "test_mean_error": test_mean_error,
                }
            )
    return pd.DataFrame(rows)


def build_dynamics_axis_bootstrap_summary(
    event_axis: pd.DataFrame,
    *,
    replicates: int = 2000,
    random_seed: int = 1,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)
    rats = tuple(sorted(event_axis["rat"].astype(str).unique()))
    rows: list[dict[str, object]] = []
    if not rats or replicates <= 0:
        return pd.DataFrame()

    for outcome in DYNAMICS_INDICES:
        values = pd.to_numeric(event_axis[outcome], errors="coerce").dropna()
        observed_mean = float(values.mean()) if not values.empty else np.nan
        observed_median = float(values.median()) if not values.empty else np.nan
        boot_mean: list[float] = []
        boot_median: list[float] = []
        for _ in range(replicates):
            sampled = rng.choice(rats, size=len(rats), replace=True)
            sample = pd.concat(
                [event_axis[event_axis["rat"].astype(str).eq(rat)] for rat in sampled],
                ignore_index=True,
            )
            sample_values = pd.to_numeric(sample[outcome], errors="coerce").dropna()
            if sample_values.empty:
                continue
            boot_mean.append(float(sample_values.mean()))
            boot_median.append(float(sample_values.median()))
        for statistic, observed, boot in (
            ("mean", observed_mean, boot_mean),
            ("median", observed_median, boot_median),
        ):
            arr = np.asarray(boot, dtype=float)
            rows.append(
                {
                    "outcome": outcome,
                    "statistic": statistic,
                    "observed": observed,
                    "ci95_low": float(np.quantile(arr, 0.025)) if arr.size else np.nan,
                    "ci95_high": float(np.quantile(arr, 0.975)) if arr.size else np.nan,
                    "bootstrap_replicates": int(arr.size),
                    "bootstrap_unit": "rat",
                    "random_seed": random_seed,
                }
            )
    return pd.DataFrame(rows)


def build_dynamics_axis_gate_summary(
    event_axis: pd.DataFrame,
    covariate_model: pd.DataFrame,
    leave_one_rat_out: pd.DataFrame,
    bootstrap: pd.DataFrame,
    covariates: Sequence[str],
) -> pd.DataFrame:
    available = _available_covariates(event_axis, covariates)
    missing = tuple(covariate for covariate in covariates if covariate not in available)
    required_rows = [
        {
            "gate": "event_rows_present",
            "passed": len(event_axis) > 0,
            "observed": str(len(event_axis)),
            "criterion": "at least one event-level dynamics row",
            "required_for_overall": True,
        },
        {
            "gate": "exact_core_complete",
            "passed": bool(event_axis["exact_core_complete"].all()) if len(event_axis) else False,
            "observed": (
                f"{int(event_axis['exact_core_complete'].sum())}/{len(event_axis)}"
                if "exact_core_complete" in event_axis
                else "0/0"
            ),
            "criterion": "all event rows have the five required exact-core models",
            "required_for_overall": True,
        },
        {
            "gate": "dynamics_indices_finite",
            "passed": bool(event_axis[list(DYNAMICS_INDICES)].notna().all().all())
            if len(event_axis)
            else False,
            "observed": (
                f"{int(event_axis[list(DYNAMICS_INDICES)].notna().all(axis=1).sum())}/{len(event_axis)}"
                if len(event_axis)
                else "0/0"
            ),
            "criterion": "all dynamics indices are finite",
            "required_for_overall": True,
        },
        {
            "gate": "at_least_one_covariate_available",
            "passed": len(available) > 0,
            "observed": " ".join(available),
            "criterion": "at least one numeric covariate is available for regression",
            "required_for_overall": True,
        },
        {
            "gate": "covariate_models_fit",
            "passed": set(covariate_model["outcome"].unique()) == set(DYNAMICS_INDICES)
            if not covariate_model.empty
            else False,
            "observed": (
                " ".join(sorted(covariate_model["outcome"].unique()))
                if not covariate_model.empty
                else ""
            ),
            "criterion": "covariate model has rows for every dynamics index",
            "required_for_overall": True,
        },
        {
            "gate": "leave_one_rat_out_rows_present",
            "passed": not leave_one_rat_out.empty,
            "observed": str(len(leave_one_rat_out)),
            "criterion": "leave-one-rat-out prediction rows are present",
            "required_for_overall": True,
        },
        {
            "gate": "bootstrap_rows_present",
            "passed": not bootstrap.empty,
            "observed": str(len(bootstrap)),
            "criterion": "rat-bootstrap axis summaries are present",
            "required_for_overall": True,
        },
        {
            "gate": "requested_external_covariates_present",
            "passed": not missing,
            "observed": f"missing={' '.join(missing)}",
            "criterion": "informational: requested external covariates are present",
            "required_for_overall": False,
        },
    ]
    required = [row for row in required_rows if row["required_for_overall"]]
    required_passed = sum(bool(row["passed"]) for row in required)
    rows = list(required_rows)
    rows.append(
        {
            "gate": "overall",
            "passed": required_passed == len(required),
            "observed": f"{required_passed}/{len(required)} required gates passed",
            "criterion": "all required dynamics-axis construction gates pass",
            "required_for_overall": True,
        }
    )
    return pd.DataFrame(rows)


def write_dynamics_axis_pack(
    event_model_evidence: pd.DataFrame,
    output: str | Path,
    *,
    covariates: Sequence[str] = DEFAULT_COVARIATES,
    bootstrap_replicates: int = 2000,
    bootstrap_random_seed: int = 1,
) -> dict[str, pd.DataFrame]:
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)

    event_axis = build_event_dynamics_axis(event_model_evidence, covariates=covariates)
    covariate_model = build_dynamics_axis_covariate_model(event_axis, covariates)
    leave_one_rat_out = build_dynamics_axis_leave_one_rat_out(event_axis, covariates)
    bootstrap = build_dynamics_axis_bootstrap_summary(
        event_axis,
        replicates=bootstrap_replicates,
        random_seed=bootstrap_random_seed,
    )
    outputs = {
        "event_dynamics_axis.csv": event_axis,
        "rat_dynamics_axis_summary.csv": build_rat_dynamics_axis_summary(event_axis),
        "dynamics_axis_covariate_model.csv": covariate_model,
        "dynamics_axis_leave_one_rat_out.csv": leave_one_rat_out,
        "dynamics_axis_bootstrap_summary.csv": bootstrap,
        "dynamics_axis_gate_summary.csv": build_dynamics_axis_gate_summary(
            event_axis,
            covariate_model,
            leave_one_rat_out,
            bootstrap,
            covariates,
        ),
    }
    for filename, frame in outputs.items():
        frame.to_csv(out / filename, index=False)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--event-model-evidence",
        required=True,
        help="Path to all_sessions_event_model_evidence.csv or event_model_evidence.csv.",
    )
    parser.add_argument("--output", required=True, help="Output directory.")
    parser.add_argument(
        "--covariates",
        default=" ".join(DEFAULT_COVARIATES),
        help=(
            "Whitespace or comma separated covariates to test. Missing covariates are "
            "reported in the gate summary and omitted from fitted models."
        ),
    )
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=2000,
        help="Rat-cluster bootstrap replicates for dynamics-axis summaries.",
    )
    parser.add_argument(
        "--bootstrap-random-seed",
        type=int,
        default=1,
        help="Random seed for rat-cluster bootstrap.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    evidence = pd.read_csv(args.event_model_evidence, low_memory=False)
    covariates = _parse_names(args.covariates, DEFAULT_COVARIATES)
    write_dynamics_axis_pack(
        evidence,
        args.output,
        covariates=covariates,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_random_seed=args.bootstrap_random_seed,
    )
    print(f"Wrote replay dynamics-axis pack to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
