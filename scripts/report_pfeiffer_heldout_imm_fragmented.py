#!/usr/bin/env python3
"""Report true held-out-cell IMM-vs-fragmented prediction for Pfeiffer/Foster."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


IMM_MODEL = "sorted-spike-state-space-first-order-imm"
FRAGMENTED_MODEL = "sorted-spike-state-space-fragmented"
PAIR_MODELS = (IMM_MODEL, FRAGMENTED_MODEL)
DEFAULT_MARGIN_THRESHOLD = 5.5


def canonical_model(value: object) -> str:
    """Map requested or emitted model labels onto the paired exact models."""

    text = str(value).strip().lower().replace("_", "-")
    if "first-order-imm" in text:
        return IMM_MODEL
    if "fragmented" in text:
        return FRAGMENTED_MODEL
    return str(value).strip()


def build_split_decisions(
    scores: pd.DataFrame,
    *,
    frozen_clean_events: set[tuple[str, int]] | None = None,
    margin_threshold: float = DEFAULT_MARGIN_THRESHOLD,
) -> pd.DataFrame:
    """Build one paired predictive decision per event and repeated cell split."""

    required = {"session", "event_index", "cell_split_index", "model"}
    missing = sorted(required.difference(scores.columns))
    if missing:
        raise ValueError(f"scores are missing required columns: {', '.join(missing)}")
    frame = scores.copy()
    frame["canonical_model"] = frame["model"].map(canonical_model)
    frame = frame[frame["canonical_model"].isin(PAIR_MODELS)].copy()
    if frame.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    keys = ["session", "event_index", "cell_split_index"]
    for (session, event_index, split_index), group in frame.groupby(keys, sort=True):
        event_key = (str(session), int(event_index))
        success = group[_success_mask(group)].copy()
        model_rows = {
            model: success[success["canonical_model"].eq(model)] for model in PAIR_MODELS
        }
        counts = {model: int(len(model_rows[model])) for model in PAIR_MODELS}
        complete = all(counts[model] == 1 for model in PAIR_MODELS)
        imm_row = model_rows[IMM_MODEL].iloc[0] if counts[IMM_MODEL] == 1 else None
        frag_row = model_rows[FRAGMENTED_MODEL].iloc[0] if counts[FRAGMENTED_MODEL] == 1 else None

        heldout_imm = _numeric_from_row(imm_row, "heldout_log_likelihood")
        heldout_frag = _numeric_from_row(frag_row, "heldout_log_likelihood")
        train_imm = _numeric_from_row(imm_row, "train_log_likelihood")
        train_frag = _numeric_from_row(frag_row, "train_log_likelihood")
        joint_imm = _numeric_from_row(imm_row, "joint_log_likelihood")
        joint_frag = _numeric_from_row(frag_row, "joint_log_likelihood")
        heldout_delta = heldout_imm - heldout_frag
        train_delta = train_imm - train_frag
        test_spikes = _first_numeric(group, "test_spikes")

        identity_errors = []
        for heldout, train, joint in (
            (heldout_imm, train_imm, joint_imm),
            (heldout_frag, train_frag, joint_frag),
        ):
            if not all(np.isfinite(value) for value in (heldout, train, joint)):
                identity_errors.append(np.nan)
            else:
                identity_errors.append(abs(heldout - (joint - train)))
        finite_identity = [value for value in identity_errors if np.isfinite(value)]
        max_identity_error = max(finite_identity) if finite_identity else np.nan
        identity_verified = bool(
            complete
            and len(finite_identity) == 2
            and max_identity_error <= 1e-8
        )

        method = _first_text(group, "heldout_predictive_method")
        latent_column_present = "heldout_replay_spikes_used_for_latent_inference" in group
        latent_values = (
            group["heldout_replay_spikes_used_for_latent_inference"].map(_as_bool)
            if latent_column_present
            else pd.Series(dtype=bool)
        )
        heldout_used_for_latent = bool(latent_values.any()) if not latent_values.empty else False
        explicit_no_latent_use = bool(latent_column_present and not heldout_used_for_latent)

        rows.append(
            {
                "session": str(session),
                "rat": _first_text(group, "rat") or str(session).split("/")[0],
                "event_index": int(event_index),
                "cell_split_index": int(split_index),
                "cell_split_seed": _first_numeric(group, "cell_split_seed"),
                "pair_complete": bool(complete),
                "imm_success_rows": counts[IMM_MODEL],
                "fragmented_success_rows": counts[FRAGMENTED_MODEL],
                "failed_pair_model_rows": int((~_success_mask(group)).sum()),
                "heldout_logZ_first_order_imm": heldout_imm,
                "heldout_logZ_fragmented": heldout_frag,
                "heldout_delta_imm_minus_fragmented": heldout_delta,
                "heldout_delta_bits_per_spike": _safe_ratio(
                    heldout_delta,
                    test_spikes * np.log(2.0),
                ),
                "train_logZ_first_order_imm": train_imm,
                "train_logZ_fragmented": train_frag,
                "train_delta_imm_minus_fragmented": train_delta,
                "train_defined_clean_imm": bool(
                    complete
                    and np.isfinite(train_delta)
                    and train_delta >= float(margin_threshold)
                ),
                "frozen_clean_imm": bool(
                    frozen_clean_events is not None and event_key in frozen_clean_events
                ),
                "heldout_imm_raw_win": bool(np.isfinite(heldout_delta) and heldout_delta > 0.0),
                "heldout_imm_confident_win": bool(
                    complete
                    and np.isfinite(heldout_delta)
                    and heldout_delta >= float(margin_threshold)
                ),
                "heldout_fragmented_confident_win": bool(
                    complete
                    and np.isfinite(heldout_delta)
                    and heldout_delta <= -float(margin_threshold)
                ),
                "margin_threshold": float(margin_threshold),
                "train_cell_count": _first_numeric(group, "train_cell_count"),
                "test_cell_count": _first_numeric(group, "test_cell_count"),
                "train_spikes": _first_numeric(group, "train_spikes"),
                "test_spikes": test_spikes,
                "n_time": _first_numeric(group, "n_time"),
                "test_cell_fraction": _first_numeric(group, "test_cell_fraction"),
                "train_cell_ids": _first_text(group, "train_cell_ids"),
                "test_cell_ids": _first_text(group, "test_cell_ids"),
                "event_start_s": _first_numeric(group, "event_start_s"),
                "event_end_s": _first_numeric(group, "event_end_s"),
                "heldout_predictive_method": method,
                "conditional_identity_max_abs_error": max_identity_error,
                "conditional_identity_verified": identity_verified,
                "latent_use_metadata_present": bool(latent_column_present),
                "heldout_replay_spikes_used_for_latent_inference": heldout_used_for_latent,
                "explicit_no_heldout_latent_use": explicit_no_latent_use,
                "train_imm_mean_posterior_entropy": _diagnostic_value(
                    imm_row,
                    "train_diagnostic_mean_trajectory_posterior_entropy",
                ),
                "joint_imm_mean_posterior_entropy": _diagnostic_value(
                    imm_row,
                    "joint_diagnostic_mean_trajectory_posterior_entropy",
                    fallback="diagnostic_mean_trajectory_posterior_entropy",
                ),
            }
        )
    return pd.DataFrame(rows)


def add_active_cell_coverage(
    split_decisions: pd.DataFrame,
    dataset_root: str | Path | None,
) -> pd.DataFrame:
    """Count replay-active train/test cells without repeating model scoring."""

    frame = split_decisions.copy()
    columns = [
        "train_active_cell_count",
        "test_active_cell_count",
        "train_active_cell_fraction",
        "test_active_cell_fraction",
    ]
    for column in columns:
        frame[column] = np.nan
    if not dataset_root or frame.empty:
        return frame
    from hipporeplayimm.data import load_replay_session

    root = Path(dataset_root)
    for session_id, session_rows in frame.groupby("session", sort=True):
        session = load_replay_session(root / str(session_id))
        spikes = np.asarray(session.spikes, dtype=float)
        times = spikes[:, 0]
        event_active: dict[int, set[int]] = {}
        for event_index, event_rows in session_rows.groupby("event_index", sort=True):
            start = _first_numeric(event_rows, "event_start_s")
            end = _first_numeric(event_rows, "event_end_s")
            left = int(np.searchsorted(times, start, side="left"))
            right = int(np.searchsorted(times, end, side="right"))
            event_active[int(event_index)] = set(spikes[left:right, 1].astype(int).tolist())
        for index, row in session_rows.iterrows():
            active = event_active[int(row["event_index"])]
            train_ids = _parse_cell_ids(row.get("train_cell_ids", ""))
            test_ids = _parse_cell_ids(row.get("test_cell_ids", ""))
            train_active = len(active.intersection(train_ids))
            test_active = len(active.intersection(test_ids))
            frame.at[index, "train_active_cell_count"] = train_active
            frame.at[index, "test_active_cell_count"] = test_active
            frame.at[index, "train_active_cell_fraction"] = _safe_ratio(
                train_active,
                len(train_ids),
            )
            frame.at[index, "test_active_cell_fraction"] = _safe_ratio(
                test_active,
                len(test_ids),
            )
    return frame


def event_medians(split_decisions: pd.DataFrame) -> pd.DataFrame:
    """Collapse repeated splits to one primary row per event and analysis scope."""

    if split_decisions.empty:
        return pd.DataFrame()
    scopes = {
        "all_events": pd.Series(True, index=split_decisions.index),
        "train_defined_clean_imm": _bool_series(
            split_decisions,
            "train_defined_clean_imm",
        ),
        "frozen_clean_imm_sensitivity": _bool_series(
            split_decisions,
            "frozen_clean_imm",
        ),
    }
    rows: list[dict[str, object]] = []
    for scope, eligible in scopes.items():
        scoped = split_decisions[eligible & _bool_series(split_decisions, "pair_complete")]
        for (session, event_index), group in scoped.groupby(
            ["session", "event_index"],
            sort=True,
        ):
            delta = pd.to_numeric(
                group["heldout_delta_imm_minus_fragmented"],
                errors="coerce",
            ).dropna()
            bits = pd.to_numeric(group["heldout_delta_bits_per_spike"], errors="coerce")
            train_delta = pd.to_numeric(
                group["train_delta_imm_minus_fragmented"],
                errors="coerce",
            )
            rows.append(
                {
                    "scope": scope,
                    "session": str(session),
                    "rat": _first_text(group, "rat") or str(session).split("/")[0],
                    "event_index": int(event_index),
                    "margin_threshold": _first_numeric(group, "margin_threshold"),
                    "eligible_splits": int(len(group)),
                    "heldout_delta_event_median": float(delta.median()),
                    "heldout_delta_event_mean": float(delta.mean()),
                    "heldout_delta_event_std": float(delta.std(ddof=1)) if len(delta) > 1 else 0.0,
                    "heldout_delta_event_iqr": float(
                        delta.quantile(0.75) - delta.quantile(0.25)
                    ),
                    "heldout_delta_bits_per_spike_event_median": float(bits.median()),
                    "heldout_positive_split_fraction": float((delta > 0.0).mean()),
                    "heldout_confident_split_fraction": float(
                        _bool_series(group, "heldout_imm_confident_win").mean()
                    ),
                    "train_delta_event_median": float(train_delta.median()),
                    "train_clean_split_fraction": float(
                        _bool_series(group, "train_defined_clean_imm").mean()
                    ),
                    "median_train_cell_count": _median_numeric(group, "train_cell_count"),
                    "median_test_cell_count": _median_numeric(group, "test_cell_count"),
                    "median_train_spikes": _median_numeric(group, "train_spikes"),
                    "median_test_spikes": _median_numeric(group, "test_spikes"),
                    "median_train_active_cell_count": _median_numeric(
                        group,
                        "train_active_cell_count",
                    ),
                    "median_test_active_cell_count": _median_numeric(
                        group,
                        "test_active_cell_count",
                    ),
                    "median_test_active_cell_fraction": _median_numeric(
                        group,
                        "test_active_cell_fraction",
                    ),
                    "median_n_time": _median_numeric(group, "n_time"),
                    "median_train_imm_posterior_entropy": _median_numeric(
                        group,
                        "train_imm_mean_posterior_entropy",
                    ),
                    "median_joint_imm_posterior_entropy": _median_numeric(
                        group,
                        "joint_imm_mean_posterior_entropy",
                    ),
                }
            )
    return pd.DataFrame(rows)


def summarize_event_medians(event_frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize each scope with every event carrying equal weight."""

    if event_frame.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for scope, group in event_frame.groupby("scope", sort=False):
        delta = pd.to_numeric(group["heldout_delta_event_median"], errors="coerce").dropna()
        threshold = _first_numeric(group, "margin_threshold")
        bits = pd.to_numeric(
            group["heldout_delta_bits_per_spike_event_median"],
            errors="coerce",
        ).dropna()
        rows.append(
            {
                "scope": scope,
                "events": int(len(group)),
                "eligible_split_event_rows": int(
                    pd.to_numeric(group["eligible_splits"], errors="coerce").sum()
                ),
                "median_eligible_splits_per_event": _median_numeric(
                    group,
                    "eligible_splits",
                ),
                "min_eligible_splits_per_event": _min_numeric(
                    group,
                    "eligible_splits",
                ),
                "rats": int(group["rat"].nunique()),
                "sessions": int(group["session"].nunique()),
                "median_event_heldout_delta": float(delta.median()),
                "mean_event_heldout_delta": float(delta.mean()),
                "event_heldout_delta_positive_count": int((delta > 0.0).sum()),
                "event_heldout_delta_positive_fraction": float((delta > 0.0).mean()),
                "event_heldout_delta_confident_count": int(
                    (delta >= threshold).sum()
                ),
                "event_heldout_delta_confident_fraction": float(
                    (delta >= threshold).mean()
                ),
                "median_event_heldout_delta_bits_per_spike": float(bits.median()),
                "median_event_predictive_split_std": _median_numeric(
                    group,
                    "heldout_delta_event_std",
                ),
            }
        )
    return pd.DataFrame(rows)


def grouped_event_summary(event_frame: pd.DataFrame, group_column: str) -> pd.DataFrame:
    """Return event-weighted summaries by rat or session and analysis scope."""

    if event_frame.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for (scope, group_value), group in event_frame.groupby(
        ["scope", group_column],
        sort=True,
    ):
        delta = pd.to_numeric(group["heldout_delta_event_median"], errors="coerce").dropna()
        rows.append(
            {
                "scope": scope,
                group_column: group_value,
                "events": int(len(group)),
                "median_event_heldout_delta": float(delta.median()),
                "mean_event_heldout_delta": float(delta.mean()),
                "event_heldout_delta_positive_fraction": float((delta > 0.0).mean()),
                "median_event_predictive_split_std": _median_numeric(
                    group,
                    "heldout_delta_event_std",
                ),
                "median_train_cell_count": _median_numeric(group, "median_train_cell_count"),
                "median_test_cell_count": _median_numeric(group, "median_test_cell_count"),
                "median_test_spikes": _median_numeric(group, "median_test_spikes"),
                "median_test_active_cell_count": _median_numeric(
                    group,
                    "median_test_active_cell_count",
                ),
                "median_test_active_cell_fraction": _median_numeric(
                    group,
                    "median_test_active_cell_fraction",
                ),
                "median_train_imm_posterior_entropy": _median_numeric(
                    group,
                    "median_train_imm_posterior_entropy",
                ),
            }
        )
    return pd.DataFrame(rows)


def leave_one_rat_out(event_frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize event medians after excluding each rat in turn."""

    rows: list[dict[str, object]] = []
    for scope, scoped in event_frame.groupby("scope", sort=False):
        for rat in sorted(scoped["rat"].astype(str).unique()):
            retained = scoped[~scoped["rat"].astype(str).eq(rat)]
            delta = pd.to_numeric(
                retained["heldout_delta_event_median"],
                errors="coerce",
            ).dropna()
            rows.append(
                {
                    "scope": scope,
                    "excluded_rat": rat,
                    "retained_rats": int(retained["rat"].nunique()),
                    "retained_events": int(len(retained)),
                    "median_event_heldout_delta": float(delta.median()),
                    "event_heldout_delta_positive_fraction": float((delta > 0.0).mean()),
                }
            )
    return pd.DataFrame(rows)


def rat_cluster_bootstrap(
    event_frame: pd.DataFrame,
    *,
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    """Bootstrap rats, retaining event medians within sampled rats."""

    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for scope, scoped in event_frame.groupby("scope", sort=False):
        rats = np.asarray(sorted(scoped["rat"].astype(str).unique()), dtype=object)
        by_rat = {
            rat: pd.to_numeric(
                scoped.loc[scoped["rat"].astype(str).eq(str(rat)), "heldout_delta_event_median"],
                errors="coerce",
            ).dropna().to_numpy(dtype=float)
            for rat in rats
        }
        values: list[float] = []
        for _ in range(int(replicates)):
            sampled = rng.choice(rats, size=len(rats), replace=True)
            draw = np.concatenate([by_rat[str(rat)] for rat in sampled])
            values.append(float(np.median(draw)))
        series = pd.Series(values, dtype=float)
        rows.append(
            {
                "scope": scope,
                "bootstrap_unit": "rat",
                "bootstrap_replicates": int(replicates),
                "seed": int(seed),
                "rats": int(len(rats)),
                "estimate": _median_numeric(scoped, "heldout_delta_event_median"),
                "ci_low": float(series.quantile(0.025)),
                "ci_high": float(series.quantile(0.975)),
                "positive_bootstrap_fraction": float((series > 0.0).mean()),
            }
        )
    return pd.DataFrame(rows)


def rat_diagnostics(
    event_frame: pd.DataFrame,
    by_rat: pd.DataFrame,
    map_specificity: pd.DataFrame | None = None,
    run_decoder_summary: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Collect the main power/quality covariates for Rat3/Rat4 diagnosis."""

    diagnostics = by_rat[by_rat["scope"].eq("all_events")].copy()
    if map_specificity is not None and not map_specificity.empty:
        required = {"group", "metric", "median_real_minus_null_median"}
        if required.issubset(map_specificity.columns):
            pivot = map_specificity.pivot_table(
                index="group",
                columns="metric",
                values="median_real_minus_null_median",
                aggfunc="first",
            ).reset_index()
            pivot = pivot.rename(columns={"group": "rat"})
            pivot.columns = [
                column
                if column == "rat"
                else f"map_specific_real_minus_null_{column}"
                for column in pivot.columns
            ]
            diagnostics = diagnostics.merge(pivot, on="rat", how="left")
    if run_decoder_summary is not None and not run_decoder_summary.empty:
        decoder = run_decoder_summary.copy()
        if "rat" not in decoder and "session" in decoder:
            decoder["rat"] = decoder["session"].astype(str).str.split("/").str[0]
        decoder_by_rat = (
            decoder.groupby("rat", as_index=False)
            .agg(
                run_decoder_error_cm_median=(
                    "median_posterior_mean_error_cm",
                    "median",
                ),
                run_decoder_map_error_cm_median=("median_map_error_cm", "median"),
                run_decoder_sessions=("session", "nunique"),
            )
        )
        diagnostics = diagnostics.merge(decoder_by_rat, on="rat", how="left")
        diagnostics["run_decoder_error_status"] = np.where(
            pd.to_numeric(
                diagnostics["run_decoder_error_cm_median"],
                errors="coerce",
            ).notna(),
            "available",
            "missing_for_rat",
        )
    else:
        diagnostics["run_decoder_error_cm_median"] = np.nan
        diagnostics["run_decoder_map_error_cm_median"] = np.nan
        diagnostics["run_decoder_sessions"] = 0
        diagnostics["run_decoder_error_status"] = "not_available_in_current_pf_artifacts"
    diagnostics["active_cell_coverage_proxy"] = pd.to_numeric(
        diagnostics.get("median_test_active_cell_fraction"),
        errors="coerce",
    )
    return diagnostics


def covariate_correlations(event_frame: pd.DataFrame) -> pd.DataFrame:
    """Relate event-level held-out effects to power and uncertainty covariates."""

    primary = event_frame[event_frame["scope"].eq("all_events")].copy()
    predictors = [
        "median_train_cell_count",
        "median_test_cell_count",
        "median_test_spikes",
        "median_test_active_cell_count",
        "median_test_active_cell_fraction",
        "median_train_imm_posterior_entropy",
        "heldout_delta_event_std",
    ]
    rows: list[dict[str, object]] = []
    groups = [("all", primary), *primary.groupby("rat", sort=True)]
    for group_name, group in groups:
        target = pd.to_numeric(group["heldout_delta_event_median"], errors="coerce")
        for predictor in predictors:
            values = pd.to_numeric(group[predictor], errors="coerce")
            valid = target.notna() & values.notna()
            n = int(valid.sum())
            if n >= 3 and target[valid].nunique() > 1 and values[valid].nunique() > 1:
                result = spearmanr(values[valid], target[valid])
                rho = float(result.statistic)
                p_value = float(result.pvalue)
            else:
                rho = np.nan
                p_value = np.nan
            rows.append(
                {
                    "scope": "all_events",
                    "group": str(group_name),
                    "predictor": predictor,
                    "events": n,
                    "spearman_rho": rho,
                    "p_value_descriptive": p_value,
                    "interpretation": "descriptive_covariate_diagnostic_not_causal",
                }
            )
    return pd.DataFrame(rows)


def gate_summary(
    scores: pd.DataFrame,
    split_decisions: pd.DataFrame,
    event_frame: pd.DataFrame,
    bootstrap: pd.DataFrame,
    by_rat: pd.DataFrame,
    loro: pd.DataFrame,
    *,
    expected_events: int,
    expected_splits: int,
    expected_rats: int,
    min_positive_rats: int,
    expected_test_cell_fraction: float = 0.3,
) -> pd.DataFrame:
    """Return technical and predictive Gate 4 decisions."""

    rows: list[dict[str, object]] = []

    def add(category: str, gate: str, passed: bool, observed: object, criterion: str) -> None:
        rows.append(
            {
                "category": category,
                "gate": gate,
                "passed": bool(passed),
                "observed": observed,
                "criterion": criterion,
            }
        )

    failures = int((~_success_mask(scores)).sum())
    add("technical", "no_scoring_failures", failures == 0, failures, "zero failed model rows")
    complete = _bool_series(split_decisions, "pair_complete")
    add(
        "technical",
        "paired_models_complete",
        bool(not split_decisions.empty and complete.all()),
        f"{int(complete.sum())}/{len(split_decisions)}",
        "every split-event has exactly one successful IMM and fragmented row",
    )
    all_events = event_frame[event_frame["scope"].eq("all_events")].copy()
    event_count = int(len(all_events))
    add(
        "technical",
        "all_expected_events_present",
        event_count == int(expected_events),
        event_count,
        f"exactly {expected_events} events have event-median decisions",
    )
    exact_splits = bool(
        event_count > 0
        and (pd.to_numeric(all_events["eligible_splits"], errors="coerce") == int(expected_splits)).all()
    )
    add(
        "technical",
        "repeated_splits_complete",
        exact_splits,
        "" if all_events.empty else f"min={int(all_events['eligible_splits'].min())}, max={int(all_events['eligible_splits'].max())}",
        f"every event has exactly {expected_splits} repeated cell splits",
    )
    identities = _bool_series(split_decisions, "conditional_identity_verified")
    add(
        "technical",
        "conditional_predictive_identity_verified",
        bool(not split_decisions.empty and identities.all()),
        f"{int(identities.sum())}/{len(split_decisions)}",
        "heldout score equals logZ(train+heldout) minus logZ(train) for both models",
    )
    no_latent = _bool_series(split_decisions, "explicit_no_heldout_latent_use")
    add(
        "technical",
        "heldout_spikes_excluded_from_latent_inference",
        bool(not split_decisions.empty and no_latent.all()),
        f"{int(no_latent.sum())}/{len(split_decisions)}",
        "scorer metadata explicitly records no heldout replay spikes in latent inference",
    )
    fractions = pd.to_numeric(
        split_decisions.get("test_cell_fraction"),
        errors="coerce",
    ).dropna()
    fraction_matches = bool(
        not fractions.empty
        and np.allclose(
            fractions.to_numpy(dtype=float),
            float(expected_test_cell_fraction),
            atol=1e-12,
            rtol=0.0,
        )
    )
    add(
        "technical",
        "seventy_thirty_cell_split",
        fraction_matches,
        "" if fractions.empty else float(fractions.median()),
        f"held-out cell fraction is {expected_test_cell_fraction:.3f}",
    )
    rats = int(all_events["rat"].nunique()) if not all_events.empty else 0
    add(
        "technical",
        "rats_represented",
        rats == int(expected_rats),
        rats,
        f"all {expected_rats} rats represented",
    )

    delta = pd.to_numeric(all_events["heldout_delta_event_median"], errors="coerce").dropna()
    median_delta = float(delta.median()) if not delta.empty else np.nan
    positive_fraction = float((delta > 0.0).mean()) if not delta.empty else np.nan
    add("predictive", "event_median_delta_positive", median_delta > 0.0, median_delta, "median of per-event repeated-split medians > 0")
    add("predictive", "majority_events_positive", positive_fraction > 0.5, positive_fraction, "more than half of event-median heldout deltas > 0")
    primary_boot = bootstrap[bootstrap["scope"].eq("all_events")]
    ci_low = _first_numeric(primary_boot, "ci_low")
    add("predictive", "rat_bootstrap_ci_excludes_zero", ci_low > 0.0, ci_low, "rat-cluster bootstrap 95% CI lower bound > 0")
    primary_rat = by_rat[by_rat["scope"].eq("all_events")]
    rat_positive = int((pd.to_numeric(primary_rat["median_event_heldout_delta"], errors="coerce") > 0.0).sum())
    add("predictive", "at_least_three_of_four_rats_positive", rat_positive >= int(min_positive_rats), f"{rat_positive}/{expected_rats}", f"at least {min_positive_rats} of {expected_rats} rat medians > 0")
    primary_loro = loro[loro["scope"].eq("all_events")]
    loro_min = _min_numeric(primary_loro, "median_event_heldout_delta")
    add("predictive", "leave_one_rat_out_positive", loro_min > 0.0, loro_min, "all leave-one-rat-out event-median estimates > 0")

    technical = [row for row in rows if row["category"] == "technical"]
    predictive = [row for row in rows if row["category"] == "predictive"]
    add("summary", "overall_technical", all(row["passed"] for row in technical), f"{sum(row['passed'] for row in technical)}/{len(technical)}", "all technical gates pass")
    add("summary", "overall_predictive", all(row["passed"] for row in predictive), f"{sum(row['passed'] for row in predictive)}/{len(predictive)}", "all primary all-event predictive gates pass")
    add("summary", "overall", all(row["passed"] for row in technical + predictive), "pass" if all(row["passed"] for row in technical + predictive) else "fail", "technical and primary predictive gates pass")
    return pd.DataFrame(rows)


def write_report(
    outdir: Path,
    scope_summary: pd.DataFrame,
    gates: pd.DataFrame,
    by_rat: pd.DataFrame,
) -> None:
    primary = scope_summary[scope_summary["scope"].eq("all_events")]
    overall = gates[gates["gate"].eq("overall")]
    verdict = "pass" if not overall.empty and _as_bool(overall.iloc[0]["passed"]) else "fail"
    lines = [
        "# Pfeiffer/Foster held-out-cell IMM-vs-fragmented Gate 4",
        "",
        f"**Overall Gate 4 verdict:** `{verdict}`",
        "",
        "The primary analysis includes all events. Each event contributes one median across repeated",
        "70/30 train/held-out cell splits, so events with more cells or usable splits cannot dominate.",
        "The train-defined clean-IMM scope is secondary and is selected independently within each split.",
        "The previously frozen clean-IMM set is sensitivity-only because its selection used all cells.",
        "",
    ]
    if not primary.empty:
        row = primary.iloc[0]
        lines.extend(
            [
                "## Primary all-event result",
                "",
                f"- Events: {int(row['events'])}",
                f"- Median held-out IMM-fragmented delta: {float(row['median_event_heldout_delta']):.3f}",
                f"- Positive event-median fraction: {float(row['event_heldout_delta_positive_fraction']):.3f}",
                "",
            ]
        )
    lines.extend(["## Rat diagnostics", "", _markdown_table(by_rat), ""])
    lines.extend(
        [
            "## Claim boundary",
            "",
            "A positive Gate 4 supports generalization of the temporal IMM advantage across neural",
            "populations. It does not by itself make that advantage spatial-map-specific; the separate",
            "map-content dissociation and order-by-map factorial retain that responsibility.",
            "",
        ]
    )
    (outdir / "pfeiffer_heldout_imm_fragmented_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def write_figure(event_frame: pd.DataFrame, outdir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    primary = event_frame[event_frame["scope"].eq("all_events")].copy()
    if primary.empty:
        return
    rats = sorted(primary["rat"].astype(str).unique())
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    axes[0].axhline(0.0, color="black", linewidth=1)
    for index, rat in enumerate(rats):
        values = pd.to_numeric(
            primary.loc[primary["rat"].astype(str).eq(rat), "heldout_delta_event_median"],
            errors="coerce",
        ).dropna()
        jitter = np.linspace(-0.12, 0.12, len(values)) if len(values) else np.asarray([])
        axes[0].scatter(np.full(len(values), index) + jitter, values, s=18, alpha=0.75)
        axes[0].plot(index, values.median(), marker="_", markersize=18, color="black")
    axes[0].set_xticks(range(len(rats)), rats)
    axes[0].set_ylabel("Event-median held-out Δ log score\n(IMM - fragmented)")
    axes[0].set_title("Primary all-event Gate 4")

    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].scatter(
        pd.to_numeric(primary["train_delta_event_median"], errors="coerce"),
        pd.to_numeric(primary["heldout_delta_event_median"], errors="coerce"),
        c=pd.Categorical(primary["rat"]).codes,
        cmap="tab10",
        s=24,
        alpha=0.75,
    )
    axes[1].set_xlabel("Train-cell Δ log evidence")
    axes[1].set_ylabel("Held-out-cell Δ log predictive score")
    axes[1].set_title("Selection-independent predictive check")
    fig.savefig(outdir / "pfeiffer_heldout_imm_fragmented_event_medians.png", dpi=180)
    plt.close(fig)


def run(args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    scores_path = Path(args.scores).resolve()
    scores = pd.read_csv(scores_path)
    frozen = _load_frozen_events(args.frozen_clean_selection)
    split = build_split_decisions(
        scores,
        frozen_clean_events=frozen,
        margin_threshold=args.margin_threshold,
    )
    split = add_active_cell_coverage(
        split,
        getattr(args, "dataset_root", None),
    )
    events = event_medians(split)
    scope = summarize_event_medians(events)
    by_rat = grouped_event_summary(events, "rat")
    by_session = grouped_event_summary(events, "session")
    loro = leave_one_rat_out(events)
    bootstrap = rat_cluster_bootstrap(
        events,
        replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    map_specificity = (
        pd.read_csv(args.map_specificity_by_rat)
        if args.map_specificity_by_rat
        else None
    )
    run_decoder_summary = (
        pd.read_csv(args.run_decoder_summary)
        if getattr(args, "run_decoder_summary", None)
        else None
    )
    diagnostics = rat_diagnostics(
        events,
        by_rat,
        map_specificity,
        run_decoder_summary,
    )
    correlations = covariate_correlations(events)
    gates = gate_summary(
        scores,
        split,
        events,
        bootstrap,
        by_rat,
        loro,
        expected_events=args.expected_events,
        expected_splits=args.expected_splits,
        expected_rats=args.expected_rats,
        min_positive_rats=args.min_positive_rats,
        expected_test_cell_fraction=getattr(args, "expected_test_cell_fraction", 0.3),
    )

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "pfeiffer_heldout_imm_fragmented_split_decisions.csv": split,
        "pfeiffer_heldout_imm_fragmented_event_medians.csv": events,
        "pfeiffer_heldout_imm_fragmented_scope_summary.csv": scope,
        "pfeiffer_heldout_imm_fragmented_by_rat.csv": by_rat,
        "pfeiffer_heldout_imm_fragmented_by_session.csv": by_session,
        "pfeiffer_heldout_imm_fragmented_leave_one_rat_out.csv": loro,
        "pfeiffer_heldout_imm_fragmented_rat_bootstrap.csv": bootstrap,
        "pfeiffer_heldout_imm_fragmented_rat_diagnostics.csv": diagnostics,
        "pfeiffer_heldout_imm_fragmented_covariate_correlations.csv": correlations,
        "pfeiffer_heldout_imm_fragmented_gate_summary.csv": gates,
    }
    for name, frame in outputs.items():
        frame.to_csv(outdir / name, index=False)
    write_report(outdir, scope, gates, diagnostics)
    write_figure(events, outdir)
    manifest = {
        "analysis": "true_heldout_cell_predictive_imm_vs_fragmented",
        "created_at_utc": datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z"),
        "code_commit": _git_value(["rev-parse", "HEAD"]),
        "git_branch": _git_value(["branch", "--show-current"]),
        "command_line": " ".join(sys.argv),
        "scores": str(scores_path),
        "scores_sha256": _sha256(scores_path),
        "dataset_root": str(Path(args.dataset_root).resolve()) if getattr(args, "dataset_root", None) else "",
        "frozen_clean_selection": str(Path(args.frozen_clean_selection).resolve()) if args.frozen_clean_selection else "",
        "run_decoder_summary": str(Path(args.run_decoder_summary).resolve()) if getattr(args, "run_decoder_summary", None) else "",
        "margin_threshold": float(args.margin_threshold),
        "expected_events": int(args.expected_events),
        "expected_splits": int(args.expected_splits),
        "test_cell_fraction_expected": float(
            getattr(args, "expected_test_cell_fraction", 0.3)
        ),
        "primary_scope": "all_events",
        "secondary_scope": "train_defined_clean_imm",
        "sensitivity_scope": "frozen_clean_imm_sensitivity",
        "event_aggregation": "median_across_repeated_cell_splits",
        "heldout_predictive_identity": "logZ(train_plus_heldout)-logZ(train)",
    }
    (outdir / "pfeiffer_heldout_imm_fragmented_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return outputs


def _load_frozen_events(path: str | None) -> set[tuple[str, int]] | None:
    if not path:
        return None
    frame = pd.read_csv(path)
    if not {"session", "event_index"}.issubset(frame.columns):
        raise ValueError("frozen clean selection must contain session and event_index")
    return {
        (str(row.session), int(row.event_index))
        for row in frame[["session", "event_index"]].drop_duplicates().itertuples(index=False)
    }


def _parse_cell_ids(value: object) -> set[int]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return set()
    return {
        int(token)
        for token in str(value).replace(" ", ",").split(",")
        if token.strip()
    }


def _success_mask(frame: pd.DataFrame) -> pd.Series:
    if "status" not in frame:
        return pd.Series(True, index=frame.index, dtype=bool)
    return frame["status"].fillna("").astype(str).str.lower().eq("success")


def _as_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def _bool_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(False, index=frame.index, dtype=bool)
    return frame[column].map(_as_bool).astype(bool)


def _numeric_from_row(row: pd.Series | None, column: str) -> float:
    if row is None or column not in row:
        return np.nan
    return float(pd.to_numeric(pd.Series([row[column]]), errors="coerce").iloc[0])


def _diagnostic_value(
    row: pd.Series | None,
    column: str,
    *,
    fallback: str | None = None,
) -> float:
    value = _numeric_from_row(row, column)
    if np.isfinite(value) or fallback is None:
        return value
    return _numeric_from_row(row, fallback)


def _first_numeric(frame: pd.DataFrame, column: str) -> float:
    if column not in frame:
        return np.nan
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.iloc[0]) if not values.empty else np.nan


def _median_numeric(frame: pd.DataFrame, column: str) -> float:
    if column not in frame:
        return np.nan
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.median()) if not values.empty else np.nan


def _min_numeric(frame: pd.DataFrame, column: str) -> float:
    if column not in frame:
        return np.nan
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.min()) if not values.empty else np.nan


def _first_text(frame: pd.DataFrame, column: str) -> str:
    if column not in frame:
        return ""
    values = frame[column].dropna().astype(str)
    return str(values.iloc[0]) if not values.empty else ""


def _safe_ratio(value: float, denominator: float) -> float:
    if not np.isfinite(value) or not np.isfinite(denominator) or denominator == 0.0:
        return np.nan
    return float(value / denominator)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_value(args: list[str]) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"
    return result.stdout.strip() or "unavailable"


def _markdown_table(frame: pd.DataFrame) -> str:
    """Render a compact Markdown table without the optional tabulate package."""

    if frame.empty:
        return "_No rows._"
    printable = frame.copy()
    for column in printable.columns:
        printable[column] = printable[column].map(
            lambda value: ""
            if pd.isna(value)
            else f"{value:.4g}"
            if isinstance(value, (float, np.floating))
            else str(value)
        )
    columns = [str(column) for column in printable.columns]
    rows = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    rows.extend(
        "| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |"
        for row in printable.itertuples(index=False, name=None)
    )
    return "\n".join(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--dataset-root")
    parser.add_argument("--frozen-clean-selection")
    parser.add_argument("--map-specificity-by-rat")
    parser.add_argument("--run-decoder-summary")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--margin-threshold", type=float, default=DEFAULT_MARGIN_THRESHOLD)
    parser.add_argument("--expected-events", type=int, default=160)
    parser.add_argument("--expected-splits", type=int, default=20)
    parser.add_argument("--expected-rats", type=int, default=4)
    parser.add_argument("--min-positive-rats", type=int, default=3)
    parser.add_argument("--expected-test-cell-fraction", type=float, default=0.3)
    parser.add_argument("--bootstrap-replicates", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1)
    return parser


def main() -> int:
    run(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
