#!/usr/bin/env python3
"""Test whether native ripple strength indexes cross-cell replay coordination.

This script does not rescore replay events. It joins the native Pfeiffer/Foster
ripple-power columns to the frozen, leakage-free repeated cell-split event table
and tests a predeclared conditional association.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _provenance import build_script_provenance  # noqa: E402
from hipporeplayimm.data import load_replay_session  # noqa: E402


PRIMARY_X = "ripple_power_z_epoch"
PRIMARY_Y = "real_frozen_heldout_delta_imm_minus_fragmented_event_median"
MAP_SPECIFIC_Y = "map_specific_frozen_heldout_delta_event_median"
CONTENT_Y = "train_map_specific_nonstationary_mass_event_median"
ABSOLUTE_CONTENT_Y = "median_real_train_nonstationary_mass"

QUALITY_CONTROLS = (
    "log1p_median_train_cell_count",
    "log1p_median_test_spikes",
    "median_real_imm_train_posterior_entropy",
    "log1p_median_n_time",
)
PRIMARY_CONTROLS = (*QUALITY_CONTROLS, CONTENT_Y)
EXTENDED_PRIMARY_CONTROLS = (
    *PRIMARY_CONTROLS,
    "log1p_median_train_spikes",
    ABSOLUTE_CONTENT_Y,
)

EVENT_OUTPUT = "pfeiffer_ripple_cross_cell_coordination_events.csv"
ASSOCIATION_OUTPUT = "pfeiffer_ripple_cross_cell_coordination_associations.csv"
BY_RAT_OUTPUT = "pfeiffer_ripple_cross_cell_coordination_by_rat.csv"
LOO_OUTPUT = "pfeiffer_ripple_cross_cell_coordination_leave_one_rat_out.csv"
BOOTSTRAP_OUTPUT = "pfeiffer_ripple_cross_cell_coordination_bootstrap.csv"
PERMUTATION_OUTPUT = "pfeiffer_ripple_cross_cell_coordination_permutation_null.csv"
DISSOCIATION_OUTPUT = "pfeiffer_ripple_cross_cell_coordination_dissociation.csv"
GATE_OUTPUT = "pfeiffer_ripple_cross_cell_coordination_gate_summary.csv"
FIGURE_OUTPUT = "pfeiffer_ripple_cross_cell_coordination_figure.png"
MANIFEST_OUTPUT = "pfeiffer_ripple_cross_cell_coordination_manifest.json"
REPORT_OUTPUT = "pfeiffer_ripple_cross_cell_coordination_report.md"


def _finite(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    keep = np.isfinite(out[list(columns)].to_numpy(dtype=float)).all(axis=1)
    return out.loc[keep].copy()


def _rank(values: pd.Series) -> np.ndarray:
    return values.rank(method="average").to_numpy(dtype=float)


def _rank_residuals(
    frame: pd.DataFrame,
    value: str,
    controls: Sequence[str],
    *,
    fixed_effect: str = "session",
) -> tuple[np.ndarray, pd.Index]:
    columns = [value, *controls]
    sub = _finite(frame, columns)
    ranked = _rank(sub[value])
    design_parts = [np.ones((len(sub), 1), dtype=float)]
    if controls:
        design_parts.append(np.column_stack([_rank(sub[column]) for column in controls]))
    if fixed_effect in sub and sub[fixed_effect].nunique() > 1:
        dummies = pd.get_dummies(
            sub[fixed_effect].astype(str),
            drop_first=True,
            dtype=float,
        )
        if not dummies.empty:
            design_parts.append(dummies.to_numpy(dtype=float))
    design = np.column_stack(design_parts)
    residuals = ranked - design @ np.linalg.lstsq(design, ranked, rcond=None)[0]
    return residuals, sub.index


def raw_spearman(
    frame: pd.DataFrame,
    x: str,
    y: str,
) -> tuple[float, float, int, int]:
    sub = _finite(frame, [x, y])
    if len(sub) < 3 or sub[x].nunique() < 2 or sub[y].nunique() < 2:
        return np.nan, np.nan, len(sub), int(sub["rat"].nunique())
    result = spearmanr(sub[x], sub[y])
    return float(result.statistic), float(result.pvalue), len(sub), int(sub["rat"].nunique())


def partial_spearman(
    frame: pd.DataFrame,
    x: str,
    y: str,
    controls: Sequence[str],
    *,
    fixed_effect: str = "session",
) -> tuple[float, float, int, int]:
    columns = [x, y, *controls]
    sub = _finite(frame, columns)
    if len(sub) < max(12, len(controls) + 6):
        return np.nan, np.nan, len(sub), int(sub["rat"].nunique())
    x_resid, x_index = _rank_residuals(
        sub,
        x,
        controls,
        fixed_effect=fixed_effect,
    )
    y_resid, y_index = _rank_residuals(
        sub,
        y,
        controls,
        fixed_effect=fixed_effect,
    )
    if not x_index.equals(y_index) or np.std(x_resid) <= 0 or np.std(y_resid) <= 0:
        return np.nan, np.nan, len(sub), int(sub["rat"].nunique())
    result = pearsonr(x_resid, y_resid)
    return float(result.statistic), float(result.pvalue), len(sub), int(sub["rat"].nunique())


def join_native_ripple_metrics(
    event_medians: pd.DataFrame,
    split_scores: pd.DataFrame,
    *,
    dataset_root: Path,
    session_loader: Callable[[str | Path], object] = load_replay_session,
) -> pd.DataFrame:
    """Join native ripple power and verify exact event-time identity."""

    required_events = {
        "session",
        "rat",
        "event_index",
        PRIMARY_Y,
        MAP_SPECIFIC_Y,
        CONTENT_Y,
        ABSOLUTE_CONTENT_Y,
        *QUALITY_CONTROLS,
        "log1p_median_train_spikes",
    }
    required_splits = {
        "session",
        "event_index",
        "status",
        "event_start_s",
        "event_end_s",
    }
    missing_events = sorted(required_events.difference(event_medians.columns))
    missing_splits = sorted(required_splits.difference(split_scores.columns))
    if missing_events:
        raise ValueError(f"event medians are missing required columns: {missing_events}")
    if missing_splits:
        raise ValueError(f"split scores are missing required columns: {missing_splits}")

    keys = ["session", "event_index"]
    successful = split_scores[split_scores["status"].astype(str).eq("success")].copy()
    timing = (
        successful.groupby(keys, sort=True)
        .agg(
            split_event_start_s=("event_start_s", "median"),
            split_event_end_s=("event_end_s", "median"),
            split_event_start_range_s=("event_start_s", lambda value: float(value.max() - value.min())),
            split_event_end_range_s=("event_end_s", lambda value: float(value.max() - value.min())),
        )
        .reset_index()
    )
    out = event_medians.merge(timing, on=keys, how="left", validate="one_to_one")
    native_rows: list[dict[str, object]] = []
    for session_id, group in out.groupby("session", sort=True):
        session_path = dataset_root.joinpath(*str(session_id).split("/"))
        session = session_loader(session_path)
        for row in group.itertuples(index=False):
            event_index = int(row.event_index)
            if event_index < 0 or event_index >= int(session.ripple_count):
                raise IndexError(
                    f"{session_id} event {event_index} is outside native ripple range "
                    f"0..{int(session.ripple_count) - 1}"
                )
            ripple = session.ripple(event_index)
            native_rows.append(
                {
                    "session": str(session_id),
                    "event_index": event_index,
                    "native_ripple_start_s": float(ripple.start),
                    "native_ripple_end_s": float(ripple.end),
                    "native_ripple_peak_s": float(ripple.peak),
                    "ripple_power_raw": float(ripple.raw_power),
                    "ripple_power_z_session": float(ripple.z_power_session),
                    "ripple_power_z_epoch": float(ripple.z_power_epoch),
                }
            )
    out = out.merge(pd.DataFrame(native_rows), on=keys, how="left", validate="one_to_one")
    out["ripple_event_start_abs_error_s"] = (
        out["split_event_start_s"] - out["native_ripple_start_s"]
    ).abs()
    out["ripple_event_end_abs_error_s"] = (
        out["split_event_end_s"] - out["native_ripple_end_s"]
    ).abs()
    out["log1p_ripple_power_raw"] = np.log1p(out["ripple_power_raw"].clip(lower=0.0))
    out["native_detected_swr"] = True
    out["event_definition"] = "native_detected_ripple_event"
    return out.sort_values(keys).reset_index(drop=True)


def _resample_rat_clusters(
    frame: pd.DataFrame,
    sampled_rats: Sequence[str],
) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for sample_index, rat in enumerate(sampled_rats):
        piece = frame[frame["rat"].astype(str).eq(str(rat))].copy()
        piece["rat"] = f"bootstrap_rat_{sample_index}"
        piece["session"] = (
            f"bootstrap_rat_{sample_index}/" + piece["session"].astype(str)
        )
        pieces.append(piece)
    return pd.concat(pieces, ignore_index=True)


def rat_cluster_bootstrap(
    frame: pd.DataFrame,
    *,
    analysis_id: str,
    x: str,
    y: str,
    controls: Sequence[str],
    replicates: int,
    seed: int,
) -> tuple[pd.DataFrame, tuple[float, float, float]]:
    rats = sorted(frame["rat"].astype(str).unique())
    rng = np.random.default_rng(int(seed))
    rows: list[dict[str, object]] = []
    for replicate in range(int(replicates)):
        sampled = rng.choice(rats, size=len(rats), replace=True)
        estimate, _, _, _ = partial_spearman(
            _resample_rat_clusters(frame, sampled),
            x,
            y,
            controls,
        )
        rows.append(
            {
                "analysis_id": analysis_id,
                "replicate": replicate,
                "partial_spearman_rho": estimate,
            }
        )
    table = pd.DataFrame(rows)
    values = pd.to_numeric(table["partial_spearman_rho"], errors="coerce").dropna()
    if values.empty:
        return table, (np.nan, np.nan, np.nan)
    return table, (
        float(values.quantile(0.025)),
        float(values.quantile(0.975)),
        float((values > 0.0).mean()),
    )


def within_session_permutation(
    frame: pd.DataFrame,
    *,
    analysis_id: str,
    x: str,
    y: str,
    controls: Sequence[str],
    replicates: int,
    seed: int,
) -> tuple[pd.DataFrame, float]:
    observed, _, _, _ = partial_spearman(frame, x, y, controls)
    rng = np.random.default_rng(int(seed))
    rows: list[dict[str, object]] = []
    for replicate in range(int(replicates)):
        shuffled = frame.copy()
        for _, indices in shuffled.groupby("session", sort=True).groups.items():
            index = np.asarray(list(indices))
            shuffled.loc[index, x] = rng.permutation(shuffled.loc[index, x].to_numpy())
        estimate, _, _, _ = partial_spearman(shuffled, x, y, controls)
        rows.append(
            {
                "analysis_id": analysis_id,
                "replicate": replicate,
                "partial_spearman_rho": estimate,
            }
        )
    table = pd.DataFrame(rows)
    values = pd.to_numeric(table["partial_spearman_rho"], errors="coerce").dropna()
    p_value = (
        float((1 + int((values >= observed).sum())) / (1 + len(values)))
        if len(values)
        else np.nan
    )
    return table, p_value


def association_analysis(
    frame: pd.DataFrame,
    *,
    analysis_id: str,
    x: str,
    y: str,
    controls: Sequence[str],
    extended_controls: Sequence[str] | None,
    bootstrap_replicates: int,
    permutation_replicates: int,
    seed: int,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    raw_rho, raw_p, events, rats = raw_spearman(frame, x, y)
    adjusted, adjusted_p, _, _ = partial_spearman(frame, x, y, controls)
    extended, extended_p, _, _ = partial_spearman(
        frame,
        x,
        y,
        controls if extended_controls is None else extended_controls,
    )
    bootstrap, (ci_low, ci_high, positive_fraction) = rat_cluster_bootstrap(
        frame,
        analysis_id=analysis_id,
        x=x,
        y=y,
        controls=controls,
        replicates=bootstrap_replicates,
        seed=seed,
    )
    permutation, permutation_p = within_session_permutation(
        frame,
        analysis_id=analysis_id,
        x=x,
        y=y,
        controls=controls,
        replicates=permutation_replicates,
        seed=seed + 1009,
    )
    return (
        {
            "analysis_id": analysis_id,
            "x_metric": x,
            "y_metric": y,
            "controls": ";".join(controls),
            "fixed_effect": "session",
            "events": int(events),
            "rats": int(rats),
            "raw_spearman_rho": raw_rho,
            "raw_p_value_descriptive": raw_p,
            "adjusted_partial_spearman_rho": adjusted,
            "adjusted_p_value_descriptive": adjusted_p,
            "extended_adjusted_partial_spearman_rho": extended,
            "extended_adjusted_p_value_descriptive": extended_p,
            "rat_cluster_bootstrap_ci_low": ci_low,
            "rat_cluster_bootstrap_ci_high": ci_high,
            "rat_cluster_bootstrap_positive_fraction": positive_fraction,
            "finite_bootstrap_replicates": int(
                pd.to_numeric(bootstrap["partial_spearman_rho"], errors="coerce").notna().sum()
            ),
            "within_session_permutation_p_one_sided": permutation_p,
        },
        bootstrap,
        permutation,
    )


def paired_bootstrap_dissociation(
    frame: pd.DataFrame,
    *,
    replicates: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare conditional coordination and content associations on paired rat draws."""

    rats = sorted(frame["rat"].astype(str).unique())
    rng = np.random.default_rng(int(seed))
    rows: list[dict[str, object]] = []
    for replicate in range(int(replicates)):
        sampled = rng.choice(rats, size=len(rats), replace=True)
        sample = _resample_rat_clusters(frame, sampled)
        coordination, _, _, _ = partial_spearman(
            sample,
            PRIMARY_X,
            PRIMARY_Y,
            PRIMARY_CONTROLS,
        )
        content, _, _, _ = partial_spearman(
            sample,
            PRIMARY_X,
            CONTENT_Y,
            QUALITY_CONTROLS,
        )
        rows.append(
            {
                "replicate": replicate,
                "coordination_partial_rho": coordination,
                "content_partial_rho": content,
                "coordination_minus_content_partial_rho": coordination - content,
            }
        )
    table = pd.DataFrame(rows)
    values = pd.to_numeric(
        table["coordination_minus_content_partial_rho"], errors="coerce"
    ).dropna()
    summary = pd.DataFrame(
        [
            {
                "contrast": "coordination_minus_content_partial_rho",
                "estimate": float(
                    partial_spearman(
                        frame,
                        PRIMARY_X,
                        PRIMARY_Y,
                        PRIMARY_CONTROLS,
                    )[0]
                    - partial_spearman(
                        frame,
                        PRIMARY_X,
                        CONTENT_Y,
                        QUALITY_CONTROLS,
                    )[0]
                ),
                "rat_cluster_bootstrap_ci_low": (
                    float(values.quantile(0.025)) if len(values) else np.nan
                ),
                "rat_cluster_bootstrap_ci_high": (
                    float(values.quantile(0.975)) if len(values) else np.nan
                ),
                "finite_bootstrap_replicates": int(len(values)),
                "predeclared_role": "specificity_refinement_not_primary_support_gate",
            }
        ]
    )
    table["analysis_id"] = "paired_coordination_minus_content"
    return table, summary


def by_rat_and_leave_one_out(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_rat_rows: list[dict[str, object]] = []
    for rat, group in frame.groupby("rat", sort=True):
        estimate, p_value, events, _ = partial_spearman(
            group,
            PRIMARY_X,
            PRIMARY_Y,
            PRIMARY_CONTROLS,
        )
        by_rat_rows.append(
            {
                "rat": rat,
                "events": events,
                "sessions": int(group["session"].nunique()),
                "adjusted_partial_spearman_rho": estimate,
                "p_value_descriptive": p_value,
            }
        )
    loo_rows: list[dict[str, object]] = []
    for rat in sorted(frame["rat"].astype(str).unique()):
        selected = frame[~frame["rat"].astype(str).eq(rat)].copy()
        estimate, p_value, events, rats = partial_spearman(
            selected,
            PRIMARY_X,
            PRIMARY_Y,
            PRIMARY_CONTROLS,
        )
        loo_rows.append(
            {
                "omitted_rat": rat,
                "events": events,
                "rats": rats,
                "adjusted_partial_spearman_rho": estimate,
                "p_value_descriptive": p_value,
            }
        )
    return pd.DataFrame(by_rat_rows), pd.DataFrame(loo_rows)


def build_gate_summary(
    events: pd.DataFrame,
    associations: pd.DataFrame,
    by_rat: pd.DataFrame,
    leave_one_out: pd.DataFrame,
    dissociation: pd.DataFrame,
    *,
    expected_events: int,
    expected_splits: int,
    event_time_tolerance_s: float,
) -> tuple[pd.DataFrame, str]:
    rows: list[dict[str, object]] = []

    def add(
        gate_type: str,
        gate: str,
        passed: bool,
        observed: object,
        criterion: str,
        *,
        required_for_overall: bool,
    ) -> None:
        rows.append(
            {
                "gate_type": gate_type,
                "gate": gate,
                "passed": bool(passed),
                "observed": observed,
                "criterion": criterion,
                "required_for_overall": bool(required_for_overall),
            }
        )

    finite_ripple = np.isfinite(
        events[["ripple_power_raw", "ripple_power_z_session", PRIMARY_X]].to_numpy(
            dtype=float
        )
    ).all(axis=1)
    max_time_error = float(
        events[
            ["ripple_event_start_abs_error_s", "ripple_event_end_abs_error_s"]
        ].max(axis=1).max()
    )
    technical = [
        ("all_events_present", len(events) == expected_events, f"{len(events)}/{expected_events}", "exactly expected events"),
        ("all_four_rats_present", events["rat"].nunique() == 4, int(events["rat"].nunique()), "4 rats"),
        ("all_eight_sessions_present", events["session"].nunique() == 8, int(events["session"].nunique()), "8 sessions"),
        ("all_repeated_splits_complete", bool(events["completed_splits"].eq(expected_splits).all()), f"min={events['completed_splits'].min()}; max={events['completed_splits'].max()}", f"{expected_splits} splits per event"),
        ("native_ripple_power_complete", bool(finite_ripple.all()), int(finite_ripple.sum()), "all native power fields finite"),
        ("native_event_identity_exact", max_time_error <= event_time_tolerance_s, max_time_error, f"max start/end error <= {event_time_tolerance_s:g} s"),
        ("native_detection_threshold_met", bool(events[PRIMARY_X].ge(3.0).all()), float(events[PRIMARY_X].min()), "within-epoch ripple z >= 3 for every detected event"),
        ("event_keys_unique", not events.duplicated(["session", "event_index"]).any(), int(events.duplicated(["session", "event_index"]).sum()), "zero duplicate event keys"),
        ("primary_values_finite", bool(np.isfinite(events[[PRIMARY_X, PRIMARY_Y, CONTENT_Y, *PRIMARY_CONTROLS]].to_numpy(dtype=float)).all()), len(events), "all primary values and controls finite"),
    ]
    for gate, passed, observed, criterion in technical:
        add("technical", gate, passed, observed, criterion, required_for_overall=True)
    technical_pass = all(bool(item[1]) for item in technical)
    add(
        "summary",
        "overall_technical",
        technical_pass,
        "pass" if technical_pass else "fail",
        "all technical gates pass",
        required_for_overall=True,
    )

    indexed = associations.set_index("analysis_id")
    primary = indexed.loc["primary_ripple_strength_predicts_coordination_beyond_content"]
    map_specific = indexed.loc["sensitivity_ripple_strength_predicts_map_specific_coordination"]
    content = indexed.loc["secondary_ripple_strength_predicts_map_specific_content"]
    primary_conditions = [
        ("primary_adjusted_positive", float(primary.adjusted_partial_spearman_rho) > 0.0, float(primary.adjusted_partial_spearman_rho), "> 0"),
        ("primary_rat_bootstrap_ci_above_zero", float(primary.rat_cluster_bootstrap_ci_low) > 0.0, f"[{primary.rat_cluster_bootstrap_ci_low:.6g}, {primary.rat_cluster_bootstrap_ci_high:.6g}]", "95% rat-cluster CI lower bound > 0"),
        ("primary_within_session_permutation_significant", float(primary.within_session_permutation_p_one_sided) <= 0.05, float(primary.within_session_permutation_p_one_sided), "one-sided p <= 0.05"),
        ("primary_all_rats_positive", bool((by_rat["adjusted_partial_spearman_rho"] > 0.0).all()), int((by_rat["adjusted_partial_spearman_rho"] > 0.0).sum()), "4/4 rats positive"),
        ("primary_all_leave_one_rat_out_positive", bool((leave_one_out["adjusted_partial_spearman_rho"] > 0.0).all()), int((leave_one_out["adjusted_partial_spearman_rho"] > 0.0).sum()), "4/4 leave-one-rat-out estimates positive"),
        ("primary_extended_controls_positive", float(primary.extended_adjusted_partial_spearman_rho) > 0.0, float(primary.extended_adjusted_partial_spearman_rho), "> 0 after extended controls"),
    ]
    for gate, passed, observed, criterion in primary_conditions:
        add("scientific_primary", gate, passed, observed, criterion, required_for_overall=True)
    coordination_supported = bool(technical_pass and all(item[1] for item in primary_conditions))
    add(
        "summary",
        "ripple_coordination_hypothesis_supported",
        coordination_supported,
        f"{sum(bool(item[1]) for item in primary_conditions)}/{len(primary_conditions)} scientific gates",
        "technical pass and every primary scientific gate passes",
        required_for_overall=True,
    )

    map_conditions = [
        float(map_specific.adjusted_partial_spearman_rho) > 0.0,
        float(map_specific.rat_cluster_bootstrap_ci_low) > 0.0,
        float(map_specific.within_session_permutation_p_one_sided) <= 0.05,
    ]
    map_supported = bool(coordination_supported and all(map_conditions))
    add(
        "scientific_refinement",
        "map_specific_coordination_sensitivity_supported",
        map_supported,
        f"rho={map_specific.adjusted_partial_spearman_rho:.6g}; CI=[{map_specific.rat_cluster_bootstrap_ci_low:.6g}, {map_specific.rat_cluster_bootstrap_ci_high:.6g}]; p={map_specific.within_session_permutation_p_one_sided:.6g}",
        "primary supported; map-specific outcome rho > 0, CI lower > 0, and permutation p <= 0.05",
        required_for_overall=False,
    )
    content_supported = bool(
        float(content.adjusted_partial_spearman_rho) > 0.0
        and float(content.rat_cluster_bootstrap_ci_low) > 0.0
        and float(content.within_session_permutation_p_one_sided) <= 0.05
    )
    add(
        "scientific_refinement",
        "map_specific_content_association_supported",
        content_supported,
        f"rho={content.adjusted_partial_spearman_rho:.6g}; CI=[{content.rat_cluster_bootstrap_ci_low:.6g}, {content.rat_cluster_bootstrap_ci_high:.6g}]; p={content.within_session_permutation_p_one_sided:.6g}",
        "content rho > 0, CI lower > 0, and permutation p <= 0.05",
        required_for_overall=False,
    )
    difference_low = float(dissociation["rat_cluster_bootstrap_ci_low"].iloc[0])
    add(
        "scientific_refinement",
        "coordination_association_stronger_than_content",
        difference_low > 0.0,
        f"difference={dissociation['estimate'].iloc[0]:.6g}; CI=[{difference_low:.6g}, {dissociation['rat_cluster_bootstrap_ci_high'].iloc[0]:.6g}]",
        "paired rat-bootstrap CI for coordination minus content lies above zero",
        required_for_overall=False,
    )

    if not technical_pass:
        decision = "technical_fail"
    elif coordination_supported and map_supported and difference_low > 0.0:
        decision = "ripple_selectively_indexes_map_specific_cross_cell_coordination"
    elif coordination_supported and content_supported:
        decision = "ripple_strength_tracks_both_coordination_and_content"
    elif coordination_supported and map_supported:
        decision = "ripple_indexes_map_specific_cross_cell_coordination"
    elif coordination_supported:
        decision = "ripple_indexes_cross_cell_coordination_spatial_specificity_unresolved"
    elif content_supported:
        decision = "ripple_strength_tracks_content_not_heldout_coordination"
    else:
        decision = "no_robust_ripple_strength_association"
    add(
        "summary",
        "overall_decision",
        decision not in {"technical_fail", "no_robust_ripple_strength_association"},
        decision,
        "classification under frozen primary and refinement gates",
        required_for_overall=False,
    )
    return pd.DataFrame(rows), decision


def _plot_results(
    events: pd.DataFrame,
    by_rat: pd.DataFrame,
    output: Path,
) -> None:
    colors = dict(zip(sorted(events["rat"].unique()), plt.cm.tab10.colors, strict=False))
    figure, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for rat, group in events.groupby("rat", sort=True):
        axes[0, 0].scatter(group[PRIMARY_X], group[PRIMARY_Y], s=24, alpha=0.75, label=rat, color=colors[rat])
        axes[0, 1].scatter(group[PRIMARY_X], group[CONTENT_Y], s=24, alpha=0.75, color=colors[rat])
        axes[1, 0].scatter(group[PRIMARY_X], group[MAP_SPECIFIC_Y], s=24, alpha=0.75, color=colors[rat])
    axes[0, 0].set(xlabel="Within-epoch ripple power (z)", ylabel="Held-out IMM - fragmented score", title="Cross-cell coordination")
    axes[0, 1].set(xlabel="Within-epoch ripple power (z)", ylabel="Real - wrong nonstationary mass", title="Map-specific posterior content")
    axes[1, 0].set(xlabel="Within-epoch ripple power (z)", ylabel="Map-specific held-out IMM - fragmented", title="Map-specific held-out sensitivity")
    axes[0, 0].legend(frameon=False, ncol=2)
    axes[1, 1].axvline(0.0, color="black", linewidth=1)
    axes[1, 1].barh(by_rat["rat"], by_rat["adjusted_partial_spearman_rho"], color=[colors[rat] for rat in by_rat["rat"]])
    axes[1, 1].set(xlabel="Adjusted partial Spearman rho", ylabel="Rat", title="Primary association by rat")
    for axis in axes.flat:
        axis.spines[["top", "right"]].set_visible(False)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _write_report(
    path: Path,
    *,
    decision: str,
    associations: pd.DataFrame,
    by_rat: pd.DataFrame,
    dissociation: pd.DataFrame,
    events: pd.DataFrame,
) -> None:
    indexed = associations.set_index("analysis_id")
    primary = indexed.loc["primary_ripple_strength_predicts_coordination_beyond_content"]
    map_specific = indexed.loc["sensitivity_ripple_strength_predicts_map_specific_coordination"]
    content = indexed.loc["secondary_ripple_strength_predicts_map_specific_content"]
    path.write_text(
        "\n".join(
            [
                "# Pfeiffer/Foster ripple strength and cross-cell coordination",
                "",
                f"**Decision:** `{decision}`",
                "",
                "## Frozen question",
                "",
                "Does native within-epoch ripple power predict leakage-free held-out-cell IMM-versus-fragmented performance after controlling map-specific nonstationary mode allocation and event/decoder quality? This is a non-rescoring analysis of the frozen 160-event repeated-cell-split artifact.",
                "",
                "## Primary result",
                "",
                f"- Events: {len(events)} across {events['session'].nunique()} sessions and {events['rat'].nunique()} rats",
                f"- Raw Spearman rho: {primary.raw_spearman_rho:.3f}",
                f"- Adjusted partial rho: {primary.adjusted_partial_spearman_rho:.3f}",
                f"- Rat-bootstrap 95% CI: [{primary.rat_cluster_bootstrap_ci_low:.3f}, {primary.rat_cluster_bootstrap_ci_high:.3f}]",
                f"- Within-session permutation p: {primary.within_session_permutation_p_one_sided:.4g}",
                f"- Per-rat positive directions: {int((by_rat['adjusted_partial_spearman_rho'] > 0).sum())}/{len(by_rat)}",
                "",
                "## Content and map-specific refinements",
                "",
                f"- Map-specific held-out outcome: adjusted rho {map_specific.adjusted_partial_spearman_rho:.3f}, CI [{map_specific.rat_cluster_bootstrap_ci_low:.3f}, {map_specific.rat_cluster_bootstrap_ci_high:.3f}], p={map_specific.within_session_permutation_p_one_sided:.4g}",
                f"- Map-specific nonstationary content: adjusted rho {content.adjusted_partial_spearman_rho:.3f}, CI [{content.rat_cluster_bootstrap_ci_low:.3f}, {content.rat_cluster_bootstrap_ci_high:.3f}], p={content.within_session_permutation_p_one_sided:.4g}",
                f"- Coordination-minus-content paired bootstrap contrast: {dissociation['estimate'].iloc[0]:.3f}, CI [{dissociation['rat_cluster_bootstrap_ci_low'].iloc[0]:.3f}, {dissociation['rat_cluster_bootstrap_ci_high'].iloc[0]:.3f}]",
                "",
                "## Claim boundary",
                "",
                "The processed PF release contains native power values only for detected ripples and no continuous LFP. This analysis can test whether strength varies with coordination within detected events; it cannot establish that promoted off-SWR windows are physiologically ripple-negative or infer a causal broadcasting role. All detected events are threshold-truncated at within-epoch z >= 3, which can attenuate associations.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def run_analysis(args: argparse.Namespace) -> dict[str, Path]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    event_medians = pd.read_csv(args.event_medians)
    split_scores = pd.read_csv(args.split_scores)
    events = join_native_ripple_metrics(
        event_medians,
        split_scores,
        dataset_root=Path(args.dataset_root),
    )

    specs = (
        (
            "primary_ripple_strength_predicts_coordination_beyond_content",
            PRIMARY_Y,
            PRIMARY_CONTROLS,
            EXTENDED_PRIMARY_CONTROLS,
        ),
        (
            "sensitivity_ripple_strength_predicts_map_specific_coordination",
            MAP_SPECIFIC_Y,
            PRIMARY_CONTROLS,
            EXTENDED_PRIMARY_CONTROLS,
        ),
        (
            "secondary_ripple_strength_predicts_map_specific_content",
            CONTENT_Y,
            QUALITY_CONTROLS,
            (*QUALITY_CONTROLS, "log1p_median_train_spikes"),
        ),
        (
            "secondary_ripple_strength_predicts_absolute_nonstationary_content",
            ABSOLUTE_CONTENT_Y,
            QUALITY_CONTROLS,
            (*QUALITY_CONTROLS, "log1p_median_train_spikes"),
        ),
    )
    association_rows: list[dict[str, object]] = []
    bootstrap_tables: list[pd.DataFrame] = []
    permutation_tables: list[pd.DataFrame] = []
    for index, (analysis_id, outcome, controls, extended) in enumerate(specs):
        row, bootstrap, permutation = association_analysis(
            events,
            analysis_id=analysis_id,
            x=PRIMARY_X,
            y=outcome,
            controls=controls,
            extended_controls=extended,
            bootstrap_replicates=args.bootstrap_replicates,
            permutation_replicates=args.permutation_replicates,
            seed=args.seed + index * 10000,
        )
        association_rows.append(row)
        bootstrap_tables.append(bootstrap)
        permutation_tables.append(permutation)

    for power_column in ("ripple_power_z_session", "log1p_ripple_power_raw"):
        row, _, _, _ = partial_spearman(
            events,
            power_column,
            PRIMARY_Y,
            PRIMARY_CONTROLS,
        )
        raw, raw_p, n_events, n_rats = raw_spearman(events, power_column, PRIMARY_Y)
        association_rows.append(
            {
                "analysis_id": f"power_scale_sensitivity_{power_column}",
                "x_metric": power_column,
                "y_metric": PRIMARY_Y,
                "controls": ";".join(PRIMARY_CONTROLS),
                "fixed_effect": "session",
                "events": n_events,
                "rats": n_rats,
                "raw_spearman_rho": raw,
                "raw_p_value_descriptive": raw_p,
                "adjusted_partial_spearman_rho": row,
                "adjusted_p_value_descriptive": np.nan,
                "extended_adjusted_partial_spearman_rho": np.nan,
                "extended_adjusted_p_value_descriptive": np.nan,
                "rat_cluster_bootstrap_ci_low": np.nan,
                "rat_cluster_bootstrap_ci_high": np.nan,
                "rat_cluster_bootstrap_positive_fraction": np.nan,
                "finite_bootstrap_replicates": 0,
                "within_session_permutation_p_one_sided": np.nan,
            }
        )

    associations = pd.DataFrame(association_rows)
    bootstrap = pd.concat(bootstrap_tables, ignore_index=True)
    permutation = pd.concat(permutation_tables, ignore_index=True)
    paired_bootstrap, dissociation = paired_bootstrap_dissociation(
        events,
        replicates=args.bootstrap_replicates,
        seed=args.seed + 50000,
    )
    bootstrap = pd.concat([bootstrap, paired_bootstrap], ignore_index=True, sort=False)
    by_rat, leave_one_out = by_rat_and_leave_one_out(events)
    gates, decision = build_gate_summary(
        events,
        associations,
        by_rat,
        leave_one_out,
        dissociation,
        expected_events=args.expected_events,
        expected_splits=args.expected_splits,
        event_time_tolerance_s=args.event_time_tolerance_s,
    )

    outputs = {
        EVENT_OUTPUT: output_dir / EVENT_OUTPUT,
        ASSOCIATION_OUTPUT: output_dir / ASSOCIATION_OUTPUT,
        BY_RAT_OUTPUT: output_dir / BY_RAT_OUTPUT,
        LOO_OUTPUT: output_dir / LOO_OUTPUT,
        BOOTSTRAP_OUTPUT: output_dir / BOOTSTRAP_OUTPUT,
        PERMUTATION_OUTPUT: output_dir / PERMUTATION_OUTPUT,
        DISSOCIATION_OUTPUT: output_dir / DISSOCIATION_OUTPUT,
        GATE_OUTPUT: output_dir / GATE_OUTPUT,
        FIGURE_OUTPUT: output_dir / FIGURE_OUTPUT,
        MANIFEST_OUTPUT: output_dir / MANIFEST_OUTPUT,
        REPORT_OUTPUT: output_dir / REPORT_OUTPUT,
    }
    events.to_csv(outputs[EVENT_OUTPUT], index=False)
    associations.to_csv(outputs[ASSOCIATION_OUTPUT], index=False)
    by_rat.to_csv(outputs[BY_RAT_OUTPUT], index=False)
    leave_one_out.to_csv(outputs[LOO_OUTPUT], index=False)
    bootstrap.to_csv(outputs[BOOTSTRAP_OUTPUT], index=False)
    permutation.to_csv(outputs[PERMUTATION_OUTPUT], index=False)
    dissociation.to_csv(outputs[DISSOCIATION_OUTPUT], index=False)
    gates.to_csv(outputs[GATE_OUTPUT], index=False)
    _plot_results(events, by_rat, outputs[FIGURE_OUTPUT])
    _write_report(
        outputs[REPORT_OUTPUT],
        decision=decision,
        associations=associations,
        by_rat=by_rat,
        dissociation=dissociation,
        events=events,
    )
    manifest = {
        "analysis": "pfeiffer_ripple_cross_cell_coordination_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "non_rescoring": True,
        "event_definition": "native detected ripple events only",
        "primary_predictor": PRIMARY_X,
        "primary_outcome": PRIMARY_Y,
        "primary_controls": list(PRIMARY_CONTROLS),
        "fixed_effect": "session",
        "selection_boundary": "all frozen 160 events; no outcome-based selection",
        "continuous_lfp_available": False,
        "off_swr_physiological_negativity_testable": False,
        "bootstrap_replicates": int(args.bootstrap_replicates),
        "permutation_replicates": int(args.permutation_replicates),
        "seed": int(args.seed),
        "provenance": build_script_provenance(
            input_paths={
                "dataset_root": args.dataset_root,
                "event_medians": args.event_medians,
                "split_scores": args.split_scores,
            }
        ),
        "outputs": {key: str(value) for key, value in outputs.items()},
    }
    outputs[MANIFEST_OUTPUT].write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--event-medians", type=Path, required=True)
    parser.add_argument("--split-scores", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-events", type=int, default=160)
    parser.add_argument("--expected-splits", type=int, default=20)
    parser.add_argument("--event-time-tolerance-s", type=float, default=1e-9)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--permutation-replicates", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260805)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_analysis(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
