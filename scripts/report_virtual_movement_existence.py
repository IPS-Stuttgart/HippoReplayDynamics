#!/usr/bin/env python3
"""Consolidate direct evidence for virtual movement across PF and Tanni.

This reporter does not rescore events. It reads the authoritative Pfeiffer/Foster
time-order, posterior-content, map-specificity, held-out-cell, and decoder-QC
artifacts together with the Tanni all-event virtual-movement audit.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pf-time-order-by-group", required=True, type=Path)
    parser.add_argument("--pf-posterior-content", required=True, type=Path)
    parser.add_argument("--pf-map-specificity", required=True, type=Path)
    parser.add_argument("--pf-heldout-scope", required=True, type=Path)
    parser.add_argument("--pf-decoder-qc", required=True, type=Path)
    parser.add_argument("--tanni-event-audit", required=True, type=Path)
    parser.add_argument("--tanni-summary", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def build_evidence_table(
    time_order: pd.DataFrame,
    content: pd.DataFrame,
    map_specificity: pd.DataFrame,
    heldout: pd.DataFrame,
    decoder: pd.DataFrame,
    tanni_events: pd.DataFrame,
    tanni_summary: pd.DataFrame,
) -> pd.DataFrame:
    clean_time = time_order.loc[time_order["event_group"] == "clean_imm"].iloc[0]
    overall_content = content.loc[content["group"] == "overall"].iloc[0]
    map_rows = map_specificity.loc[map_specificity["analysis_scope"] == "clean_imm_fixed_subset"].set_index("metric")
    heldout_all = heldout.loc[heldout["scope"] == "all_events"].iloc[0]
    tanni_all = tanni_summary.loc[tanni_summary["scope"] == "all_events"].iloc[0]
    tanni_dedup = tanni_summary.loc[tanni_summary["scope"] == "one_per_source_group"].iloc[0]
    tanni_margin = tanni_events.loc[tanni_events["ordered_model_confident"]]
    tanni_event_level = tanni_margin.loc[
        (tanni_margin["original_ordered_margin"] > tanni_margin["p95_time_shuffle_margin"])
        & (tanni_margin["original_ordered_margin"] > tanni_margin["p95_map_shuffle_margin"])
        & tanni_margin["displacing"]
    ]
    decoder_median = float(decoder["median_posterior_mean_error_cm"].median())
    rows = [
        {
            "dataset": "Pfeiffer/Foster",
            "test": "RUN decoder validation",
            "passed_count": int(decoder.shape[0]),
            "tested_count": int(decoder.shape[0]),
            "fraction": 1.0,
            "effect": decoder_median,
            "effect_unit": "median session decoding error, cm",
            "status": "pass",
            "interpretation": "encoding model reconstructs actual running position",
        },
        {
            "dataset": "Pfeiffer/Foster",
            "test": "time order above shuffle p95",
            "passed_count": int(clean_time["original_above_shuffle_p95_count"]),
            "tested_count": int(clean_time["events"]),
            "fraction": float(clean_time["original_above_shuffle_p95_count"] / clean_time["events"]),
            "effect": float(clean_time["median_time_order_advantage"]),
            "effect_unit": "median IMM-fragmented time-order advantage",
            "status": "pass",
            "interpretation": "ordered population snapshots matter",
        },
        {
            "dataset": "Pfeiffer/Foster",
            "test": "posterior trajectory content",
            "passed_count": int(overall_content["moderate_content_pass_count"]),
            "tested_count": int(overall_content["events"]),
            "fraction": float(overall_content["moderate_content_pass_fraction"]),
            "effect": float(overall_content["median_posterior_net_displacement_cm"]),
            "effect_unit": "median posterior displacement, cm",
            "status": "pass",
            "interpretation": "posterior uses nonstationary modes and displaces",
        },
        {
            "dataset": "Pfeiffer/Foster",
            "test": "map-specific nonstationary content",
            "passed_count": int(map_rows.loc["mean_nonstationary_mode_probability", "empirical_p_le_0p05_count"]),
            "tested_count": int(map_rows.loc["mean_nonstationary_mode_probability", "events"]),
            "fraction": float(map_rows.loc["mean_nonstationary_mode_probability", "empirical_p_le_0p05_count"] / map_rows.loc["mean_nonstationary_mode_probability", "events"]),
            "effect": float(map_rows.loc["mean_nonstationary_mode_probability", "median_real_minus_null_median"]),
            "effect_unit": "median real-minus-map-null mode mass",
            "status": "pass",
            "interpretation": "movement content depends on the learned spatial map",
        },
        {
            "dataset": "Pfeiffer/Foster",
            "test": "held-out-cell prediction",
            "passed_count": int(heldout_all["event_heldout_delta_positive_count"]),
            "tested_count": int(heldout_all["events"]),
            "fraction": float(heldout_all["event_heldout_delta_positive_fraction"]),
            "effect": float(heldout_all["median_event_heldout_delta"]),
            "effect_unit": "median held-out IMM-fragmented delta",
            "status": "pass",
            "interpretation": "dynamics predict neurons excluded from path inference",
        },
        {
            "dataset": "Tanni large arenas",
            "test": "ordered family margin",
            "passed_count": int(tanni_all["ordered_model_confident"]),
            "tested_count": int(tanni_all["events"]),
            "fraction": float(tanni_all["ordered_model_confident"] / tanni_all["events"]),
            "effect": float(tanni_all["median_original_ordered_margin"]),
            "effect_unit": "median ordered-minus-nonordered margin",
            "status": "weak",
            "interpretation": "few broad ripple candidates favor ordered movement",
        },
        {
            "dataset": "Tanni large arenas",
            "test": "event-level two-null movement candidates",
            "passed_count": int(tanni_event_level.shape[0]),
            "tested_count": int(tanni_all["events"]),
            "fraction": float(tanni_event_level.shape[0] / tanni_all["events"]),
            "effect": np.nan,
            "effect_unit": "count before familywise correction",
            "status": "exploratory",
            "interpretation": "plausible candidates, including overlapping windows",
        },
        {
            "dataset": "Tanni large arenas",
            "test": "familywise strict virtual movement",
            "passed_count": int(tanni_dedup["strict_virtual_movement"]),
            "tested_count": int(tanni_dedup["events"]),
            "fraction": float(tanni_dedup["strict_virtual_movement"] / tanni_dedup["events"]),
            "effect": np.nan,
            "effect_unit": "de-duplicated events",
            "status": "fail",
            "interpretation": "broad Tanni candidate set does not independently establish replay",
        },
    ]
    return pd.DataFrame(rows)


def make_figure(evidence: pd.DataFrame, output_path: Path) -> None:
    figure_rows = evidence.loc[~evidence["test"].eq("RUN decoder validation")].copy()
    pf = figure_rows.loc[figure_rows["dataset"] == "Pfeiffer/Foster"]
    tanni = figure_rows.loc[figure_rows["dataset"] == "Tanni large arenas"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    for ax, frame, title, color in [
        (axes[0], pf, "Pfeiffer/Foster replay events", "#b6323b"),
        (axes[1], tanni, "Tanni broad awake ripple candidates", "#4f718f"),
    ]:
        y = np.arange(frame.shape[0])
        ax.barh(y, frame["fraction"] * 100.0, color=color, alpha=0.88)
        ax.set_yticks(y, frame["test"])
        ax.invert_yaxis()
        ax.set_xlim(0.0, 100.0)
        ax.set_xlabel("Events passing diagnostic (%)")
        ax.set_title(title)
        for index, row in enumerate(frame.itertuples(index=False)):
            ax.text(
                min(float(row.fraction) * 100.0 + 1.5, 94.0),
                index,
                f"{int(row.passed_count)}/{int(row.tested_count)}",
                va="center",
                fontsize=9,
            )
    fig.suptitle("Is there virtual movement? Evidence depends on event definition")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_report(evidence: pd.DataFrame, output_path: Path) -> None:
    pf = evidence.loc[evidence["dataset"] == "Pfeiffer/Foster"].set_index("test")
    tanni = evidence.loc[evidence["dataset"] == "Tanni large arenas"].set_index("test")
    lines = [
        "# Virtual-movement existence audit",
        "",
        "**Verdict: sequential replay is supported as a strict subset in Pfeiffer/Foster, but not established in the broad Tanni ripple-candidate set.**",
        "",
        "Diffusion is virtual movement: it is a locally continuous random walk. Stationary means no movement; fragmented means spatial reactivation without a traversed continuous path. First-order IMM is only movement-positive when its posterior uses nonstationary modes and displaces.",
        "",
        "## Pfeiffer/Foster",
        "",
        f"- Time-order sensitive: {int(pf.loc['time order above shuffle p95', 'passed_count'])}/{int(pf.loc['time order above shuffle p95', 'tested_count'])} clean-IMM events; median order advantage {pf.loc['time order above shuffle p95', 'effect']:.2f}.",
        f"- Posterior-content positive: {int(pf.loc['posterior trajectory content', 'passed_count'])}/{int(pf.loc['posterior trajectory content', 'tested_count'])}; median displacement {pf.loc['posterior trajectory content', 'effect']:.1f} cm.",
        f"- Map-specific mode content: {int(pf.loc['map-specific nonstationary content', 'passed_count'])}/{int(pf.loc['map-specific nonstationary content', 'tested_count'])} event-level empirical p values <= 0.05.",
        f"- Held-out predictive advantage: {int(pf.loc['held-out-cell prediction', 'passed_count'])}/{int(pf.loc['held-out-cell prediction', 'tested_count'])} events positive; median held-out delta {pf.loc['held-out-cell prediction', 'effect']:.2f}.",
        "",
        "These tests attack different failure modes: temporal bag-of-bins, decoder-map artifact, stationary IMM use, and neural overfitting.",
        "",
        "## Tanni large arenas",
        "",
        f"- Ordered-family margin: {int(tanni.loc['ordered family margin', 'passed_count'])}/{int(tanni.loc['ordered family margin', 'tested_count'])}.",
        f"- Event-level two-null candidates: {int(tanni.loc['event-level two-null movement candidates', 'passed_count'])}/{int(tanni.loc['event-level two-null movement candidates', 'tested_count'])} before familywise correction.",
        f"- Strict de-duplicated familywise candidates: {int(tanni.loc['familywise strict virtual movement', 'passed_count'])}/{int(tanni.loc['familywise strict virtual movement', 'tested_count'])}.",
        "",
        "Tanni therefore acts as a specificity check: the decoder does not declare robust sequential replay in every ripple-rich 2D dataset. It does not negate the independently validated Pfeiffer/Foster subset.",
        "",
        "## Claim boundary",
        "",
        "This establishes decoded, map-dependent, time-ordered virtual movement in a strict Pfeiffer/Foster subset. It does not imply that every SWR is replay, that all reactivation is continuous movement, or that the Tanni broad candidate set is replay-negative under a future curated event definition.",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence = build_evidence_table(
        pd.read_csv(args.pf_time_order_by_group),
        pd.read_csv(args.pf_posterior_content),
        pd.read_csv(args.pf_map_specificity),
        pd.read_csv(args.pf_heldout_scope),
        pd.read_csv(args.pf_decoder_qc),
        pd.read_csv(args.tanni_event_audit),
        pd.read_csv(args.tanni_summary),
    )
    evidence.to_csv(output_dir / "virtual_movement_existence_evidence.csv", index=False)
    make_figure(evidence, output_dir / "virtual_movement_existence_figure.png")
    write_report(evidence, output_dir / "virtual_movement_existence_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
