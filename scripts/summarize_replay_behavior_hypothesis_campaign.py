#!/usr/bin/env python3
"""Consolidate the frozen H1-H10 replay-behavior hypothesis campaign.

This reporter never rescales, reselects, or rescores events. Missing or
insufficient primary tests enter the ten-hypothesis Benjamini-Hochberg family
as p=1, so incomplete hypotheses cannot reduce the multiplicity correction.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

RESULT_OUTPUT = "replay_behavior_hypothesis_campaign_results.csv"
COMPANION_OUTPUT = "replay_behavior_hypothesis_companion_tests.csv"
FDR_OUTPUT = "replay_behavior_hypothesis_campaign_fdr.csv"
GATE_OUTPUT = "replay_behavior_hypothesis_campaign_gate_summary.csv"
REPORT_OUTPUT = "replay_behavior_hypothesis_campaign_report.md"
MANIFEST_OUTPUT = "replay_behavior_hypothesis_campaign_manifest.json"
H7_CONTEXT_OUTPUT = "pf_h7_off_swr_route_context.csv"

HYPOTHESES = {
    "H1": "Replay commitment during a pause",
    "H2": "Prospective planning versus retrospective reinstatement",
    "H3": "Novel route construction",
    "H4": "Goal certainty controls dynamics",
    "H5": "Policy branching, not geometric curvature, evokes IMM",
    "H6": "Neural ensemble turnover is the biological switch",
    "H7": "SWRs mark commitment rather than generate trajectories",
    "H8": "Replay has a within-event grammar",
    "H9": "Learning or reward change increases switching",
    "H10": "Event duration reflects computational composition",
}


def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    """Return monotone BH-adjusted q-values for a complete finite family."""

    values = pd.to_numeric(p_values, errors="raise").to_numpy(dtype=float)
    if not np.all(np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("BH input p-values must be finite and lie in [0, 1]")
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    out = np.empty_like(adjusted)
    out[order] = np.minimum(adjusted, 1.0)
    return pd.Series(out, index=p_values.index, dtype=float)


def _first(frame: pd.DataFrame, **filters: str) -> pd.Series:
    selected = frame.copy()
    for column, value in filters.items():
        selected = selected[selected[column].astype(str).eq(str(value))]
    if selected.empty:
        return pd.Series(dtype=object)
    return selected.iloc[0]


def _float(row: pd.Series, column: str, default: float = np.nan) -> float:
    try:
        value = float(row.get(column, default))
    except (TypeError, ValueError):
        return float(default)
    return value if np.isfinite(value) else float(default)


def _int(row: pd.Series, column: str, default: int = 0) -> int:
    try:
        return int(float(row.get(column, default)))
    except (TypeError, ValueError):
        return int(default)


def _context_result(context_tests: pd.DataFrame, hypothesis: str) -> pd.Series:
    return _first(context_tests, hypothesis=hypothesis, role="primary")


def _row(
    hypothesis: str,
    *,
    estimand: str,
    expected_direction: str,
    events: int,
    rats: int,
    sessions: int,
    estimate: float,
    ci_low: float,
    ci_high: float,
    primary_p_value: float,
    null_control: str,
    technical_status: str,
    pre_fdr_interpretation: str,
    robustness_gate: bool,
    evidence_note: str,
) -> dict[str, object]:
    return {
        "hypothesis": hypothesis,
        "hypothesis_name": HYPOTHESES[hypothesis],
        "estimand": estimand,
        "expected_direction": expected_direction,
        "events": int(events),
        "rats": int(rats),
        "sessions": int(sessions),
        "estimate": float(estimate) if np.isfinite(estimate) else np.nan,
        "ci_low": float(ci_low) if np.isfinite(ci_low) else np.nan,
        "ci_high": float(ci_high) if np.isfinite(ci_high) else np.nan,
        "primary_p_value": float(primary_p_value) if np.isfinite(primary_p_value) else np.nan,
        "null_control": null_control,
        "technical_status": technical_status,
        "pre_fdr_interpretation": pre_fdr_interpretation,
        "robustness_gate": bool(robustness_gate),
        "evidence_note": evidence_note,
    }


def build_campaign_results(
    *,
    context_tests: pd.DataFrame,
    h5_tests: pd.DataFrame,
    h6_transitions: pd.DataFrame,
    h6_splits: pd.DataFrame,
    h7_summary: pd.DataFrame,
    h7_context: pd.DataFrame,
    h8_summary: pd.DataFrame,
    h8_gates: pd.DataFrame,
    h9_inference: pd.DataFrame,
) -> pd.DataFrame:
    """Build one frozen primary row per hypothesis."""

    rows: list[dict[str, object]] = []
    directions = {
        "H1": "positive",
        "H2": "negative",
        "H3": "negative",
        "H4": "two-sided",
        "H10": "negative",
    }
    for hypothesis in ("H1", "H2", "H3", "H4"):
        source = _context_result(context_tests, hypothesis)
        source_status = str(source.get("status_before_campaign_fdr", "missing"))
        robust = source_status == "directional_pass_unadjusted"
        if hypothesis == "H1":
            companion = _first(
                context_tests,
                hypothesis="H1",
                role="companion_required",
            )
            robust = robust and str(
                companion.get("status_before_campaign_fdr", "missing")
            ) == "directional_pass_unadjusted"
        rows.append(
            _row(
                hypothesis,
                estimand=str(source.get("test", "")),
                expected_direction=directions[hypothesis],
                events=_int(source, "events"),
                rats=_int(source, "rats"),
                sessions=_int(source, "sessions"),
                estimate=_float(source, "estimate"),
                ci_low=_float(source, "rat_bootstrap_ci_low"),
                ci_high=_float(source, "rat_bootstrap_ci_high"),
                primary_p_value=_float(source, "permutation_p_value"),
                null_control=str(source.get("null_control", "")),
                technical_status="complete" if not source.empty else "missing",
                pre_fdr_interpretation=source_status,
                robustness_gate=robust,
                evidence_note=str(source.get("outcome", "")),
            )
        )

    h5 = _first(h5_tests, hypothesis="H5", role="primary")
    h5_robust = _int(h5, "positive_rats") == 4 and bool(h5.get("leave_one_rat_out_positive", False))
    rows.append(
        _row(
            "H5",
            estimand=str(h5.get("test", "")),
            expected_direction="positive",
            events=_int(h5, "events"),
            rats=_int(h5, "rats"),
            sessions=_int(h5, "sessions"),
            estimate=_float(h5, "estimate"),
            ci_low=_float(h5, "rat_bootstrap_ci_low"),
            ci_high=_float(h5, "rat_bootstrap_ci_high"),
            primary_p_value=_float(h5, "permutation_p_value"),
            null_control=str(h5.get("null_control", "")),
            technical_status="complete" if not h5.empty else "missing",
            pre_fdr_interpretation=(
                "selective_positive_not_rat_uniform" if not h5_robust else str(h5.get("status_before_campaign_fdr", ""))
            ),
            robustness_gate=h5_robust,
            evidence_note=f"positive rats={_int(h5, 'positive_rats')}/4",
        )
    )

    conditional = pd.to_numeric(
        h6_transitions.get(
            "stationary_continuous_switch_probability_given_nonfragmented",
            pd.Series(dtype=float),
        ),
        errors="coerce",
    ).dropna()
    all_cell_candidates = int((conditional >= 0.5).sum())
    split_candidates = int(
        pd.to_numeric(
            h6_splits.get("assembly_boundary_candidate_count", pd.Series(dtype=float)),
            errors="coerce",
        ).fillna(0).gt(0).sum()
    )
    h6_events = int(h6_splits[["session", "event_index"]].drop_duplicates().shape[0]) if {"session", "event_index"}.issubset(h6_splits) else 0
    rows.append(
        _row(
            "H6",
            estimand="heldout assembly turnover at training-defined stationary-continuous boundary",
            expected_direction="positive",
            events=h6_events,
            rats=int(h6_splits.get("rat", pd.Series(dtype=str)).nunique()),
            sessions=int(h6_splits.get("session", pd.Series(dtype=str)).nunique()),
            estimate=np.nan,
            ci_low=np.nan,
            ci_high=np.nan,
            primary_p_value=np.nan,
            null_control="matched same-event low-switch transitions with held-out cells",
            technical_status="insufficient_boundary_events",
            pre_fdr_interpretation="insufficient",
            robustness_gate=False,
            evidence_note=(
                f"frozen >=0.5 boundary candidates: all-cell={all_cell_candidates}/{len(conditional)} transitions; "
                f"train-split={split_candidates}/{h6_events} pilot events; max all-cell={conditional.max() if len(conditional) else np.nan:.3f}"
            ),
        )
    )

    h7 = _first(h7_summary, selection_rule="strongest_exact_margin")
    h7_groups = _int(h7, "source_event_groups")
    h7_pause_events = int(
        h7_context.get("route_timing_relation", pd.Series(dtype=str))
        .astype(str)
        .eq("pre_departure_pause")
        .sum()
    )
    h7_status = (
        "incompatible_event_context"
        if h7_pause_events == 0
        else "insufficient_source_groups"
        if h7_groups < 10
        else "complete"
    )
    rows.append(
        _row(
            "H7",
            estimand="off-SWR versus SWR pause timing and chosen-route commitment",
            expected_direction="off-SWR earlier/less committed",
            events=h7_groups,
            rats=_int(h7, "candidate_rats"),
            sessions=_int(h7, "candidate_sessions"),
            estimate=np.nan,
            ci_low=np.nan,
            ci_high=np.nan,
            primary_p_value=np.nan,
            null_control="source-event de-duplication and SWR-positive comparison",
            technical_status=h7_status,
            pre_fdr_interpretation="insufficient" if h7_status != "complete" else "inconclusive",
            robustness_gate=False,
            evidence_note=(
                f"{_int(h7, 'trajectory_confident_candidates')}/{h7_groups} source-deduplicated off-SWR candidates are trajectory-confident; "
                f"pause-before-departure candidates={h7_pause_events}/{h7_groups}"
            ),
        )
    )

    h8 = h8_summary.iloc[0] if len(h8_summary) else pd.Series(dtype=object)
    h8_gate = _first(h8_gates, gate="ordered_grammar_exceeds_whole_bin_shuffle")
    h8_robust = bool(h8_gate.get("passed", False))
    rows.append(
        _row(
            "H8",
            estimand="ordered trajectory grammar fraction minus whole-bin shuffle",
            expected_direction="positive",
            events=_int(h8, "events"),
            rats=_int(h8, "rats"),
            sessions=0,
            estimate=_float(h8, "ordered_trajectory_fraction_excess"),
            ci_low=np.nan,
            ci_high=np.nan,
            primary_p_value=_float(h8, "empirical_p_value_one_sided"),
            null_control="whole-population-bin temporal-order shuffle",
            technical_status="complete" if not h8.empty else "missing",
            pre_fdr_interpretation="directional_pass_unadjusted" if h8_robust else "inconclusive",
            robustness_gate=h8_robust,
            evidence_note=(
                f"original={_float(h8, 'original_ordered_trajectory_fraction'):.3f}; "
                f"shuffle median={_float(h8, 'median_shuffle_ordered_trajectory_fraction'):.3f}"
            ),
        )
    )

    h9 = _first(
        h9_inference,
        population_contrast="all",
        metric="post_minus_pre_time_order_advantage_imm_minus_fragmented",
    )
    h9_robust = bool(h9.get("positive_robust", False))
    rows.append(
        _row(
            "H9",
            estimand="POST minus PRE time-order advantage for IMM minus fragmented",
            expected_direction="positive",
            events=320,
            rats=_int(h9, "animals"),
            sessions=8,
            estimate=_float(h9, "equal_animal_mean"),
            ci_low=_float(h9, "rat_bootstrap_ci_low"),
            ci_high=_float(h9, "rat_bootstrap_ci_high"),
            primary_p_value=_float(h9, "one_sided_sign_test_p"),
            null_control="matched PRE/POST events and equal-animal inference",
            technical_status="complete" if not h9.empty else "missing",
            pre_fdr_interpretation="selective_order_effect_without_validated_clean_events" if h9_robust else "inconclusive",
            robustness_gate=False,
            evidence_note="time-order advantage rises POST, but map-specific content, held-out prediction, and validated clean-IMM counts do not",
        )
    )

    h10 = _context_result(context_tests, "H10")
    h10_status = str(h10.get("status_before_campaign_fdr", "missing"))
    rows.append(
        _row(
            "H10",
            estimand=str(h10.get("test", "")),
            expected_direction=directions["H10"],
            events=_int(h10, "events"),
            rats=_int(h10, "rats"),
            sessions=_int(h10, "sessions"),
            estimate=_float(h10, "estimate"),
            ci_low=_float(h10, "rat_bootstrap_ci_low"),
            ci_high=_float(h10, "rat_bootstrap_ci_high"),
            primary_p_value=_float(h10, "permutation_p_value"),
            null_control=str(h10.get("null_control", "")),
            technical_status="complete" if not h10.empty else "missing",
            pre_fdr_interpretation=h10_status,
            robustness_gate=h10_status == "directional_pass_unadjusted",
            evidence_note=str(h10.get("outcome", "")),
        )
    )
    result = pd.DataFrame(rows)
    if result["hypothesis"].tolist() != list(HYPOTHESES):
        raise AssertionError("campaign output must contain H1-H10 in frozen order")
    result["fdr_input_p_value"] = result["primary_p_value"].fillna(1.0)
    result["bh_q_value_10_hypotheses"] = benjamini_hochberg(result["fdr_input_p_value"])
    result["campaign_significant"] = (
        result["bh_q_value_10_hypotheses"].le(0.05)
        & result["robustness_gate"].astype(bool)
    )

    def final_status(row: pd.Series) -> str:
        technical = str(row["technical_status"])
        if technical.startswith("insufficient") or technical == "incompatible_event_context":
            return "insufficient"
        if technical == "missing":
            return "technical_failure"
        if bool(row["campaign_significant"]):
            return "supported"
        if np.isfinite(row["primary_p_value"]) and float(row["primary_p_value"]) <= 0.05:
            return "selective_unadjusted_only"
        return "not_supported"

    result["final_status"] = result.apply(final_status, axis=1)
    return result


def map_off_swr_route_context(
    decisions: pd.DataFrame,
    route_segments: pd.DataFrame,
    *,
    selection_rule: str = "strongest_exact_margin",
) -> pd.DataFrame:
    """Map selected off-SWR windows to independently segmented behavior routes."""

    selected = decisions[
        decisions["selection_rule"].astype(str).eq(selection_rule)
    ].copy()
    rows: list[dict[str, object]] = []
    for candidate in selected.itertuples(index=False):
        event_time = 0.5 * (
            float(candidate.window_start_s) + float(candidate.window_end_s)
        )
        session_routes = route_segments[
            route_segments["session"].astype(str).eq(str(candidate.session))
        ]
        containing = session_routes[
            (session_routes["interval_start_time_s"].astype(float) <= event_time)
            & (session_routes["interval_end_time_s"].astype(float) >= event_time)
        ]
        following = session_routes[
            session_routes["movement_start_time_s"].astype(float) > event_time
        ]
        if len(containing):
            route = containing.sort_values("duration_s").iloc[0]
        elif len(following):
            route = following.sort_values("movement_start_time_s").iloc[0]
        else:
            route = pd.Series(dtype=object)
        if route.empty:
            relation = "no_current_or_future_route"
            time_to_departure = np.nan
            route_id = ""
        else:
            movement_start = float(route["movement_start_time_s"])
            movement_end = float(route["movement_end_time_s"])
            time_to_departure = movement_start - event_time
            route_id = str(route["route_id"])
            if event_time < movement_start:
                relation = "pre_departure_pause"
            elif event_time <= movement_end:
                relation = "during_segmented_movement"
            else:
                relation = "post_segmented_movement_within_interval"
        rows.append(
            {
                "session": str(candidate.session),
                "rat": str(candidate.rat),
                "source_event_index": int(candidate.event_index),
                "null_index": int(candidate.null_index),
                "off_swr_window_midpoint_s": event_time,
                "route_id": route_id,
                "route_timing_relation": relation,
                "time_to_route_movement_start_s": time_to_departure,
            }
        )
    return pd.DataFrame(rows)


def build_companion_tests(context_tests: pd.DataFrame, context_events: pd.DataFrame) -> pd.DataFrame:
    """Collect required and descriptive companions outside the H1-H10 FDR family."""

    rows: list[dict[str, object]] = []
    h1 = _first(context_tests, hypothesis="H1", role="companion_required")
    if not h1.empty:
        rows.append(
            {
                "hypothesis": "H1",
                "test": str(h1["test"]),
                "events": _int(h1, "events"),
                "estimate": _float(h1, "estimate"),
                "p_value_descriptive": _float(h1, "permutation_p_value"),
                "interpretation": "required companion; H1 cannot pass when its primary pause-order test fails",
            }
        )
    companions = [
        ("duration_vs_nonfragmented_switch_count", "nonfragmented_map_switch_count", False),
        ("duration_vs_nonfragmented_switch_rate", "nonfragmented_map_switch_rate_hz", False),
        ("duration_vs_composition_gain_per_bin", "composition_index_cm", True),
    ]
    for name, outcome, divide_by_bins in companions:
        columns = ["log_event_duration_s", outcome, "n_time"]
        frame = context_events[columns].replace([np.inf, -np.inf], np.nan).dropna().copy()
        if divide_by_bins:
            frame[outcome] = frame[outcome].astype(float) / frame["n_time"].astype(float)
        statistic, p_value = spearmanr(frame["log_event_duration_s"], frame[outcome]) if len(frame) >= 3 else (np.nan, np.nan)
        rows.append(
            {
                "hypothesis": "H10",
                "test": name,
                "events": int(len(frame)),
                "estimate": float(statistic),
                "p_value_descriptive": float(p_value),
                "interpretation": "descriptive companion; not a replacement for the frozen per-bin primary test",
            }
        )
    return pd.DataFrame(rows)


def build_gate_summary(results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    def add(gate: str, passed: bool, observed: object, criterion: str) -> None:
        rows.append({"gate": gate, "passed": bool(passed), "observed": observed, "criterion": criterion})

    add("all_ten_hypotheses_present", len(results) == 10 and results["hypothesis"].nunique() == 10, len(results), "exactly H1-H10")
    add("all_ten_enter_fdr_family", len(results["fdr_input_p_value"]) == 10 and results["fdr_input_p_value"].notna().all(), len(results["fdr_input_p_value"]), "ten finite BH inputs; insufficient tests use p=1")
    add("no_incomplete_hypothesis_promoted", not results.loc[results["technical_status"].ne("complete"), "campaign_significant"].any(), int(results.loc[results["technical_status"].ne("complete"), "campaign_significant"].sum()), "zero incomplete or insufficient hypotheses promoted")
    add("claim_status_assigned", results["final_status"].notna().all(), int(results["final_status"].notna().sum()), "one final status per hypothesis")
    add("overall", all(row["passed"] for row in rows), f"{sum(row['passed'] for row in rows)}/{len(rows)}", "all campaign reporting gates pass")
    return pd.DataFrame(rows)


def write_report(results: pd.DataFrame, companions: pd.DataFrame, gates: pd.DataFrame, path: Path) -> None:
    supported = results[results["final_status"].eq("supported")]
    selective = results[results["final_status"].eq("selective_unadjusted_only")]
    insufficient = results[results["final_status"].eq("insufficient")]
    lines = [
        "# Replay-behavior hypothesis campaign",
        "",
        f"Reporting gates pass: **{bool(gates.set_index('gate').loc['overall', 'passed'])}**.",
        "",
        "All ten frozen hypotheses enter one Benjamini-Hochberg family. Missing or "
        "insufficient primary tests enter as p=1; companion tests do not replace primaries.",
        "",
        f"Campaign-supported hypotheses: **{len(supported)}/10**.",
        f"Unadjusted-only signals: **{len(selective)}**. Insufficient tests: **{len(insufficient)}**.",
        "",
        "## Primary results",
        "",
        "```text",
        results[["hypothesis", "estimate", "primary_p_value", "bh_q_value_10_hypotheses", "final_status"]].to_string(index=False),
        "```",
        "",
        "## Interpretation boundary",
        "",
        "This campaign tests behavioral specialization of replay dynamics. It does not "
        "alter the independently established Pfeiffer/Foster evidence that clean-IMM "
        "events have order-dependent, map-specific posterior content and held-out-cell "
        "generalization. A null behavioral association is not evidence that those neural "
        "dynamics are absent.",
        "",
        "## Companion tests",
        "",
        "```text",
        companions.to_string(index=False),
        "```",
    ]
    path.write_text("\n".join(lines) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context-tests", required=True)
    parser.add_argument("--context-events", required=True)
    parser.add_argument("--h5-tests", required=True)
    parser.add_argument("--h6-transitions", required=True)
    parser.add_argument("--h6-splits", required=True)
    parser.add_argument("--h7-summary", required=True)
    parser.add_argument("--h7-decisions", required=True)
    parser.add_argument("--route-segments", required=True)
    parser.add_argument("--h8-summary", required=True)
    parser.add_argument("--h8-gates", required=True)
    parser.add_argument("--h9-inference", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    inputs = {
        "context_tests": pd.read_csv(args.context_tests),
        "context_events": pd.read_csv(args.context_events),
        "h5_tests": pd.read_csv(args.h5_tests),
        "h6_transitions": pd.read_csv(args.h6_transitions),
        "h6_splits": pd.read_csv(args.h6_splits),
        "h7_summary": pd.read_csv(args.h7_summary),
        "h7_decisions": pd.read_csv(args.h7_decisions),
        "route_segments": pd.read_csv(args.route_segments),
        "h8_summary": pd.read_csv(args.h8_summary),
        "h8_gates": pd.read_csv(args.h8_gates),
        "h9_inference": pd.read_csv(args.h9_inference),
    }
    h7_context = map_off_swr_route_context(
        inputs["h7_decisions"],
        inputs["route_segments"],
    )
    results = build_campaign_results(
        context_tests=inputs["context_tests"],
        h5_tests=inputs["h5_tests"],
        h6_transitions=inputs["h6_transitions"],
        h6_splits=inputs["h6_splits"],
        h7_summary=inputs["h7_summary"],
        h7_context=h7_context,
        h8_summary=inputs["h8_summary"],
        h8_gates=inputs["h8_gates"],
        h9_inference=inputs["h9_inference"],
    )
    companions = build_companion_tests(inputs["context_tests"], inputs["context_events"])
    gates = build_gate_summary(results)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    results.to_csv(output / RESULT_OUTPUT, index=False)
    results[["hypothesis", "primary_p_value", "fdr_input_p_value", "bh_q_value_10_hypotheses", "campaign_significant"]].to_csv(output / FDR_OUTPUT, index=False)
    companions.to_csv(output / COMPANION_OUTPUT, index=False)
    h7_context.to_csv(output / H7_CONTEXT_OUTPUT, index=False)
    gates.to_csv(output / GATE_OUTPUT, index=False)
    write_report(results, companions, gates, output / REPORT_OUTPUT)
    manifest = {
        "analysis": "replay_behavior_hypothesis_campaign_H1_H10",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "fdr_family_size": 10,
        "missing_primary_p_policy": "enter_as_p_equals_1",
        "inputs": {key: str(Path(getattr(args, key.replace('_', '-'), '')).resolve()) for key in []},
        "input_paths": {
            key: str(Path(value).resolve())
            for key, value in vars(args).items()
            if key != "output_dir"
        },
        "claim_boundary": "behavioral specialization only; standalone PF model/content gates remain separate",
    }
    (output / MANIFEST_OUTPUT).write_text(json.dumps(manifest, indent=2) + "\n")
    print(results.to_string(index=False))
    print(gates.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
