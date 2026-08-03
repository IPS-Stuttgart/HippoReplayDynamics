#!/usr/bin/env python3
"""Assemble the frozen replay commitment/composition hypothesis report."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _provenance import build_script_provenance  # noqa: E402
from scripts.test_replay_dynamics_behavior_hypotheses import (  # noqa: E402
    PRIMARY_PREDICTOR,
    _analysis_frame,
    adjusted_coefficient,
)


EVENT_SHIFT_OUTPUT = "replay_dynamics_behavior_event_time_shift_null_scores.csv"
NULL_OUTPUT = "replay_dynamics_behavior_null_summary.csv"
GATE_OUTPUT = "replay_dynamics_behavior_gate_summary.csv"
REPORT_OUTPUT = "replay_commitment_composition_report.md"
MANIFEST_OUTPUT = "replay_commitment_composition_report_manifest.json"


PRIMARY_SPECS = (
    (
        "composition_decreases_with_momentum_axis",
        "composition_index",
        -1,
    ),
    (
        "future_commitment_increases_with_momentum_axis",
        "future_commitment_index",
        1,
    ),
    (
        "momentum_axis_incrementally_predicts_actual_future_route",
        "actual_future_closer_than_alternatives",
        1,
    ),
)


def circularly_shift_predictor_within_session(
    frame: pd.DataFrame,
    *,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Break event/behavior alignment while preserving within-session serial structure."""

    shifted = frame.copy()
    order_column = "event_peak_s" if "event_peak_s" in shifted else "event_index"
    for _, indices in shifted.groupby("session", sort=True).groups.items():
        ordered_indices = (
            shifted.loc[list(indices)]
            .sort_values(order_column, kind="stable")
            .index.to_numpy()
        )
        if len(ordered_indices) < 2:
            continue
        offset = int(rng.integers(1, len(ordered_indices)))
        values = shifted.loc[ordered_indices, PRIMARY_PREDICTOR].to_numpy(copy=True)
        shifted.loc[ordered_indices, PRIMARY_PREDICTOR] = np.roll(values, offset)
    return shifted


def build_event_time_shift_null(
    events: pd.DataFrame,
    *,
    replicates: int = 2000,
    seed: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = _analysis_frame(events)
    frame["actual_future_closer_than_alternatives"] = (
        frame["future_commitment_index"] > 0.0
    ).astype(float)
    frame.loc[
        frame["future_commitment_index"].isna(),
        "actual_future_closer_than_alternatives",
    ] = np.nan
    rng = np.random.default_rng(seed)
    score_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for test_index, (test, outcome, expected_sign) in enumerate(PRIMARY_SPECS):
        selected = frame[
            np.isfinite(pd.to_numeric(frame[outcome], errors="coerce"))
            & np.isfinite(pd.to_numeric(frame[PRIMARY_PREDICTOR], errors="coerce"))
        ].copy()
        observed = adjusted_coefficient(selected, outcome=outcome)[0]
        null_values: list[float] = []
        for replicate in range(int(replicates)):
            shifted = circularly_shift_predictor_within_session(selected, rng=rng)
            value = adjusted_coefficient(shifted, outcome=outcome)[0]
            if np.isfinite(value):
                null_values.append(float(value))
                score_rows.append(
                    {
                        "test": test,
                        "replicate": replicate,
                        "adjusted_standardized_coefficient": value,
                    }
                )
        null = np.asarray(null_values, dtype=float)
        if len(null):
            extreme = (
                int(np.sum(null <= observed))
                if expected_sign < 0
                else int(np.sum(null >= observed))
            )
            empirical_p = float((1 + extreme) / (1 + len(null)))
        else:
            empirical_p = np.nan
        summary_rows.append(
            {
                "control": "within_session_circular_event_order_shift",
                "test": test,
                "outcome": outcome,
                "events": int(len(selected)),
                "observed_adjusted_coefficient": observed,
                "null_median": float(np.median(null)) if len(null) else np.nan,
                "null_p05": float(np.quantile(null, 0.05)) if len(null) else np.nan,
                "null_p95": float(np.quantile(null, 0.95)) if len(null) else np.nan,
                "directional_empirical_p": empirical_p,
                "replicates_completed": int(len(null)),
                "status": (
                    "supported"
                    if np.isfinite(empirical_p)
                    and empirical_p < 0.05
                    and observed * expected_sign > 0.0
                    else "not_supported"
                ),
                "interpretation_role": "primary_behavior_association_null",
            }
        )
    return pd.DataFrame(score_rows), pd.DataFrame(summary_rows)


def _gate_pass(frame: pd.DataFrame, gate: str) -> bool:
    if frame.empty or "gate" not in frame or "passed" not in frame:
        return False
    match = frame[frame["gate"].astype(str).eq(gate)]
    if match.empty:
        return False
    return str(match["passed"].iloc[0]).strip().lower() in {"true", "1", "yes", "pass"}


def classify_hypothesis(
    *,
    composition_supported: bool,
    commitment_supported: bool,
    boundary_supported: bool,
    external_supported: bool,
) -> str:
    if (
        composition_supported
        and commitment_supported
        and boundary_supported
        and external_supported
    ):
        return "strong_support"
    if composition_supported and not commitment_supported:
        return "composition_only"
    if commitment_supported and not composition_supported:
        return "commitment_only"
    if composition_supported and commitment_supported:
        return "within_dataset_support_without_boundary_or_replication"
    return "neither_primary_specialization_supported"


def _external_confirmation_ready(external: pd.DataFrame) -> bool:
    if external.empty or "suitable_for_commitment_confirmation" not in external:
        return False
    values = external["suitable_for_commitment_confirmation"].astype(str).str.lower()
    return bool(values.isin({"true", "1", "yes"}).any())


def build_null_summary(
    *,
    event_shift_summary: pd.DataFrame,
    boundary_summary: pd.DataFrame,
    boundary_gates: pd.DataFrame,
    matched_summary: pd.DataFrame,
    time_order_gates: pd.DataFrame | None,
    heldout_gates: pd.DataFrame | None,
) -> pd.DataFrame:
    rows = event_shift_summary.to_dict("records")
    for row in boundary_summary.itertuples(index=False):
        rows.append(
            {
                "control": "circular_IMM_boundary_shift",
                "test": str(row.metric),
                "outcome": str(row.metric),
                "events": int(row.events),
                "observed_adjusted_coefficient": float(row.median_advantage),
                "null_median": 0.0,
                "null_p05": np.nan,
                "null_p95": np.nan,
                "directional_empirical_p": np.nan,
                "replicates_completed": np.nan,
                "status": (
                    "supported"
                    if str(row.metric) == "switch_alignment_advantage"
                    and _gate_pass(boundary_gates, "boundary_specificity_supported")
                    else (
                        "segmentation_specificity_only"
                        if str(row.metric) == "composition_boundary_advantage"
                        and float(row.rat_bootstrap_ci_low) > 0.0
                        else "not_supported"
                    )
                ),
                "interpretation_role": "behavior_boundary_null",
            }
        )
    for row in matched_summary.itertuples(index=False):
        expected_positive = str(row.expected_direction) == "positive"
        observed = float(row.median_treated_minus_control)
        rows.append(
            {
                "control": "quality_matched_raw_momentum_vs_clean_IMM",
                "test": str(row.test),
                "outcome": str(row.test),
                "events": int(row.matched_pairs),
                "observed_adjusted_coefficient": observed,
                "null_median": 0.0,
                "null_p05": np.nan,
                "null_p95": np.nan,
                "directional_empirical_p": np.nan,
                "replicates_completed": np.nan,
                "status": (
                    "predicted_direction_descriptive"
                    if (observed > 0.0) == expected_positive
                    else "opposite_direction_descriptive"
                ),
                "interpretation_role": "secondary_descriptive_only",
            }
        )
    rows.extend(
        [
            {
                "control": "whole_replay_time_bin_shuffle",
                "test": "clean_IMM_model_identity_requires_time_order",
                "outcome": "IMM_minus_fragmented_log_evidence",
                "events": np.nan,
                "observed_adjusted_coefficient": np.nan,
                "null_median": np.nan,
                "null_p05": np.nan,
                "null_p95": np.nan,
                "directional_empirical_p": np.nan,
                "replicates_completed": np.nan,
                "status": (
                    "supported_model_validity_only"
                    if time_order_gates is not None
                    and _gate_pass(time_order_gates, "overall")
                    else "not_available"
                ),
                "interpretation_role": "model_validity_not_behavior_function",
            },
            {
                "control": "heldout_cells",
                "test": "IMM_model_identity_generalizes_to_heldout_cells",
                "outcome": "heldout_IMM_minus_fragmented",
                "events": np.nan,
                "observed_adjusted_coefficient": np.nan,
                "null_median": np.nan,
                "null_p05": np.nan,
                "null_p95": np.nan,
                "directional_empirical_p": np.nan,
                "replicates_completed": np.nan,
                "status": (
                    "supported_model_validity_only"
                    if _heldout_overall(heldout_gates)
                    else "not_available"
                ),
                "interpretation_role": "model_validity_not_behavior_function",
            },
            {
                "control": "wrong_spatial_map_behavior_outcome",
                "test": "primary_behavior_specialization_wrong_map_sensitivity",
                "outcome": "composition_and_commitment",
                "events": np.nan,
                "observed_adjusted_coefficient": np.nan,
                "null_median": np.nan,
                "null_p05": np.nan,
                "null_p95": np.nan,
                "directional_empirical_p": np.nan,
                "replicates_completed": np.nan,
                "status": "not_triggered_because_primary_associations_failed",
                "interpretation_role": "positive_claim_control_not_used_to_rescue_null_result",
            },
        ]
    )
    return pd.DataFrame(rows)


def _heldout_overall(gates: pd.DataFrame | None) -> bool:
    if gates is None or gates.empty:
        return False
    match = gates[
        gates.get("category", pd.Series(dtype=str)).astype(str).eq("summary")
        & gates["gate"].astype(str).eq("overall")
    ]
    return bool(
        not match.empty
        and str(match["passed"].iloc[0]).strip().lower() in {"true", "1", "yes"}
    )


def build_final_gates(
    *,
    feasibility_gates: pd.DataFrame,
    metric_gates: pd.DataFrame,
    primary: pd.DataFrame,
    primary_gates: pd.DataFrame,
    boundary_summary: pd.DataFrame,
    boundary_gates: pd.DataFrame,
    event_shift_summary: pd.DataFrame,
    external: pd.DataFrame,
    time_order_gates: pd.DataFrame | None,
    heldout_gates: pd.DataFrame | None,
) -> tuple[pd.DataFrame, str]:
    status = primary.set_index("test")["status"]
    composition_supported = status.get(
        "composition_decreases_with_momentum_axis", "missing"
    ) == "supported" and all(
        _gate_pass(primary_gates, gate)
        for gate in (
            "composition_decreases_with_momentum_axis_primary_support",
            "composition_decreases_with_momentum_axis_all_rats_estimable",
            "composition_decreases_with_momentum_axis_leave_one_rat_out_direction",
        )
    )
    commitment_supported = status.get(
        "future_commitment_increases_with_momentum_axis", "missing"
    ) == "supported" and all(
        _gate_pass(primary_gates, gate)
        for gate in (
            "future_commitment_increases_with_momentum_axis_primary_support",
            "future_commitment_increases_with_momentum_axis_all_rats_estimable",
            "future_commitment_increases_with_momentum_axis_leave_one_rat_out_direction",
        )
    )
    incremental_supported = status.get(
        "momentum_axis_incrementally_predicts_actual_future_route", "missing"
    ) == "supported" and all(
        _gate_pass(primary_gates, gate)
        for gate in (
            "momentum_axis_incrementally_predicts_actual_future_route_primary_support",
            "momentum_axis_incrementally_predicts_actual_future_route_all_rats_estimable",
            "momentum_axis_incrementally_predicts_actual_future_route_leave_one_rat_out_direction",
        )
    )
    boundary_supported = _gate_pass(boundary_gates, "boundary_specificity_supported")
    external_supported = _external_confirmation_ready(external)
    shifted = event_shift_summary.set_index("test")["status"]
    event_shift_composition = shifted.get(
        "composition_decreases_with_momentum_axis", "missing"
    ) == "supported"
    event_shift_commitment = shifted.get(
        "future_commitment_increases_with_momentum_axis", "missing"
    ) == "supported"
    metric_overall = _gate_pass(metric_gates, "overall")
    feasibility_ready = _gate_pass(feasibility_gates, "pf_primary_analysis_ready")
    time_order_ready = bool(
        time_order_gates is not None and _gate_pass(time_order_gates, "overall")
    )
    heldout_ready = _heldout_overall(heldout_gates)
    rows = [
        ("phase0_pf_analysis_ready", feasibility_ready, feasibility_ready, True),
        ("event_metrics_and_decoder_controls_ready", metric_overall, metric_overall, True),
        (
            "composition_primary_supported",
            composition_supported,
            status.get("composition_decreases_with_momentum_axis", "missing"),
            "supported",
        ),
        (
            "commitment_primary_supported",
            commitment_supported,
            status.get("future_commitment_increases_with_momentum_axis", "missing"),
            "supported",
        ),
        (
            "incremental_future_route_prediction_supported",
            incremental_supported,
            status.get(
                "momentum_axis_incrementally_predicts_actual_future_route", "missing"
            ),
            "supported",
        ),
        (
            "switch_boundary_route_identity_specificity_supported",
            boundary_supported,
            boundary_supported,
            True,
        ),
        (
            "composition_survives_event_order_shift_null",
            event_shift_composition,
            event_shift_composition,
            True,
        ),
        (
            "commitment_survives_event_order_shift_null",
            event_shift_commitment,
            event_shift_commitment,
            True,
        ),
        ("model_time_order_gate_available", time_order_ready, time_order_ready, True),
        ("model_heldout_cell_gate_available", heldout_ready, heldout_ready, True),
        (
            "independent_dataset_predicted_direction",
            external_supported,
            external_supported,
            True,
        ),
    ]
    gates = pd.DataFrame(
        [
            {"gate": gate, "passed": bool(passed), "value": value, "required": required}
            for gate, passed, value, required in rows
        ]
    )
    strong = bool(gates["passed"].all())
    gates.loc[len(gates)] = {
        "gate": "overall_strong_target_hypothesis_support",
        "passed": strong,
        "value": int(gates["passed"].sum()),
        "required": len(rows),
    }
    classification = classify_hypothesis(
        composition_supported=composition_supported,
        commitment_supported=commitment_supported,
        boundary_supported=boundary_supported,
        external_supported=external_supported,
    )
    return gates, classification


def run_report(
    *,
    feasibility_dir: str | Path,
    metrics_dir: str | Path,
    primary_dir: str | Path,
    boundary_dir: str | Path,
    output_dir: str | Path,
    time_order_gate_csv: str | Path | None = None,
    heldout_gate_csv: str | Path | None = None,
    null_replicates: int = 2000,
    seed: int = 1,
) -> dict[str, Path]:
    feasibility_root = Path(feasibility_dir)
    metrics_root = Path(metrics_dir)
    primary_root = Path(primary_dir)
    boundary_root = Path(boundary_dir)
    inputs: dict[str, Path] = {
        "feasibility_gates": feasibility_root
        / "replay_commitment_composition_feasibility_gate_summary.csv",
        "external": feasibility_root
        / "replay_commitment_composition_external_dataset_suitability.csv",
        "event_metrics": metrics_root / "replay_event_commitment_composition_metrics.csv",
        "metric_gates": metrics_root
        / "replay_event_commitment_composition_metrics_gate_summary.csv",
        "primary": primary_root / "replay_dynamics_behavior_primary_tests.csv",
        "primary_gates": primary_root
        / "replay_dynamics_behavior_primary_gate_summary.csv",
        "matched": primary_root
        / "replay_dynamics_behavior_matched_sensitivity_summary.csv",
        "boundary_summary": boundary_root
        / "replay_dynamics_behavior_boundary_shift_summary.csv",
        "boundary_gates": boundary_root
        / "replay_dynamics_behavior_boundary_shift_gate_summary.csv",
    }
    if time_order_gate_csv is not None:
        inputs["time_order_gates"] = Path(time_order_gate_csv)
    if heldout_gate_csv is not None:
        inputs["heldout_gates"] = Path(heldout_gate_csv)
    tables = {name: pd.read_csv(path) for name, path in inputs.items()}
    event_shift_scores, event_shift_summary = build_event_time_shift_null(
        tables["event_metrics"],
        replicates=int(null_replicates),
        seed=int(seed),
    )
    time_order = tables.get("time_order_gates")
    heldout = tables.get("heldout_gates")
    null_summary = build_null_summary(
        event_shift_summary=event_shift_summary,
        boundary_summary=tables["boundary_summary"],
        boundary_gates=tables["boundary_gates"],
        matched_summary=tables["matched"],
        time_order_gates=time_order,
        heldout_gates=heldout,
    )
    gates, classification = build_final_gates(
        feasibility_gates=tables["feasibility_gates"],
        metric_gates=tables["metric_gates"],
        primary=tables["primary"],
        primary_gates=tables["primary_gates"],
        boundary_summary=tables["boundary_summary"],
        boundary_gates=tables["boundary_gates"],
        event_shift_summary=event_shift_summary,
        external=tables["external"],
        time_order_gates=time_order,
        heldout_gates=heldout,
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, frame in {
        EVENT_SHIFT_OUTPUT: event_shift_scores,
        NULL_OUTPUT: null_summary,
        GATE_OUTPUT: gates,
    }.items():
        path = output / name
        frame.to_csv(path, index=False)
        paths[name] = path
    primary = tables["primary"].set_index("test")
    composition = primary.loc["composition_decreases_with_momentum_axis"]
    commitment = primary.loc["future_commitment_increases_with_momentum_axis"]
    incremental = primary.loc[
        "momentum_axis_incrementally_predicts_actual_future_route"
    ]
    boundary = tables["boundary_summary"].set_index("metric")
    segmentation = boundary.loc["composition_boundary_advantage"]
    shifted = event_shift_summary.set_index("test")
    shifted_composition = shifted.loc["composition_decreases_with_momentum_axis"]
    matched = tables["matched"].set_index("test")
    matched_composition = matched.loc["composition_momentum_minus_clean_imm"]
    matched_commitment = matched.loc["commitment_momentum_minus_clean_imm"]
    decoder_by_session = tables["event_metrics"].groupby("session")[
        "run_decoder_error_cm"
    ].first()
    report = [
        "# Replay commitment/composition hypothesis report",
        "",
        f"**Decision:** `{classification}`",
        "",
        "## Frozen design",
        "",
        "The primary predictor is the predeclared continuous exact-sparse-momentum minus first-order-IMM log-evidence margin. Only seven events were confident momentum wins, so categorical winner comparisons remain secondary. Behavioral route primitives were learned from RUN behavior only, with whole-route cross-validation. Composition excludes stationary and fragmented/jump MAP phases.",
        "",
        "## Primary results",
        "",
        f"Composition used {int(composition['events'])} events across four rats: adjusted standardized beta {float(composition['adjusted_standardized_coefficient']):.3f}, rat-bootstrap 95% CI [{float(composition['rat_cluster_bootstrap_ci_low']):.3f}, {float(composition['rat_cluster_bootstrap_ci_high']):.3f}], `{composition['status']}`.",
        f"Its raw Spearman correlation was {float(composition['raw_spearman_r']):.3f}. The adjusted coefficient exceeded the directional within-session circular-shift null (empirical p={float(shifted_composition['directional_empirical_p']):.4f}) but failed rat-bootstrap and leave-one-rat-out robustness, so this is not counted as support.",
        f"Future commitment used {int(commitment['events'])} events: adjusted standardized beta {float(commitment['adjusted_standardized_coefficient']):.3f}, rat-bootstrap 95% CI [{float(commitment['rat_cluster_bootstrap_ci_low']):.3f}, {float(commitment['rat_cluster_bootstrap_ci_high']):.3f}], `{commitment['status']}`.",
        f"Incremental actual-future-route prediction was `{incremental['status']}` (beta {float(incremental['adjusted_standardized_coefficient']):.3f}).",
        f"The categorical quality-matched sensitivity was also unfavorable: momentum-minus-clean-IMM median composition difference {float(matched_composition['median_treated_minus_control']):.2f} cm across {int(matched_composition['matched_pairs'])} evaluable pairs (predicted negative), and commitment difference {float(matched_commitment['median_treated_minus_control']):.2f} cm across {int(matched_commitment['matched_pairs'])} pairs (predicted positive).",
        f"Cross-validated RUN decoder median errors were available for {int(decoder_by_session.notna().sum())}/{len(decoder_by_session)} sessions, spanning {float(decoder_by_session.min()):.2f}-{float(decoder_by_session.max()):.2f} cm.",
        "",
        "## Boundary null",
        "",
        "Route identity is the directed origin-well to destination-well pair, not a fine subpath-cluster ID.",
        f"Actual IMM boundaries did not improve route-identity-change alignment over circularly shifted boundaries: median advantage {float(boundary.loc['switch_alignment_advantage', 'median_advantage']):.3f}, 95% CI [{float(boundary.loc['switch_alignment_advantage', 'rat_bootstrap_ci_low']):.3f}, {float(boundary.loc['switch_alignment_advantage', 'rat_bootstrap_ci_high']):.3f}].",
        f"A narrower segmentation result survived: actual continuous-bout boundaries improved familiar-primitive-versus-whole-route fit by a median {float(segmentation['median_advantage']):.2f} cm relative to shifted boundaries, CI [{float(segmentation['rat_bootstrap_ci_low']):.2f}, {float(segmentation['rat_bootstrap_ci_high']):.2f}]. This does not establish IMM-versus-momentum functional specialization.",
        "",
        "## Interpretation",
        "",
        "The target hypothesis is not supported under the frozen definitions. Momentum-like evidence did not predict stronger imminent-route commitment, and clean-IMM evidence did not predict greater composition. IMM boundaries identify locally coherent continuous segments, but those boundaries are not preferentially aligned with changes between behavioral route identities.",
        "",
        "This negative behavioral specialization result does not alter the independent model-validation findings that clean IMM depends on temporal order and generalizes to held-out cells. It means those statistically validated dynamics have not been shown here to implement the proposed commitment-versus-composition division of labor.",
    ]
    report_path = output / REPORT_OUTPUT
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    paths[REPORT_OUTPUT] = report_path
    provenance = build_script_provenance(input_paths=inputs, cwd=ROOT)
    manifest = {
        "analysis": "replay_commitment_composition_target_hypothesis_report",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "classification": classification,
        "null_replicates": int(null_replicates),
        "seed": int(seed),
        "provenance": provenance,
    }
    manifest_path = output / MANIFEST_OUTPUT
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    paths[MANIFEST_OUTPUT] = manifest_path
    return paths


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feasibility-dir", required=True)
    parser.add_argument("--metrics-dir", required=True)
    parser.add_argument("--primary-dir", required=True)
    parser.add_argument("--boundary-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--time-order-gate")
    parser.add_argument("--heldout-gate")
    parser.add_argument("--null-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_report(
        feasibility_dir=args.feasibility_dir,
        metrics_dir=args.metrics_dir,
        primary_dir=args.primary_dir,
        boundary_dir=args.boundary_dir,
        output_dir=args.output_dir,
        time_order_gate_csv=args.time_order_gate,
        heldout_gate_csv=args.heldout_gate,
        null_replicates=args.null_replicates,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
