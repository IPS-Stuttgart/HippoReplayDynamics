#!/usr/bin/env python3
"""Merge hc-11 native-ripple holdout shards and apply the replication stopgate.

This reporter never rescored events. It verifies the frozen cohort and exact-core
coverage, rebuilds descriptive summaries, and decides whether a distributed
clean-IMM subset exists across animals and sessions.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
SRC_DIR = ROOT / "src"
for path in (SCRIPT_DIR, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from _provenance import build_script_provenance  # noqa: E402
from score_hc11_webshare_native_ripple_evidence import (  # noqa: E402
    DECODER_OUTPUT,
    DIRECTION_OUTPUT,
    EVIDENCE_OUTPUT,
    EXCLUSION_OUTPUT,
    GATE_OUTPUT,
    MANIFEST_OUTPUT,
    MODELS,
    PRIMARY_ENCODING_VARIANT,
    SELECTION_OUTPUT,
    UNIT_OUTPUT,
    direction_sensitivity,
    event_decisions,
)


COMBINED_EVIDENCE_OUTPUT = "hc11_native_ripple_holdout_event_model_evidence.csv"
DECISIONS_OUTPUT = "hc11_native_ripple_holdout_event_decisions.csv"
BY_SESSION_OUTPUT = "hc11_native_ripple_holdout_by_session.csv"
BY_ANIMAL_OUTPUT = "hc11_native_ripple_holdout_by_animal.csv"
MODEL_SUMMARY_OUTPUT = "hc11_native_ripple_holdout_model_summary.csv"
GATES_OUTPUT = "hc11_native_ripple_holdout_replication_gate_summary.csv"
EXCLUSIONS_OUTPUT = "hc11_native_ripple_holdout_prior_event_exclusions.csv"
REPORT_OUTPUT = "hc11_native_ripple_holdout_replication_report.md"
REPORT_MANIFEST_OUTPUT = "hc11_native_ripple_holdout_replication_manifest.json"


def _read_shards(shard_root: Path, filename: str) -> tuple[pd.DataFrame, list[Path]]:
    paths = sorted(shard_root.glob(f"*/{filename}"))
    if not paths:
        raise FileNotFoundError(f"no {filename} files found below {shard_root}")
    return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True), paths


def _summary(decisions: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    iterator = decisions.groupby(group_columns, sort=True) if group_columns else [((), decisions)]
    for keys, group in iterator:
        keys = keys if isinstance(keys, tuple) else (keys,)
        best = group["best_model"].value_counts()
        row = dict(zip(group_columns, keys, strict=True))
        row.update(
            {
                "events": int(len(group)),
                "trajectory_confident_count": int(group["trajectory_confident_claim"].sum()),
                "trajectory_confident_fraction": float(group["trajectory_confident_claim"].mean()),
                "stationary_confident_count": int(group["stationary_confident_claim"].sum()),
                "imm_confident_over_fragmented_count": int(group["imm_confident_over_fragmented"].sum()),
                "fragmented_confident_over_imm_count": int(group["fragmented_confident_over_imm"].sum()),
                "strict_clean_imm_count": int(group["strict_clean_imm"].sum()),
                "strict_clean_imm_fraction": float(group["strict_clean_imm"].mean()),
                "median_trajectory_minus_stationary": float(group["delta_trajectory_minus_stationary"].median()),
                "median_imm_minus_fragmented": float(group["delta_imm_minus_fragmented"].median()),
                "stationary_best_count": int(best.get("stationary", 0)),
                "diffusion_best_count": int(best.get("diffusion", 0)),
                "fragmented_best_count": int(best.get("fragmented", 0)),
                "first_order_imm_best_count": int(best.get("first_order_imm", 0)),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _gate_row(category: str, gate: str, passed: bool, value: object, detail: str) -> dict[str, object]:
    return {
        "category": category,
        "gate": gate,
        "passed": bool(passed),
        "value": value,
        "detail": detail,
    }


def build_report(
    *,
    shard_root: str | Path,
    output_dir: str | Path,
    expected_sessions: int,
    expected_events_per_session: int,
    margin_threshold: float,
) -> dict[str, pd.DataFrame | str]:
    shard_root = Path(shard_root).resolve()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    evidence, evidence_paths = _read_shards(shard_root, EVIDENCE_OUTPUT)
    selection, selection_paths = _read_shards(shard_root, SELECTION_OUTPUT)
    decoder, decoder_paths = _read_shards(shard_root, DECODER_OUTPUT)
    units, unit_paths = _read_shards(shard_root, UNIT_OUTPUT)
    exclusion_audit, exclusion_paths = _read_shards(shard_root, EXCLUSION_OUTPUT)
    shard_gates, shard_gate_paths = _read_shards(shard_root, GATE_OUTPUT)
    _, direction_paths = _read_shards(shard_root, DIRECTION_OUTPUT)
    manifest_paths = sorted(shard_root.glob(f"*/{MANIFEST_OUTPUT}"))
    manifests = [json.loads(path.read_text(encoding="utf-8")) for path in manifest_paths]

    event_keys = ["animal", "session", "event_id"]
    evidence_keys = [*event_keys, "encoding_variant", "model"]
    duplicate_selection = int(selection.duplicated(event_keys).sum())
    duplicate_evidence = int(evidence.duplicated(evidence_keys).sum())
    sessions = int(selection["session"].nunique()) if not selection.empty else 0
    animals = int(selection["animal"].nunique()) if not selection.empty else 0
    expected_events = int(sessions * expected_events_per_session)
    successful = evidence[evidence["status"].eq("success")]
    model_coverage = successful.groupby([*event_keys, "encoding_variant"])["model"].agg(lambda values: set(values))
    required_models_complete = bool(len(model_coverage) and model_coverage.map(lambda value: value == set(MODELS)).all())

    overlap_gates = shard_gates[shard_gates["gate"].eq("selected_events_exclude_prior_pilots")]
    no_prior_overlap = bool(len(overlap_gates) == sessions and overlap_gates["passed"].astype(bool).all())
    parameter_signatures = {
        (
            manifest.get("event_definition"),
            manifest.get("cohort_label"),
            tuple(manifest.get("models", [])),
            manifest.get("parameters", {}).get("time_bin_s"),
            manifest.get("parameters", {}).get("event_padding_s"),
            manifest.get("parameters", {}).get("event_ranking"),
        )
        for manifest in manifests
    }
    frozen_parameters_match = bool(len(manifests) == sessions and len(parameter_signatures) == 1)

    decisions = event_decisions(evidence, margin_threshold=margin_threshold)
    decisions["strict_clean_imm"] = (
        decisions["trajectory_confident_claim"].astype(bool)
        & decisions["imm_confident_over_fragmented"].astype(bool)
    )
    by_session = _summary(decisions, ["animal", "session", "geometry"])
    by_animal = _summary(decisions, ["animal"])
    model_summary = _summary(decisions, [])
    direction = direction_sensitivity(evidence)

    strict = decisions[decisions["strict_clean_imm"]]
    strict_events = int(len(strict))
    strict_animals = int(strict["animal"].nunique()) if strict_events else 0
    strict_sessions = int(strict["session"].nunique()) if strict_events else 0

    technical_rows = [
        _gate_row("technical", "expected_event_count_complete", len(selection) == expected_events and expected_events > 0, f"{len(selection)}/{expected_events}", "balanced session target"),
        _gate_row(
            "technical",
            "expected_sessions_present",
            sessions == expected_sessions and expected_sessions > 0,
            f"{sessions}/{expected_sessions}",
            "public native-ripple sessions",
        ),
        _gate_row("technical", "both_animals_present", animals == 2, animals, "Achilles and Cicero"),
        _gate_row("technical", "selection_keys_unique", duplicate_selection == 0, duplicate_selection, "duplicate event keys"),
        _gate_row("technical", "evidence_keys_unique", duplicate_evidence == 0, duplicate_evidence, "duplicate event/model keys"),
        _gate_row("technical", "all_model_rows_successful", len(successful) == len(evidence) and len(evidence) > 0, f"{len(successful)}/{len(evidence)}", "no failed model rows"),
        _gate_row("technical", "required_models_complete", required_models_complete, len(model_coverage), "four exact models for each event and encoding variant"),
        _gate_row("technical", "prior_pilot_overlap_zero", no_prior_overlap, len(overlap_gates), "all shard overlap gates pass"),
        _gate_row("technical", "frozen_parameters_match", frozen_parameters_match, len(parameter_signatures), "one parameter signature across shards"),
        _gate_row("technical", "decisions_complete", len(decisions) == len(selection) and len(selection) > 0, f"{len(decisions)}/{len(selection)}", "one primary decision per event"),
    ]
    technical_pass = all(bool(row["passed"]) for row in technical_rows)
    interpretation_rows = [
        _gate_row("interpretation", "strict_clean_imm_events_present", strict_events > 0, strict_events, "trajectory-confident and IMM-confident over fragmented"),
        _gate_row("interpretation", "strict_clean_imm_spans_both_animals", strict_animals >= 2, strict_animals, "required before external IMM replication ladder"),
        _gate_row("interpretation", "strict_clean_imm_spans_multiple_sessions", strict_sessions >= 2, strict_sessions, "required before external IMM replication ladder"),
    ]
    replication_supported = bool(
        technical_pass
        and strict_animals >= 2
        and strict_sessions >= 2
    )
    verdict = (
        "external_clean_imm_replication_candidate"
        if replication_supported
        else "external_clean_imm_replication_not_established"
    )
    recommendation = (
        "proceed_to_gate_2_3_4"
        if replication_supported
        else "stop_full_gate_ladder_no_distributed_strict_subset"
    )
    gates = pd.DataFrame(
        [
            *technical_rows,
            _gate_row("technical", "overall_technical", technical_pass, technical_pass, "all technical gates"),
            *interpretation_rows,
            _gate_row("decision", "external_clean_imm_replication_supported", replication_supported, verdict, recommendation),
        ]
    )

    exclusion_audit = exclusion_audit.drop_duplicates(event_keys).sort_values(event_keys).reset_index(drop=True)
    outputs = {
        COMBINED_EVIDENCE_OUTPUT: evidence,
        DECISIONS_OUTPUT: decisions,
        BY_SESSION_OUTPUT: by_session,
        BY_ANIMAL_OUTPUT: by_animal,
        MODEL_SUMMARY_OUTPUT: model_summary,
        GATES_OUTPUT: gates,
        EXCLUSIONS_OUTPUT: exclusion_audit,
    }
    for filename, frame in outputs.items():
        frame.to_csv(output_dir / filename, index=False)

    overall = model_summary.iloc[0]
    report = "\n".join(
        [
            "# hc-11 native-ripple holdout replication",
            "",
            f"Technical status: **{'pass' if technical_pass else 'fail'}**.",
            f"External clean-IMM replication status: **{verdict}**.",
            f"Recommended action: **{recommendation}**.",
            "",
            "## Result",
            "",
            f"- Events: {int(overall['events'])} across {sessions} sessions and {animals} animals.",
            f"- Trajectory-confident: {int(overall['trajectory_confident_count'])}/{int(overall['events'])}.",
            f"- IMM-confident over fragmented: {int(overall['imm_confident_over_fragmented_count'])}/{int(overall['events'])}.",
            f"- Strict clean IMM: {strict_events}/{int(overall['events'])}, spanning {strict_sessions} session(s) and {strict_animals} animal(s).",
            f"- Median trajectory-minus-stationary: {float(overall['median_trajectory_minus_stationary']):+.3f} log evidence.",
            f"- Median IMM-minus-fragmented: {float(overall['median_imm_minus_fragmented']):+.3f} log evidence.",
            "",
            "## Claim boundary",
            "",
            "The non-overlapping high-information holdout is a technically valid diagnostic, but the strict clean-IMM subset is not distributed across both animals and multiple sessions. It therefore does not establish external replication of the Pfeiffer/Foster clean-IMM result, and Gate 2/3/4 scaling is stopped.",
            "The event-strength ranking was informed by earlier hc-11 pilots, so this cohort is diagnostic rather than preregistered confirmation.",
            "hc-11 remains a constrained linear/circular maze dataset, not a 2D open field.",
            "",
        ]
    )
    (output_dir / REPORT_OUTPUT).write_text(report, encoding="utf-8")

    provenance_inputs = {
        **{f"evidence_shard_{index}": path for index, path in enumerate(evidence_paths)},
        **{f"selection_shard_{index}": path for index, path in enumerate(selection_paths)},
        **{f"decoder_shard_{index}": path for index, path in enumerate(decoder_paths)},
        **{f"unit_shard_{index}": path for index, path in enumerate(unit_paths)},
        **{f"exclusion_shard_{index}": path for index, path in enumerate(exclusion_paths)},
        **{f"gate_shard_{index}": path for index, path in enumerate(shard_gate_paths)},
        **{f"direction_shard_{index}": path for index, path in enumerate(direction_paths)},
        **{f"manifest_shard_{index}": path for index, path in enumerate(manifest_paths)},
    }
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "hc-11_Grosmark_Buzsaki_Webshare",
        "analysis": "non_rescoring_native_ripple_holdout_replication_stopgate",
        "primary_encoding_variant": PRIMARY_ENCODING_VARIANT,
        "models": list(MODELS),
        "expected_events_per_session": int(expected_events_per_session),
        "expected_sessions": int(expected_sessions),
        "margin_threshold": float(margin_threshold),
        "verdict": verdict,
        "recommended_action": recommendation,
        **build_script_provenance(input_paths=provenance_inputs),
    }
    (output_dir / REPORT_MANIFEST_OUTPUT).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "evidence": evidence,
        "decisions": decisions,
        "by_session": by_session,
        "by_animal": by_animal,
        "model_summary": model_summary,
        "gates": gates,
        "direction": direction,
        "verdict": verdict,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-events-per-session", type=int, default=50)
    parser.add_argument("--expected-sessions", type=int, default=5)
    parser.add_argument("--margin-threshold", type=float, default=5.5)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    build_report(
        shard_root=args.shard_root,
        output_dir=args.output_dir,
        expected_sessions=args.expected_sessions,
        expected_events_per_session=args.expected_events_per_session,
        margin_threshold=args.margin_threshold,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
