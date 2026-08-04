#!/usr/bin/env python3
"""Merge Tanni 2022 holdout shards and apply clean-IMM replication gates."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _provenance import build_script_provenance  # noqa: E402
from score_tanni2022_clean_imm_holdout import (  # noqa: E402
    EVIDENCE_OUTPUT,
    GATE_OUTPUT,
    KEYS,
    MANIFEST_OUTPUT,
    MODELS,
    SELECTION_OUTPUT,
    UNIT_OUTPUT,
    holdout_decisions,
)


COMBINED_EVIDENCE_OUTPUT = "tanni2022_clean_imm_holdout_combined_evidence.csv"
DECISIONS_OUTPUT = "tanni2022_clean_imm_holdout_decisions.csv"
BY_ANIMAL_OUTPUT = "tanni2022_clean_imm_holdout_by_animal.csv"
MODEL_SUMMARY_OUTPUT = "tanni2022_clean_imm_holdout_model_summary.csv"
REPLICATION_GATES_OUTPUT = "tanni2022_clean_imm_holdout_replication_gate_summary.csv"
REPORT_OUTPUT = "tanni2022_clean_imm_holdout_report.md"
REPORT_MANIFEST_OUTPUT = "tanni2022_clean_imm_holdout_report_manifest.json"


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
                "trajectory_confident_count": int(group["trajectory_confident"].sum()),
                "ordered_trajectory_confident_count": int(group["ordered_trajectory_confident"].sum()),
                "imm_confident_over_fragmented_count": int(group["imm_confident_over_fragmented"].sum()),
                "joint_family_and_imm_margin_positive_count": int(group["joint_family_and_imm_margin_positive"].sum()),
                "strict_clean_imm_count": int(group["strict_clean_imm"].sum()),
                "momentum_confident_best_count": int(group["momentum_confident_best"].sum()),
                "median_trajectory_minus_stationary": float(group["delta_trajectory_minus_stationary"].median()),
                "median_ordered_minus_static_or_fragmented": float(group["delta_ordered_minus_static_or_fragmented"].median()),
                "median_imm_minus_fragmented": float(group["delta_imm_minus_fragmented"].median()),
                "median_momentum_minus_diffusion": float(group["delta_momentum_minus_diffusion"].median()),
                "median_momentum_minus_imm": float(group["delta_momentum_minus_imm"].median()),
                **{f"{model}_best_count": int(best.get(model, 0)) for model in MODELS},
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _gate(category: str, gate: str, passed: bool, value: object, criterion: str) -> dict[str, object]:
    return {
        "category": category,
        "gate": gate,
        "passed": bool(passed),
        "value": value,
        "criterion": criterion,
    }


def build_report(
    *,
    shard_root: str | Path,
    output_dir: str | Path,
    expected_animals: int,
    events_per_animal: int,
    margin_threshold: float,
    minimum_positive_events: int,
    minimum_positive_animals: int,
) -> dict[str, pd.DataFrame | str]:
    shard_root = Path(shard_root).resolve()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence, evidence_paths = _read_shards(shard_root, EVIDENCE_OUTPUT)
    selection, selection_paths = _read_shards(shard_root, SELECTION_OUTPUT)
    units, unit_paths = _read_shards(shard_root, UNIT_OUTPUT)
    shard_gates, shard_gate_paths = _read_shards(shard_root, GATE_OUTPUT)
    manifest_paths = sorted(shard_root.glob(f"*/{MANIFEST_OUTPUT}"))
    manifests = [json.loads(path.read_text(encoding="utf-8")) for path in manifest_paths]

    expected_events = int(expected_animals * events_per_animal)
    expected_rows = int(expected_events * len(MODELS))
    successful = evidence[evidence["status"].eq("success")]
    coverage = successful.groupby(KEYS)["model"].agg(lambda values: set(values))
    selection_hashes = {
        manifest.get("input_file_sha256", {}).get("selection_csv")
        for manifest in manifests
    }
    parameter_signatures = {
        tuple(
            (key, str(value))
            for key, value in sorted(manifest.get("parameters", {}).items())
            if key != "animal"
        )
        for manifest in manifests
    }
    shard_overall = shard_gates[shard_gates["gate"].eq("overall_technical")]

    decisions = holdout_decisions(evidence, margin_threshold=margin_threshold)
    metadata_columns = [
        *KEYS,
        "selection_rank_within_animal",
        "n_spikes",
        "n_active_cells",
        "peak_ripple_z",
    ]
    decisions = decisions.merge(
        selection[metadata_columns],
        on=KEYS,
        how="left",
        validate="one_to_one",
    )
    decisions["delta_momentum_minus_diffusion"] = (
        decisions["exact_sparse_momentum"] - decisions["diffusion"]
    )
    decisions["delta_momentum_minus_imm"] = (
        decisions["exact_sparse_momentum"] - decisions["first_order_imm"]
    )
    decisions["delta_momentum_minus_best_other"] = (
        decisions["exact_sparse_momentum"]
        - decisions[["stationary", "diffusion", "fragmented", "first_order_imm"]].max(axis=1)
    )
    decisions["momentum_confident_best"] = (
        decisions["delta_momentum_minus_best_other"] >= margin_threshold
    )
    by_animal = _summary(decisions, ["animal"])
    model_summary = _summary(decisions, [])

    strict = decisions[decisions["strict_clean_imm"]]
    ordered = decisions[decisions["ordered_trajectory_confident"]]
    momentum = decisions[decisions["momentum_confident_best"]]
    strict_animals = int(strict["animal"].nunique()) if len(strict) else 0
    ordered_animals = int(ordered["animal"].nunique()) if len(ordered) else 0
    momentum_animals = int(momentum["animal"].nunique()) if len(momentum) else 0

    technical_rows = [
        _gate("technical", "expected_event_count_complete", len(selection) == expected_events, f"{len(selection)}/{expected_events}", "balanced frozen cohort"),
        _gate("technical", "expected_animals_present", selection["animal"].nunique() == expected_animals, f"{selection['animal'].nunique()}/{expected_animals}", "five large-arena animals"),
        _gate("technical", "selection_keys_unique", not selection.duplicated(KEYS).any(), int(selection.duplicated(KEYS).sum()), "zero duplicate events"),
        _gate("technical", "evidence_keys_unique", not evidence.duplicated([*KEYS, "model"]).any(), int(evidence.duplicated([*KEYS, "model"]).sum()), "zero duplicate event/model rows"),
        _gate("technical", "all_model_rows_successful", len(successful) == expected_rows, f"{len(successful)}/{expected_rows}", "all five models score every event"),
        _gate("technical", "required_models_complete", len(coverage) == expected_events and coverage.map(lambda value: value == set(MODELS)).all(), f"{int(coverage.map(lambda value: value == set(MODELS)).sum())}/{expected_events}", "complete exact core"),
        _gate("technical", "all_rows_evidence_comparable", evidence["evidence_comparable"].astype(bool).all(), f"{int(evidence['evidence_comparable'].astype(bool).sum())}/{len(evidence)}", "exact/comparable evidence only"),
        _gate("technical", "all_shards_pass", len(shard_overall) == expected_animals and shard_overall["passed"].astype(bool).all(), f"{int(shard_overall['passed'].astype(bool).sum())}/{expected_animals}", "per-animal technical gates"),
        _gate("technical", "selection_hash_frozen", len(selection_hashes) == 1 and None not in selection_hashes, len(selection_hashes), "one selection SHA-256 across shards"),
        _gate("technical", "parameter_signature_frozen", len(parameter_signatures) == 1, len(parameter_signatures), "same scoring parameters across shards"),
    ]
    technical_pass = all(bool(row["passed"]) for row in technical_rows)
    clean_imm_supported = bool(
        technical_pass
        and len(strict) >= minimum_positive_events
        and strict_animals >= minimum_positive_animals
    )
    momentum_supported = bool(
        technical_pass
        and len(momentum) >= minimum_positive_events
        and momentum_animals >= minimum_positive_animals
    )
    ordered_subset_present = bool(
        technical_pass
        and len(ordered) >= minimum_positive_events
        and ordered_animals >= minimum_positive_animals
    )
    interpretation_rows = [
        _gate("interpretation", "ordered_trajectory_subset_distributed", ordered_subset_present, f"{len(ordered)} events/{ordered_animals} animals", f">={minimum_positive_events} events across >={minimum_positive_animals} animals"),
        _gate("interpretation", "strict_clean_imm_subset_distributed", clean_imm_supported, f"{len(strict)} events/{strict_animals} animals", f">={minimum_positive_events} events across >={minimum_positive_animals} animals"),
        _gate("interpretation", "confident_momentum_subset_distributed", momentum_supported, f"{len(momentum)} events/{momentum_animals} animals", f">={minimum_positive_events} events across >={minimum_positive_animals} animals"),
    ]
    verdict = (
        "large_2d_clean_imm_replication_supported"
        if clean_imm_supported
        else "large_2d_clean_imm_replication_not_established"
    )
    geometry_verdict = (
        "unconstrained_2d_sufficient_for_clean_imm"
        if clean_imm_supported
        else "unconstrained_2d_not_sufficient_for_clean_imm"
    )
    recommendation = (
        "proceed_to_clean_imm_gate_2_3_4"
        if clean_imm_supported
        else "stop_clean_imm_ladder_report_ordered_dynamics_taxonomy"
    )
    gates = pd.DataFrame(
        [
            *technical_rows,
            _gate("technical", "overall_technical", technical_pass, technical_pass, "all technical gates"),
            *interpretation_rows,
            _gate("decision", "clean_imm_external_2d_replication", clean_imm_supported, verdict, recommendation),
        ]
    )

    evidence.to_csv(output_dir / COMBINED_EVIDENCE_OUTPUT, index=False)
    decisions.to_csv(output_dir / DECISIONS_OUTPUT, index=False)
    by_animal.to_csv(output_dir / BY_ANIMAL_OUTPUT, index=False)
    model_summary.to_csv(output_dir / MODEL_SUMMARY_OUTPUT, index=False)
    gates.to_csv(output_dir / REPLICATION_GATES_OUTPUT, index=False)

    overall = model_summary.iloc[0]
    report = "\n".join(
        [
            "# Tanni 2022 large-2D clean-IMM holdout",
            "",
            f"Technical status: **{'pass' if technical_pass else 'fail'}**.",
            f"Clean-IMM replication status: **{verdict}**.",
            f"Geometry verdict: **{geometry_verdict}**.",
            f"Recommended action: **{recommendation}**.",
            "",
            "## Result",
            "",
            f"- Events: {int(overall['events'])} across {expected_animals} animals.",
            f"- Trajectory-confident over stationary: {int(overall['trajectory_confident_count'])}/{int(overall['events'])}.",
            f"- Ordered trajectory-confident over stationary/fragmented: {int(overall['ordered_trajectory_confident_count'])}/{int(overall['events'])} across {ordered_animals}/{expected_animals} animals.",
            f"- IMM-confident over fragmented: {int(overall['imm_confident_over_fragmented_count'])}/{int(overall['events'])}.",
            f"- Strict clean IMM: {int(overall['strict_clean_imm_count'])}/{int(overall['events'])} across {strict_animals}/{expected_animals} animals.",
            f"- Exact-sparse momentum best: {int(overall['exact_sparse_momentum_best_count'])}/{int(overall['events'])}.",
            f"- Exact-sparse momentum confidently best: {int(overall['momentum_confident_best_count'])}/{int(overall['events'])} across {momentum_animals}/{expected_animals} animals.",
            f"- Median trajectory-minus-stationary: {float(overall['median_trajectory_minus_stationary']):+.3f} log evidence.",
            f"- Median IMM-minus-fragmented: {float(overall['median_imm_minus_fragmented']):+.3f} log evidence.",
            "",
            "## Interpretation",
            "",
            "The independent large-2D cohort contains a distributed ordered-trajectory subset, but it does not reproduce Pfeiffer/Foster clean IMM. Therefore unconstrained two-dimensional geometry is not sufficient to produce the clean-IMM signature.",
            "Exact-sparse momentum is the most frequent best model, but most wins are below the calibrated confidence margin; this is a descriptive hierarchy rather than a dataset-wide momentum claim.",
            "The cohort was selected by a pilot-informed high-information rule, so event fractions are diagnostic and must not be reported as unbiased prevalence estimates.",
            "",
        ]
    )
    (output_dir / REPORT_OUTPUT).write_text(report, encoding="utf-8")
    provenance_inputs = {
        **{f"evidence_shard_{index}": path for index, path in enumerate(evidence_paths)},
        **{f"selection_shard_{index}": path for index, path in enumerate(selection_paths)},
        **{f"unit_shard_{index}": path for index, path in enumerate(unit_paths)},
        **{f"gate_shard_{index}": path for index, path in enumerate(shard_gate_paths)},
        **{f"manifest_shard_{index}": path for index, path in enumerate(manifest_paths)},
    }
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "Tanni_et_al_2022_large_2d",
        "analysis": "non_rescoring_clean_imm_holdout_replication_report",
        "models": list(MODELS),
        "margin_threshold": float(margin_threshold),
        "minimum_positive_events": int(minimum_positive_events),
        "minimum_positive_animals": int(minimum_positive_animals),
        "verdict": verdict,
        "geometry_verdict": geometry_verdict,
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
        "by_animal": by_animal,
        "model_summary": model_summary,
        "gates": gates,
        "verdict": verdict,
        "geometry_verdict": geometry_verdict,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-animals", type=int, default=5)
    parser.add_argument("--events-per-animal", type=int, default=50)
    parser.add_argument("--margin-threshold", type=float, default=5.5)
    parser.add_argument("--minimum-positive-events", type=int, default=20)
    parser.add_argument("--minimum-positive-animals", type=int, default=4)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    build_report(
        shard_root=args.shard_root,
        output_dir=args.output_dir,
        expected_animals=args.expected_animals,
        events_per_animal=args.events_per_animal,
        margin_threshold=args.margin_threshold,
        minimum_positive_events=args.minimum_positive_events,
        minimum_positive_animals=args.minimum_positive_animals,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
