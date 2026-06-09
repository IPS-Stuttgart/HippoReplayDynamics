#!/usr/bin/env python3
"""Build the trajectory-family paper pack from completed benchmark artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:  # Allows both `python scripts/...py` and `pytest` imports.
    from scripts.aggregate_all_session_model_evidence import (
        DEFAULT_MOMENTUM_CONFIDENCE_THRESHOLD,
        DEFAULT_PAPER_REQUIRED_FULL_CORE_MODELS,
        DEFAULT_PAPER_EXACT_TRAJECTORY_MODELS,
        DEFAULT_RAT_BOOTSTRAP_RANDOM_SEED,
        DEFAULT_RAT_BOOTSTRAP_REPLICATES,
        exact_core_model_claim_decisions,
        exact_core_model_claim_summary,
        exact_trajectory_nontrajectory_margin_decisions,
        exact_trajectory_nontrajectory_margin_summary,
        leave_one_rat_out_exact_trajectory_nontrajectory_margin_summary,
        paired_momentum_diffusion_margin_decisions,
        paired_momentum_diffusion_margin_summary,
        rat_bootstrap_exact_trajectory_nontrajectory_margin_summary,
        rat_exact_trajectory_nontrajectory_margin_summary,
    )
except ModuleNotFoundError:  # pragma: no cover
    from aggregate_all_session_model_evidence import (  # type: ignore[no-redef]
        DEFAULT_MOMENTUM_CONFIDENCE_THRESHOLD,
        DEFAULT_PAPER_REQUIRED_FULL_CORE_MODELS,
        DEFAULT_PAPER_EXACT_TRAJECTORY_MODELS,
        DEFAULT_RAT_BOOTSTRAP_RANDOM_SEED,
        DEFAULT_RAT_BOOTSTRAP_REPLICATES,
        exact_core_model_claim_decisions,
        exact_core_model_claim_summary,
        exact_trajectory_nontrajectory_margin_decisions,
        exact_trajectory_nontrajectory_margin_summary,
        leave_one_rat_out_exact_trajectory_nontrajectory_margin_summary,
        paired_momentum_diffusion_margin_decisions,
        paired_momentum_diffusion_margin_summary,
        rat_bootstrap_exact_trajectory_nontrajectory_margin_summary,
        rat_exact_trajectory_nontrajectory_margin_summary,
    )

FULL_CORE_EVENT_TABLE_CANDIDATES = (
    "all_sessions_event_model_evidence.csv",
    "event_model_evidence.csv",
    "event_model_evidence_with_margins.csv",
)

WRONG_MAP_TABLES = (
    "wrong_map_control_gate_summary.csv",
    "wrong_map_family_evidence_attenuation_summary.csv",
    "rat_wrong_map_family_evidence_attenuation.csv",
    "leave_one_rat_out_wrong_map_family_evidence_attenuation.csv",
    "rat_bootstrap_wrong_map_family_evidence_attenuation.csv",
    "wrong_map_margin_difference_in_differences.csv",
)
EVENT_WINDOW_TABLES = (
    "event_window_control_gate_summary_v2.csv",
    "event_window_control_gate_summary.csv",
    "event_window_family_margin_summary.csv",
    "event_window_comparison_to_core.csv",
    "event_window_core_matched_attenuation.csv",
)
CELL_SPLIT_TABLES = (
    "cell_split_control_gate_summary.csv",
    "cell_split_heldout_family_margin_summary.csv",
    "rat_cell_split_heldout_summary.csv",
)
MATCHED_NULL_TABLES = (
    "matched_null_control_gate_summary.csv",
    "lightweight_matched_null_control_gate_summary.csv",
    "matched_null_family_margin_summary.csv",
    "matched_null_empirical_p_values.csv",
    "rat_matched_null_summary.csv",
    "leave_one_rat_out_matched_null_summary.csv",
    "rat_bootstrap_matched_null_summary.csv",
    "targeted_matched_null_session_diagnostics.csv",
    "targeted_matched_null_event_diagnostics.csv",
)

OUTPUT_FILES = (
    "paper_claim_manifest.json",
    "main_trajectory_family_summary.csv",
    "rat_trajectory_family_summary.csv",
    "leave_one_rat_out_trajectory_family_summary.csv",
    "bootstrap_trajectory_family_summary.csv",
    "exact_core_model_winner_summary.csv",
    "paired_momentum_diffusion_summary.csv",
    "control_stack_summary.csv",
    "matched_null_summary.csv",
    "cell_split_summary.csv",
    "event_window_summary.csv",
    "wrong_map_summary.csv",
    "figure_source_manifest.csv",
    "trajectory_family_paper_claim_summary.md",
)

CALIBRATED_PARAMETER_COLUMNS = (
    "time_bin_s",
    "spike_rate_scale",
    "emission_likelihood_temperature",
    "emission_negative_binomial_overdispersion",
    "sorted_spike_emission_model",
    "replay_gain_mode",
    "replay_gain_prior_count",
    "replay_gain_max_gain",
    "negative_binomial_dispersion",
    "state_space_effective_imm_mode_stickiness",
    "state_space_imm_switch_tau_s",
    "state_space_momentum_candidate_source",
    "state_space_momentum_candidate_mass_threshold",
    "state_space_momentum_candidate_min_k",
    "state_space_momentum_candidate_max_k",
    "state_space_momentum_predicted_candidate_top_k",
    "state_space_valid_occupancy_threshold_s",
    "diagnostic_state_space_stationary_sigma_cm",
    "diagnostic_state_space_diffusion_sigma_cm_sqrt_s",
    "diagnostic_state_space_max_step_sigma",
    "diagnostic_state_space_imm_mode_stickiness",
    "diagnostic_state_space_momentum_sigma_cm_sqrt_s",
    "diagnostic_state_space_momentum_initial_sigma_cm_sqrt_s",
    "diagnostic_state_space_momentum_velocity_decay",
    "diagnostic_state_space_momentum_velocity_decay_tau_s",
    "diagnostic_state_space_valid_occupancy_threshold_s",
    "diagnostic_state_space_observation_model",
)

PRIMARY_CLAIM = (
    "Exact trajectory-family dynamics dominate static/nontrajectory alternatives "
    "in full-core all-session evidence."
)
EXPLICIT_CAVEATS = (
    "Exact-sparse momentum is a recovered paired momentum-vs-diffusion signal, not the full-core winner.",
    "First-order IMM is interpreted as the leading exact core row when supported by exact-core summaries.",
    "Candidate-pruned momentum/IMM rows, when present in source artifacts, are lower-bound audit rows and are not mixed into exact headline rankings.",
    "Control artifacts are summarized from supplied artifacts; omitted control inputs are marked missing in control_stack_summary.csv.",
)


@dataclass(frozen=True)
class ArtifactSpec:
    label: str
    path: Path | None
    output_name: str
    table_names: tuple[str, ...]


def build_trajectory_family_paper_pack(
    *,
    full_core_artifact: str | Path,
    output: str | Path,
    wrong_map_artifact: str | Path | None = None,
    event_window_artifact: str | Path | None = None,
    cell_split_artifact: str | Path | None = None,
    matched_null_k10_artifact: str | Path | None = None,
    matched_null_k50_artifact: str | Path | None = None,
    confidence_threshold: float = DEFAULT_MOMENTUM_CONFIDENCE_THRESHOLD,
    n_bootstrap: int = DEFAULT_RAT_BOOTSTRAP_REPLICATES,
    random_seed: int = DEFAULT_RAT_BOOTSTRAP_RANDOM_SEED,
    code_commit: str | None = None,
    require_controls: bool = False,
) -> dict[str, object]:
    """Write the paper pack and return the manifest."""

    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)

    full_core_path = Path(full_core_artifact)
    full_core_csv = _resolve_csv(full_core_path, FULL_CORE_EVENT_TABLE_CANDIDATES)
    scores = pd.read_csv(full_core_csv)
    _validate_event_scores(scores, full_core_csv)

    trajectory_decisions = exact_trajectory_nontrajectory_margin_decisions(
        scores,
        margin_threshold=float(confidence_threshold),
    )
    main_summary = exact_trajectory_nontrajectory_margin_summary(trajectory_decisions)
    rat_summary = rat_exact_trajectory_nontrajectory_margin_summary(trajectory_decisions)
    leave_one_summary = leave_one_rat_out_exact_trajectory_nontrajectory_margin_summary(trajectory_decisions)
    bootstrap_summary = rat_bootstrap_exact_trajectory_nontrajectory_margin_summary(
        trajectory_decisions,
        n_bootstrap=int(n_bootstrap),
        random_seed=int(random_seed),
    )

    core_decisions = exact_core_model_claim_decisions(scores, margin_threshold=float(confidence_threshold))
    core_summary = exact_core_model_claim_summary(core_decisions)
    paired_decisions = paired_momentum_diffusion_margin_decisions(scores, margin_threshold=float(confidence_threshold))
    paired_summary = paired_momentum_diffusion_margin_summary(paired_decisions)

    _write_csv(out_dir, "main_trajectory_family_summary.csv", main_summary)
    _write_csv(out_dir, "rat_trajectory_family_summary.csv", rat_summary)
    _write_csv(out_dir, "leave_one_rat_out_trajectory_family_summary.csv", leave_one_summary)
    _write_csv(out_dir, "bootstrap_trajectory_family_summary.csv", bootstrap_summary)
    _write_csv(out_dir, "exact_core_model_winner_summary.csv", core_summary)
    _write_csv(out_dir, "paired_momentum_diffusion_summary.csv", paired_summary)

    control_specs = _control_specs(
        wrong_map_artifact=wrong_map_artifact,
        event_window_artifact=event_window_artifact,
        cell_split_artifact=cell_split_artifact,
        matched_null_k10_artifact=matched_null_k10_artifact,
        matched_null_k50_artifact=matched_null_k50_artifact,
    )
    missing_required = [
        spec.label
        for spec in control_specs
        if require_controls and (spec.path is None or not spec.path.exists())
    ]
    if missing_required:
        raise FileNotFoundError(f"Required control artifact(s) missing: {', '.join(missing_required)}")

    control_stack, control_tables = _collect_control_tables(control_specs)
    _write_csv(out_dir, "control_stack_summary.csv", control_stack)
    _write_csv(out_dir, "wrong_map_summary.csv", control_tables["wrong_map_summary.csv"])
    _write_csv(out_dir, "event_window_summary.csv", control_tables["event_window_summary.csv"])
    _write_csv(out_dir, "cell_split_summary.csv", control_tables["cell_split_summary.csv"])
    _write_csv(out_dir, "matched_null_summary.csv", control_tables["matched_null_summary.csv"])

    manifest = _build_manifest(
        full_core_artifact=full_core_path,
        full_core_event_table=full_core_csv,
        control_specs=control_specs,
        scores=scores,
        trajectory_decisions=trajectory_decisions,
        main_summary=main_summary,
        confidence_threshold=float(confidence_threshold),
        n_bootstrap=int(n_bootstrap),
        random_seed=int(random_seed),
        code_commit=code_commit or _current_commit(),
    )

    markdown_summary = _render_claim_summary(
        manifest=manifest,
        main_summary=main_summary,
        rat_summary=rat_summary,
        leave_one_summary=leave_one_summary,
        bootstrap_summary=bootstrap_summary,
        core_summary=core_summary,
        paired_summary=paired_summary,
        control_stack=control_stack,
    )
    (out_dir / "trajectory_family_paper_claim_summary.md").write_text(markdown_summary, encoding="utf-8")

    figure_manifest = _build_figure_source_manifest(
        out_dir=out_dir,
        full_core_artifact=full_core_path,
        full_core_event_table=full_core_csv,
        control_specs=control_specs,
    )
    _write_csv(out_dir, "figure_source_manifest.csv", figure_manifest)

    manifest["outputs"] = {name: str(out_dir / name) for name in OUTPUT_FILES}
    manifest["output_digest"] = _directory_digest(out_dir, exclude_names={"paper_claim_manifest.json"})
    (out_dir / "paper_claim_manifest.json").write_text(
        json.dumps(_json_ready(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _control_specs(
    *,
    wrong_map_artifact: str | Path | None,
    event_window_artifact: str | Path | None,
    cell_split_artifact: str | Path | None,
    matched_null_k10_artifact: str | Path | None,
    matched_null_k50_artifact: str | Path | None,
) -> list[ArtifactSpec]:
    return [
        ArtifactSpec("wrong_map_control", _optional_path(wrong_map_artifact), "wrong_map_summary.csv", WRONG_MAP_TABLES),
        ArtifactSpec(
            "event_window_control",
            _optional_path(event_window_artifact),
            "event_window_summary.csv",
            EVENT_WINDOW_TABLES,
        ),
        ArtifactSpec("cell_split_control", _optional_path(cell_split_artifact), "cell_split_summary.csv", CELL_SPLIT_TABLES),
        ArtifactSpec(
            "matched_null_k10_control",
            _optional_path(matched_null_k10_artifact),
            "matched_null_summary.csv",
            MATCHED_NULL_TABLES,
        ),
        ArtifactSpec(
            "matched_null_k50_control",
            _optional_path(matched_null_k50_artifact),
            "matched_null_summary.csv",
            MATCHED_NULL_TABLES,
        ),
    ]


def _optional_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    return Path(value)


def _resolve_csv(root: Path, candidates: Sequence[str]) -> Path:
    if root.is_file():
        if root.suffix.lower() != ".csv":
            raise FileNotFoundError(f"Expected a CSV file, got: {root}")
        return root
    if not root.exists():
        raise FileNotFoundError(f"Artifact path does not exist: {root}")
    for name in candidates:
        direct = root / name
        if direct.exists():
            return direct
    for name in candidates:
        matches = sorted(path for path in root.rglob(name) if path.is_file())
        if matches:
            return matches[0]
    raise FileNotFoundError(f"No known CSV table found in {root}; tried: {', '.join(candidates)}")


def _validate_event_scores(scores: pd.DataFrame, source: Path) -> None:
    required = {"session", "event_index", "model", "log_evidence"}
    missing = sorted(required - set(scores.columns))
    if missing:
        raise KeyError(f"{source} is missing required column(s): {', '.join(missing)}")


def _write_csv(out_dir: Path, name: str, frame: pd.DataFrame) -> None:
    frame.to_csv(out_dir / name, index=False)


def _collect_control_tables(specs: Sequence[ArtifactSpec]) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    stack_rows: list[dict[str, object]] = []
    by_output: dict[str, list[pd.DataFrame]] = {
        "wrong_map_summary.csv": [],
        "event_window_summary.csv": [],
        "cell_split_summary.csv": [],
        "matched_null_summary.csv": [],
    }
    for spec in specs:
        found = _load_artifact_tables(spec)
        stack_rows.append(_control_stack_row(spec, found))
        if found:
            for source_table, frame in found:
                tagged = _tag_control_table(spec, source_table, frame)
                by_output[spec.output_name].append(tagged)
        else:
            by_output[spec.output_name].append(_missing_control_table(spec))

    stack = pd.DataFrame(stack_rows)
    outputs = {
        name: (pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame())
        for name, frames in by_output.items()
    }
    return stack, outputs


def _load_artifact_tables(spec: ArtifactSpec) -> list[tuple[str, pd.DataFrame]]:
    if spec.path is None or not spec.path.exists():
        return []
    tables: list[tuple[str, pd.DataFrame]] = []
    if spec.path.is_file():
        if spec.path.suffix.lower() == ".csv":
            tables.append((spec.path.name, pd.read_csv(spec.path)))
        return tables
    for name in spec.table_names:
        matches = sorted(path for path in spec.path.rglob(name) if path.is_file())
        for path in matches:
            tables.append((path.name, pd.read_csv(path)))
    return tables


def _control_stack_row(spec: ArtifactSpec, found: Sequence[tuple[str, pd.DataFrame]]) -> dict[str, object]:
    path_text = "" if spec.path is None else str(spec.path)
    if spec.path is None:
        status = "missing_path"
    elif not spec.path.exists():
        status = "missing_artifact"
    elif not found:
        status = "no_known_tables"
    else:
        status = "ok"

    gate_tables = [(name, frame) for name, frame in found if {"gate", "passed"}.issubset(frame.columns)]
    gates_total = 0
    gates_passed = 0
    overall_gate = ""
    overall_passed: object = ""
    for source_table, frame in gate_tables:
        passed = frame["passed"].map(_boolish)
        valid = passed.dropna()
        gates_total += int(len(valid))
        gates_passed += int(valid.sum()) if not valid.empty else 0
        if "gate" in frame:
            overall = frame[frame["gate"].fillna("").astype(str).str.contains("overall", case=False)]
            if not overall.empty and overall_gate == "":
                overall_gate = f"{source_table}:{overall.iloc[0]['gate']}"
                overall_passed = _boolish(overall.iloc[0]["passed"])
    if overall_passed == "" and gates_total:
        overall_passed = bool(gates_passed == gates_total)

    return {
        "artifact_label": spec.label,
        "artifact_run_id": _artifact_run_id(spec.path),
        "artifact_name": "" if spec.path is None else spec.path.name,
        "artifact_path": path_text,
        "status": status,
        "known_tables_found": int(len(found)),
        "source_tables": " ".join(name for name, _ in found),
        "gate_tables_found": int(len(gate_tables)),
        "gates_passed": gates_passed,
        "gates_total": gates_total,
        "overall_gate": overall_gate,
        "overall_passed": overall_passed,
    }


def _tag_control_table(spec: ArtifactSpec, source_table: str, frame: pd.DataFrame) -> pd.DataFrame:
    tagged = frame.copy()
    tagged.insert(0, "source_table", source_table)
    tagged.insert(0, "artifact_status", "ok")
    tagged.insert(0, "artifact_path", "" if spec.path is None else str(spec.path))
    tagged.insert(0, "artifact_run_id", _artifact_run_id(spec.path))
    tagged.insert(0, "artifact_label", spec.label)
    return tagged


def _missing_control_table(spec: ArtifactSpec) -> pd.DataFrame:
    if spec.path is None:
        status = "missing_path"
    elif not spec.path.exists():
        status = "missing_artifact"
    else:
        status = "no_known_tables"
    return pd.DataFrame(
        [
            {
                "artifact_label": spec.label,
                "artifact_run_id": _artifact_run_id(spec.path),
                "artifact_path": "" if spec.path is None else str(spec.path),
                "artifact_status": status,
                "source_table": "",
            }
        ]
    )


def _build_manifest(
    *,
    full_core_artifact: Path,
    full_core_event_table: Path,
    control_specs: Sequence[ArtifactSpec],
    scores: pd.DataFrame,
    trajectory_decisions: pd.DataFrame,
    main_summary: pd.DataFrame,
    confidence_threshold: float,
    n_bootstrap: int,
    random_seed: int,
    code_commit: str,
) -> dict[str, object]:
    artifacts = [ArtifactSpec("full_core_model_evidence", full_core_artifact, "", tuple())]
    artifacts.extend(control_specs)
    event_count = _event_count(trajectory_decisions)
    if event_count == 0 and not main_summary.empty and "events" in main_summary:
        event_count = int(main_summary.iloc[0]["events"])

    return {
        "schema_version": 1,
        "code_commit": code_commit,
        "artifact_run_ids": {artifact.label: _artifact_run_id(artifact.path) for artifact in artifacts},
        "artifact_names": {
            artifact.label: "" if artifact.path is None else artifact.path.name for artifact in artifacts
        },
        "artifact_paths": {
            artifact.label: "" if artifact.path is None else str(artifact.path) for artifact in artifacts
        },
        "artifact_digests": {artifact.label: _artifact_digest(artifact.path) for artifact in artifacts},
        "full_core_event_table": str(full_core_event_table),
        "model_list": sorted(scores["model"].dropna().astype(str).unique().tolist()),
        "required_exact_core_models": list(DEFAULT_PAPER_REQUIRED_FULL_CORE_MODELS),
        "trajectory_family_models": list(DEFAULT_PAPER_EXACT_TRAJECTORY_MODELS),
        "calibrated_row_parameters": _calibrated_row_parameters(scores),
        "confidence_threshold": float(confidence_threshold),
        "bootstrap": {"replicates": int(n_bootstrap), "random_seed": int(random_seed)},
        "event_count": event_count,
        "rat_session_coverage": _rat_session_coverage(scores),
        "primary_claim": PRIMARY_CLAIM,
        "explicit_caveats": list(EXPLICIT_CAVEATS),
    }


def _render_claim_summary(
    *,
    manifest: dict[str, object],
    main_summary: pd.DataFrame,
    rat_summary: pd.DataFrame,
    leave_one_summary: pd.DataFrame,
    bootstrap_summary: pd.DataFrame,
    core_summary: pd.DataFrame,
    paired_summary: pd.DataFrame,
    control_stack: pd.DataFrame,
) -> str:
    lines = [
        "# Trajectory-family paper claim summary",
        "",
        f"Primary claim: {PRIMARY_CLAIM}",
        "",
        "## Full-core trajectory-family gate",
        "",
    ]
    if main_summary.empty:
        lines.append("No complete trajectory-family summary rows were produced.")
    else:
        row = main_summary.iloc[0]
        lines.append(
            "- "
            + f"{_fmt_int(row, 'trajectory_raw_wins')}/{_fmt_int(row, 'events')} raw trajectory-family wins; "
            + f"{_fmt_int(row, 'trajectory_confident_claims')}/{_fmt_int(row, 'events')} confident trajectory claims; "
            + f"{_fmt_int(row, 'nontrajectory_confident_claims')}/{_fmt_int(row, 'events')} confident nontrajectory claims."
        )
        lines.append(
            "- "
            + "Mean/median trajectory-minus-nontrajectory log-evidence margin: "
            + f"{_fmt_float(row, 'mean_trajectory_minus_nontrajectory_log_evidence')} / "
            + f"{_fmt_float(row, 'median_trajectory_minus_nontrajectory_log_evidence')}."
        )
    lines.extend(["", "## Rat and robustness summaries", ""])
    if not rat_summary.empty:
        lines.append(
            "- Rat coverage: "
            + f"{len(rat_summary)} rats; weakest rat trajectory confident-claim fraction "
            + f"{float(rat_summary['trajectory_confident_claim_fraction'].min()):.6g}."
        )
    if not leave_one_summary.empty:
        lines.append(
            "- Leave-one-rat-out subsets retain minimum median margin "
            + f"{float(leave_one_summary['median_trajectory_minus_nontrajectory_log_evidence'].min()):.6g}."
        )
    if not bootstrap_summary.empty:
        row = bootstrap_summary.iloc[0]
        lines.append(
            "- Rat bootstrap lower bounds: claim fraction "
            + f"{_fmt_float(row, 'positive_claim_fraction_ci95_low')}, mean margin "
            + f"{_fmt_float(row, 'mean_delta_ci95_low')}, median margin "
            + f"{_fmt_float(row, 'median_delta_ci95_low')}."
        )
    lines.extend(["", "## Exact-core and momentum checks", ""])
    if not core_summary.empty:
        leader = core_summary.sort_values(
            ["raw_best_events", "confident_claims", "mean_winning_margin_when_best"],
            ascending=[False, False, False],
        ).iloc[0]
        lines.append(
            "- Leading exact core row: "
            + f"`{leader['model']}` with {_fmt_int(leader, 'raw_best_events')} raw best events "
            + f"and {_fmt_int(leader, 'confident_claims')} confident exact-core claims."
        )
    if not paired_summary.empty:
        row = paired_summary.iloc[0]
        lines.append(
            "- Paired exact-sparse momentum vs diffusion: "
            + f"{_fmt_int(row, 'positive_raw_wins')}/{_fmt_int(row, 'events')} momentum raw wins; "
            + f"median margin {_fmt_float(row, 'median_positive_minus_reference_log_evidence')}."
        )
    lines.extend(["", "## Control stack", ""])
    if control_stack.empty:
        lines.append("No control stack rows were produced.")
    else:
        for _, row in control_stack.iterrows():
            lines.append(
                "- "
                + f"{row['artifact_label']}: {row['status']}; "
                + f"{_fmt_int(row, 'known_tables_found')} known tables; "
                + f"{_fmt_int(row, 'gates_passed')}/{_fmt_int(row, 'gates_total')} gates passed."
            )
    lines.extend(["", "## Caveats", ""])
    for caveat in manifest["explicit_caveats"]:
        lines.append(f"- {caveat}")
    lines.append("")
    return "\n".join(lines)


def _build_figure_source_manifest(
    *,
    out_dir: Path,
    full_core_artifact: Path,
    full_core_event_table: Path,
    control_specs: Sequence[ArtifactSpec],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = [
        _source_manifest_row("input", "full_core_model_evidence", full_core_event_table, _artifact_run_id(full_core_artifact))
    ]
    for spec in control_specs:
        if spec.path is None:
            rows.append(_source_manifest_missing_input_row(spec, "missing_path"))
        elif not spec.path.exists():
            rows.append(_source_manifest_missing_input_row(spec, "missing_artifact"))
        else:
            for table_name in spec.table_names:
                if spec.path.is_file():
                    paths = [spec.path] if spec.path.name == table_name else []
                else:
                    paths = sorted(path for path in spec.path.rglob(table_name) if path.is_file())
                for path in paths:
                    rows.append(_source_manifest_row("input", spec.label, path, _artifact_run_id(spec.path)))
    for name in OUTPUT_FILES:
        if name == "figure_source_manifest.csv":
            rows.append(
                {
                    "kind": "output",
                    "label": "figure_source_manifest",
                    "artifact_run_id": "",
                    "path": str(out_dir / name),
                    "status": "self_manifest",
                    "sha256": "",
                    "bytes": "",
                    "rows": "",
                    "columns": "",
                }
            )
            continue
        if name == "paper_claim_manifest.json":
            rows.append(
                {
                    "kind": "output",
                    "label": "paper_claim_manifest",
                    "artifact_run_id": "",
                    "path": str(out_dir / name),
                    "status": "metadata_manifest",
                    "sha256": "",
                    "bytes": "",
                    "rows": "",
                    "columns": "",
                }
            )
            continue
        path = out_dir / name
        if path.exists():
            rows.append(_source_manifest_row("output", Path(name).stem, path, ""))
    return pd.DataFrame(rows)


def _source_manifest_missing_input_row(spec: ArtifactSpec, status: str) -> dict[str, object]:
    return {
        "kind": "input",
        "label": spec.label,
        "artifact_run_id": _artifact_run_id(spec.path),
        "path": "" if spec.path is None else str(spec.path),
        "status": status,
        "sha256": "",
        "bytes": "",
        "rows": "",
        "columns": "",
    }


def _source_manifest_row(kind: str, label: str, path: Path, run_id: str) -> dict[str, object]:
    rows, columns = _csv_shape(path) if path.suffix.lower() == ".csv" else ("", "")
    return {
        "kind": kind,
        "label": label,
        "artifact_run_id": run_id,
        "path": str(path),
        "status": "ok",
        "sha256": _file_digest(path),
        "bytes": path.stat().st_size,
        "rows": rows,
        "columns": columns,
    }


def _csv_shape(path: Path) -> tuple[int | str, int | str]:
    try:
        frame = pd.read_csv(path)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError):
        return "", ""
    return int(len(frame)), int(len(frame.columns))


def _artifact_run_id(path: Path | None) -> str:
    if path is None:
        return ""
    match = re.search(r"(\d{8,})", path.name)
    return "" if match is None else match.group(1)


def _artifact_digest(path: Path | None) -> str:
    if path is None or not path.exists():
        return ""
    if path.is_file():
        return _file_digest(path)
    return _directory_digest(path)


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _directory_digest(path: Path, *, exclude_names: set[str] | None = None) -> str:
    exclude_names = set() if exclude_names is None else exclude_names
    digest = hashlib.sha256()
    for item in sorted(p for p in path.rglob("*") if p.is_file() and p.name not in exclude_names):
        digest.update(str(item.relative_to(path)).replace("\\", "/").encode("utf-8"))
        digest.update(b"\0")
        digest.update(_file_digest(item).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _event_count(frame: pd.DataFrame) -> int:
    if frame.empty or not {"session", "event_index"}.issubset(frame.columns):
        return 0
    return int(frame[["session", "event_index"]].drop_duplicates().shape[0])


def _rat_session_coverage(scores: pd.DataFrame) -> dict[str, object]:
    events = scores[["session", "event_index"]].drop_duplicates().copy()
    events["rat"] = events["session"].map(_rat_from_session)
    session_event_counts = events.groupby("session").size().sort_index()
    rat_event_counts = events.groupby("rat").size().sort_index()
    return {
        "rats": sorted(events["rat"].dropna().astype(str).unique().tolist()),
        "rat_count": int(events["rat"].nunique()),
        "sessions": sorted(events["session"].dropna().astype(str).unique().tolist()),
        "session_count": int(events["session"].nunique()),
        "events_per_rat": {str(key): int(value) for key, value in rat_event_counts.items()},
        "events_per_session": {str(key): int(value) for key, value in session_event_counts.items()},
    }


def _rat_from_session(session: object) -> str:
    return str(session).replace("\\", "/").split("/", 1)[0]


def _calibrated_row_parameters(scores: pd.DataFrame) -> dict[str, object]:
    parameters: dict[str, object] = {}
    for column in CALIBRATED_PARAMETER_COLUMNS:
        if column in scores:
            parameters[column] = _column_value_summary(scores[column])
    return parameters


def _column_value_summary(series: pd.Series) -> object:
    values = series.dropna()
    values = values[values.astype(str) != ""]
    uniques = sorted({str(value) for value in values.unique()})
    if not uniques:
        return None
    if len(uniques) == 1:
        return _json_scalar(uniques[0])
    return {"unique_count": len(uniques), "unique_values": [_json_scalar(value) for value in uniques[:50]]}


def _json_scalar(value: object) -> object:
    if isinstance(value, str):
        text = value.strip()
        if text == "":
            return None
        lowered = text.lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        try:
            if re.fullmatch(r"[-+]?\d+", text):
                return int(text)
            return float(text)
        except ValueError:
            return text
    return _json_ready(value)


def _boolish(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return None
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "pass", "passed"}:
        return True
    if text in {"false", "0", "no", "n", "fail", "failed"}:
        return False
    return None


def _fmt_int(row: pd.Series, column: str) -> str:
    if column not in row or pd.isna(row[column]):
        return "NA"
    return str(int(float(row[column])))


def _fmt_float(row: pd.Series, column: str) -> str:
    if column not in row or pd.isna(row[column]):
        return "NA"
    return f"{float(row[column]):.6g}"


def _current_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):  # pragma: no cover
        return ""


def _json_ready(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    try:
        scalar = value.item()
    except AttributeError:
        scalar = value
    if isinstance(scalar, float):
        return None if not math.isfinite(scalar) else float(scalar)
    if isinstance(scalar, (int, bool, str)) or scalar is None:
        return scalar
    return str(scalar)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a trajectory-family paper pack from benchmark artifacts.")
    parser.add_argument("--full-core-artifact", required=True, help="Artifact dir or event evidence CSV for run 27011374643.")
    parser.add_argument("--wrong-map-artifact", help="Wrong-map control artifact directory.")
    parser.add_argument("--event-window-artifact", help="Event-window artifact directory, e.g. run 26884355750.")
    parser.add_argument("--cell-split-artifact", help="Cell-split held-out artifact directory, e.g. run 26965909403.")
    parser.add_argument("--matched-null-k10-artifact", help="K=10 matched-null artifact directory, e.g. run 26886723196.")
    parser.add_argument("--matched-null-k50-artifact", help="K=50 lightweight matched-null artifact directory, e.g. run 27060148887.")
    parser.add_argument("--output", required=True, help="Output directory for the paper pack.")
    parser.add_argument("--confidence-threshold", type=float, default=DEFAULT_MOMENTUM_CONFIDENCE_THRESHOLD)
    parser.add_argument("--n-bootstrap", type=int, default=DEFAULT_RAT_BOOTSTRAP_REPLICATES)
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RAT_BOOTSTRAP_RANDOM_SEED)
    parser.add_argument("--code-commit", help="Override commit recorded in the manifest.")
    parser.add_argument("--require-controls", action="store_true", help="Fail if any control artifact path is missing.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = build_trajectory_family_paper_pack(
        full_core_artifact=args.full_core_artifact,
        output=args.output,
        wrong_map_artifact=args.wrong_map_artifact,
        event_window_artifact=args.event_window_artifact,
        cell_split_artifact=args.cell_split_artifact,
        matched_null_k10_artifact=args.matched_null_k10_artifact,
        matched_null_k50_artifact=args.matched_null_k50_artifact,
        confidence_threshold=args.confidence_threshold,
        n_bootstrap=args.n_bootstrap,
        random_seed=args.random_seed,
        code_commit=args.code_commit,
        require_controls=args.require_controls,
    )
    print(json.dumps(_json_ready(manifest), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
