#!/usr/bin/env python3
"""Select state-space replay parameters from evidence and recovery sweeps."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd


PARAMETER_COLUMNS = [
    "state_space_diffusion_sigma_cm_sqrt_s",
    "state_space_momentum_sigma_cm_sqrt_s",
    "state_space_momentum_initial_sigma_cm_sqrt_s",
    "state_space_momentum_velocity_decay",
    "state_space_momentum_candidate_top_k",
]


def select_parameters(
    evidence: str | Path,
    recovery: str | Path,
    *,
    output: str | Path,
    min_momentum_recovery_accuracy: float = 0.5,
    min_overall_recovery_accuracy: float = 0.5,
    max_failures: int = 0,
) -> dict[str, pd.DataFrame]:
    evidence_frame = _load_table(evidence, "state_space_evidence_sweep_config_ranked.csv")
    recovery_frame = _load_table(recovery, "simulation_recovery_sweep_config_ranked.csv")
    evidence_frame = _prepare_evidence(evidence_frame)
    recovery_frame = _prepare_recovery(recovery_frame)

    decision = evidence_frame.merge(
        recovery_frame,
        on=PARAMETER_COLUMNS,
        how="outer",
        suffixes=("_evidence", "_recovery"),
        indicator="source_match",
    )
    decision["has_evidence"] = decision["source_match"].isin(["both", "left_only"])
    decision["has_recovery"] = decision["source_match"].isin(["both", "right_only"])
    decision["passes_recovery_gate"] = (
        decision["has_recovery"]
        & (decision.get("failures", pd.Series(0, index=decision.index)).fillna(0) <= max_failures)
        & (
            decision.get("momentum_recovery_accuracy", pd.Series(float("nan"), index=decision.index)).fillna(-1.0)
            >= min_momentum_recovery_accuracy
        )
        & (
            decision.get("overall_recovery_accuracy", pd.Series(float("nan"), index=decision.index)).fillna(-1.0)
            >= min_overall_recovery_accuracy
        )
    )
    decision["is_recommendable"] = decision["has_evidence"] & decision["passes_recovery_gate"]
    decision["recovery_gate"] = decision["passes_recovery_gate"].map({True: "pass", False: "fail"})
    decision.loc[~decision["has_recovery"], "recovery_gate"] = "missing-recovery"
    decision.loc[~decision["has_evidence"], "recovery_gate"] = "missing-evidence"

    ranked = _rank_decision_table(decision)
    ranked.insert(0, "recommendation_rank", range(1, len(ranked) + 1))
    candidates = ranked[ranked["is_recommendable"]].copy()
    recommendation = candidates.head(1).copy()
    if recommendation.empty and not ranked.empty:
        recommendation = ranked.head(1).copy()
        recommendation["recommendation_note"] = "No configuration passed the recovery gate; showing the best available row."
    elif not recommendation.empty:
        recommendation["recommendation_note"] = "Top configuration passing the recovery gate."

    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(out_dir / "state_space_parameter_decision_table.csv", index=False)
    candidates.to_csv(out_dir / "state_space_parameter_candidates.csv", index=False)
    recommendation.to_csv(out_dir / "state_space_parameter_recommendation.csv", index=False)
    _write_selected_parameter_files(recommendation, out_dir)
    _write_selection_manifest(
        evidence=evidence,
        recovery=recovery,
        output_dir=out_dir,
        decision=ranked,
        candidates=candidates,
        recommendation=recommendation,
        min_momentum_recovery_accuracy=min_momentum_recovery_accuracy,
        min_overall_recovery_accuracy=min_overall_recovery_accuracy,
        max_failures=max_failures,
    )
    return {
        "decision": ranked,
        "candidates": candidates,
        "recommendation": recommendation,
    }


def _load_table(path: str | Path, default_name: str) -> pd.DataFrame:
    path = Path(path)
    if path.is_dir():
        path = path / default_name
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist")
    frame = pd.read_csv(path)
    missing = set(PARAMETER_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing parameter columns: {sorted(missing)}")
    return frame


def _prepare_evidence(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    _normalize_parameter_columns(frame)
    if "events" in frame and "momentum_beats_diffusion_events" in frame:
        events = frame["events"].replace(0, pd.NA)
        frame["momentum_beats_diffusion_event_fraction"] = frame["momentum_beats_diffusion_events"] / events
    keep = PARAMETER_COLUMNS + [
        col
        for col in [
            "matrix_id",
            "events",
            "momentum_beats_diffusion_events",
            "momentum_beats_diffusion_event_fraction",
            "mean_momentum_minus_diffusion_log_evidence",
            "median_momentum_minus_diffusion_log_evidence",
            "mean_momentum_minus_imm_log_evidence",
            "median_momentum_minus_imm_log_evidence",
        ]
        if col in frame.columns
    ]
    return frame[keep].rename(columns={"matrix_id": "evidence_matrix_id"})


def _prepare_recovery(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    _normalize_parameter_columns(frame)
    keep = PARAMETER_COLUMNS + [
        col
        for col in [
            "matrix_id",
            "failures",
            "overall_recovery_accuracy",
            "momentum_recovery_accuracy",
            "diffusion_recovery_accuracy",
            "momentum_recovered_events",
            "overall_recovered_events",
            "overall_simulated_events",
            "momentum_most_common_best_model",
        ]
        if col in frame.columns
    ]
    return frame[keep].rename(columns={"matrix_id": "recovery_matrix_id"})


def _normalize_parameter_columns(frame: pd.DataFrame) -> None:
    for col in PARAMETER_COLUMNS:
        if col == "state_space_momentum_candidate_top_k":
            frame[col] = pd.to_numeric(frame[col], errors="raise").astype("int64")
        else:
            frame[col] = pd.to_numeric(frame[col], errors="raise").round(8)


def _rank_decision_table(frame: pd.DataFrame) -> pd.DataFrame:
    sortable = frame.copy()
    sort_columns = [
        "is_recommendable",
        "momentum_recovery_accuracy",
        "overall_recovery_accuracy",
        "diffusion_recovery_accuracy",
        "momentum_beats_diffusion_event_fraction",
        "median_momentum_minus_diffusion_log_evidence",
        "mean_momentum_minus_diffusion_log_evidence",
        "failures",
    ]
    ascending_by_column = {
        "is_recommendable": False,
        "momentum_recovery_accuracy": False,
        "overall_recovery_accuracy": False,
        "diffusion_recovery_accuracy": False,
        "momentum_beats_diffusion_event_fraction": False,
        "median_momentum_minus_diffusion_log_evidence": False,
        "mean_momentum_minus_diffusion_log_evidence": False,
        "failures": True,
    }
    present = [col for col in sort_columns if col in sortable.columns]
    for col in present:
        fill_value = float("inf") if ascending_by_column[col] else float("-inf")
        sortable[col] = sortable[col].fillna(fill_value)
    return sortable.sort_values(
        present,
        ascending=[ascending_by_column[col] for col in present],
        kind="stable",
    )


def _write_selected_parameter_files(recommendation: pd.DataFrame, out_dir: Path) -> None:
    yaml_path = out_dir / "state_space_selected_workflow_inputs.yml"
    cli_path = out_dir / "state_space_selected_cli_args.txt"
    if recommendation.empty:
        yaml_path.write_text("# No state-space parameter recommendation was available.\n", encoding="utf-8")
        cli_path.write_text("# No state-space parameter recommendation was available.\n", encoding="utf-8")
        return

    row = recommendation.iloc[0]
    values = {col: _format_parameter_value(row[col]) for col in PARAMETER_COLUMNS}
    yaml_lines = [
        "# Selected state-space workflow inputs generated by select_state_space_parameters.py\n",
        *[f"{col}: {value}\n" for col, value in values.items()],
    ]
    yaml_path.write_text("".join(yaml_lines), encoding="utf-8")

    args = [f"--{col.replace('_', '-')} {value}" for col, value in values.items()]
    continuation = " \\" + "\n  "
    cli_text = continuation.join(args) + "\n"
    cli_path.write_text(cli_text, encoding="utf-8")


def _write_selection_manifest(
    *,
    evidence: str | Path,
    recovery: str | Path,
    output_dir: Path,
    decision: pd.DataFrame,
    candidates: pd.DataFrame,
    recommendation: pd.DataFrame,
    min_momentum_recovery_accuracy: float,
    min_overall_recovery_accuracy: float,
    max_failures: int,
) -> None:
    selected_row = None if recommendation.empty else _json_ready(recommendation.iloc[0].to_dict())
    manifest = {
        "schema_version": 1,
        "input_paths": {
            "evidence": str(evidence),
            "recovery": str(recovery),
        },
        "recovery_gate": {
            "min_momentum_recovery_accuracy": min_momentum_recovery_accuracy,
            "min_overall_recovery_accuracy": min_overall_recovery_accuracy,
            "max_failures": max_failures,
        },
        "parameter_columns": PARAMETER_COLUMNS,
        "row_counts": {
            "decision_rows": int(len(decision)),
            "candidate_rows": int(len(candidates)),
            "recommendation_rows": int(len(recommendation)),
        },
        "selected_parameters": None if selected_row is None else {col: selected_row[col] for col in PARAMETER_COLUMNS},
        "recommendation": selected_row,
    }
    path = output_dir / "state_space_parameter_selection_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _json_ready(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if pd.isna(value):
        return None
    try:
        scalar = value.item()
    except AttributeError:
        scalar = value
    if isinstance(scalar, bool):
        return bool(scalar)
    if isinstance(scalar, float):
        if not math.isfinite(scalar):
            return None
        return float(scalar)
    if isinstance(scalar, int):
        return int(scalar)
    return scalar


def _format_parameter_value(value: object) -> str:
    if pd.isna(value):
        return "null"
    if isinstance(value, float):
        if value.is_integer():
            return f"{value:.1f}"
        return f"{value:.8g}"
    try:
        scalar = value.item()
    except AttributeError:
        scalar = value
    if isinstance(scalar, float):
        if scalar.is_integer():
            return f"{scalar:.1f}"
        return f"{scalar:.8g}"
    return str(scalar)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Join state-space evidence and simulation-recovery sweeps into a parameter decision table."
    )
    parser.add_argument("--evidence", required=True, help="Evidence sweep summary directory or ranked CSV")
    parser.add_argument("--recovery", required=True, help="Simulation-recovery sweep summary directory or ranked CSV")
    parser.add_argument("--output", default="results/state-space-parameter-selection")
    parser.add_argument("--min-momentum-recovery-accuracy", type=float, default=0.5)
    parser.add_argument("--min-overall-recovery-accuracy", type=float, default=0.5)
    parser.add_argument("--max-failures", type=int, default=0)
    args = parser.parse_args()

    tables = select_parameters(
        args.evidence,
        args.recovery,
        output=args.output,
        min_momentum_recovery_accuracy=args.min_momentum_recovery_accuracy,
        min_overall_recovery_accuracy=args.min_overall_recovery_accuracy,
        max_failures=args.max_failures,
    )
    print("Top parameter rows:")
    print(tables["decision"].head(10).to_string(index=False))
    print("\nRecommendation:")
    print(tables["recommendation"].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
