#!/usr/bin/env python3
"""Audit GitHub Actions parameter-sweep artifact completeness.

The evidence and recovery sweep workflows deliberately run many independent
matrix jobs.  A partial grid is useful for debugging, but it should not be used
as a final paper-level parameter ranking or marginalized evidence grid.  This
script joins the planned matrix with the artifacts that were actually produced
and writes an auditable completeness table.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd


@dataclass(frozen=True)
class SweepPreset:
    """Artifact naming convention for one sweep type."""

    artifact_prefix: str
    score_filenames: tuple[str, ...]
    required_summary_filenames: tuple[str, ...]


PRESETS: dict[str, SweepPreset] = {
    "state-space-evidence": SweepPreset(
        artifact_prefix="state-space-evidence-sweep-",
        score_filenames=("event_model_evidence.csv",),
        required_summary_filenames=("model_evidence_summary.csv",),
    ),
    "simulation-recovery": SweepPreset(
        artifact_prefix="simulation-recovery-sweep-",
        score_filenames=("simulation_recovery_event_scores.csv",),
        required_summary_filenames=(
            "simulation_recovery_summary.csv",
            "simulation_recovery_confusion_matrix.csv",
        ),
    ),
}


def audit_sweep_completeness(
    *,
    artifact_root: str | Path,
    output: str | Path,
    mode: str,
    fail_on_incomplete: bool = False,
) -> pd.DataFrame:
    """Write and return a matrix-cell completeness table."""

    preset = _preset(mode)
    root = Path(artifact_root)
    if not root.exists():
        raise FileNotFoundError(f"artifact root does not exist: {root}")

    planned = _load_plan_rows(root)
    score_paths = _paths_by_matrix_id(root, preset.score_filenames, preset.artifact_prefix)
    summary_paths = _paths_by_matrix_id(root, preset.required_summary_filenames, preset.artifact_prefix)
    all_matrix_ids = sorted(
        set(planned.get("matrix_id", pd.Series(dtype=str)).astype(str))
        | set(score_paths)
        | set(summary_paths)
    )
    if planned.empty:
        planned = pd.DataFrame({"matrix_id": all_matrix_ids, "planned": False})
    else:
        planned = planned.copy()
        planned["planned"] = True

    rows: list[dict[str, object]] = []
    for matrix_id in all_matrix_ids:
        plan_rows = planned[planned["matrix_id"].astype(str).eq(str(matrix_id))]
        base = _row_from_plan(plan_rows, matrix_id)
        cell_score_paths = score_paths.get(str(matrix_id), [])
        cell_summary_paths = summary_paths.get(str(matrix_id), [])
        score_counts = _score_counts(cell_score_paths)
        score_found = bool(cell_score_paths)
        required_summary_names = set(preset.required_summary_filenames)
        observed_summary_names = {path.name for path in cell_summary_paths}
        summary_found = required_summary_names.issubset(observed_summary_names)
        artifact_complete = score_found and summary_found and score_counts["n_event_rows"] > 0
        included_in_final_ranking = artifact_complete and score_counts["n_failure_rows"] == 0
        reasons = _completeness_reasons(
            score_found=score_found,
            summary_found=summary_found,
            n_event_rows=score_counts["n_event_rows"],
            n_failure_rows=score_counts["n_failure_rows"],
        )
        rows.append(
            {
                **base,
                "score_artifact_found": score_found,
                "summary_artifact_found": summary_found,
                "artifact_complete": artifact_complete,
                "grid_complete": artifact_complete,
                "included_in_final_ranking": included_in_final_ranking,
                "missing_matrix_cell": not artifact_complete,
                "n_missing_matrix_cells": int(not artifact_complete),
                "n_score_artifacts": len(cell_score_paths),
                "n_summary_artifacts": len(cell_summary_paths),
                "score_files": ";".join(str(path) for path in cell_score_paths),
                "summary_files": ";".join(str(path) for path in cell_summary_paths),
                **score_counts,
                "completeness_reason": ";".join(reasons) if reasons else "complete",
            }
        )

    table = pd.DataFrame(rows).sort_values(
        ["planned", "matrix_id"],
        ascending=[False, True],
    ).reset_index(drop=True)
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_path, index=False)
    _write_summary_json(table, out_path.with_suffix(".summary.json"), mode=mode, artifact_root=root)
    if fail_on_incomplete and bool(_bool_series(table["missing_matrix_cell"]).any()):
        raise SystemExit(f"Sweep is incomplete; see {out_path}")
    return table


def _preset(mode: str) -> SweepPreset:
    normalized = str(mode).strip().lower().replace("_", "-")
    aliases = {
        "evidence": "state-space-evidence",
        "state-space": "state-space-evidence",
        "recovery": "simulation-recovery",
        "simulation": "simulation-recovery",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in PRESETS:
        raise ValueError(f"mode must be one of {sorted(PRESETS)}")
    return PRESETS[normalized]


def _load_plan_rows(root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(root.rglob("matrix.csv")):
        try:
            frame = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
        if frame.empty:
            continue
        frame = frame.copy()
        if "matrix_id" not in frame and "id" in frame:
            frame = frame.rename(columns={"id": "matrix_id"})
        if "matrix_id" not in frame:
            continue
        frame["source_matrix_plan_file"] = str(path)
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["matrix_id"])
    planned = pd.concat(frames, ignore_index=True, sort=False)
    planned["matrix_id"] = planned["matrix_id"].astype(str)
    return planned.drop_duplicates("matrix_id", keep="first").reset_index(drop=True)


def _paths_by_matrix_id(root: Path, filenames: Sequence[str], artifact_prefix: str) -> dict[str, list[Path]]:
    out: dict[str, list[Path]] = {}
    for filename in filenames:
        for path in sorted(root.rglob(filename)):
            matrix_id = _matrix_id_from_csv(path) or _matrix_id_from_path(path, artifact_prefix)
            if not matrix_id:
                continue
            out.setdefault(str(matrix_id), []).append(path)
    return out


def _matrix_id_from_csv(path: Path) -> str | None:
    try:
        frame = pd.read_csv(path, nrows=1)
    except Exception:
        return None
    for column in ("matrix_id", "id"):
        if column in frame.columns and not frame.empty:
            value = frame[column].iloc[0]
            if pd.notna(value) and str(value):
                return str(value)
    return None


def _matrix_id_from_path(path: Path, artifact_prefix: str) -> str | None:
    for part in reversed(path.parts):
        if not part.startswith(artifact_prefix):
            continue
        suffix = part.removeprefix(artifact_prefix)
        if suffix.startswith("plan-") or suffix == "plan":
            continue
        return suffix or None
    return None


def _row_from_plan(plan_rows: pd.DataFrame, matrix_id: str) -> dict[str, object]:
    if plan_rows.empty:
        return {"matrix_id": matrix_id, "planned": False}
    row = plan_rows.iloc[0].to_dict()
    row["matrix_id"] = matrix_id
    row["planned"] = True
    return row


def _score_counts(paths: Sequence[Path]) -> dict[str, int]:
    n_event_rows = 0
    n_success_rows = 0
    n_failure_rows = 0
    for path in paths:
        try:
            frame = pd.read_csv(path)
        except Exception:
            n_failure_rows += 1
            continue
        n_event_rows += int(len(frame))
        if "status" in frame.columns:
            ok = frame["status"].astype(str).eq("success")
            n_success_rows += int(ok.sum())
            n_failure_rows += int((~ok).sum())
        else:
            n_success_rows += int(len(frame))
    return {
        "n_event_rows": n_event_rows,
        "n_success_rows": n_success_rows,
        "n_failure_rows": n_failure_rows,
    }


def _completeness_reasons(
    *,
    score_found: bool,
    summary_found: bool,
    n_event_rows: int,
    n_failure_rows: int,
) -> list[str]:
    reasons: list[str] = []
    if not score_found:
        reasons.append("missing-score-artifact")
    if not summary_found:
        reasons.append("missing-summary-artifact")
    if score_found and n_event_rows <= 0:
        reasons.append("empty-score-artifact")
    if n_failure_rows > 0:
        reasons.append("score-artifact-has-failures")
    return reasons


def _write_summary_json(table: pd.DataFrame, path: Path, *, mode: str, artifact_root: Path) -> None:
    planned = table[_bool_series(table["planned"])] if "planned" in table else table
    summary = {
        "schema_version": 1,
        "mode": mode,
        "artifact_root": str(artifact_root),
        "matrix_cells": int(len(table)),
        "planned_matrix_cells": int(len(planned)),
        "artifact_complete_cells": int(_bool_series(table["artifact_complete"]).sum()),
        "included_in_final_ranking_cells": int(
            _bool_series(table["included_in_final_ranking"]).sum()
        ),
        "missing_or_incomplete_cells": int(_bool_series(table["missing_matrix_cell"]).sum()),
    }
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _bool_value(value: object) -> bool:
    if isinstance(value, bool):
        return value
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    text = str(value).strip().lower()
    if text in {"1", "1.0", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "0.0", "false", "f", "no", "n", "", "nan", "none", "null", "off"}:
        return False
    try:
        numeric = float(text)
    except ValueError:
        return False
    return bool(math.isfinite(numeric) and numeric != 0.0)


def _bool_series(values: pd.Series) -> pd.Series:
    return values.map(_bool_value).astype(bool)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit parameter-sweep artifact completeness against the planned matrix.")
    parser.add_argument("--mode", choices=sorted(PRESETS), required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fail-on-incomplete", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    table = audit_sweep_completeness(
        artifact_root=args.artifact_root,
        output=args.output,
        mode=args.mode,
        fail_on_incomplete=args.fail_on_incomplete,
    )
    print(table.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
