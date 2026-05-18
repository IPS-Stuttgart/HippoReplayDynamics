"""Load selected state-space benchmark parameters from sweep-selection outputs."""

from __future__ import annotations

import json
import shlex
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd

STATE_SPACE_PARAMETER_COLUMNS = (
    "state_space_stationary_sigma_cm",
    "state_space_diffusion_sigma_cm_sqrt_s",
    "state_space_max_step_sigma",
    "state_space_imm_mode_stickiness",
    "state_space_momentum_sigma_cm_sqrt_s",
    "state_space_momentum_initial_sigma_cm_sqrt_s",
    "state_space_momentum_velocity_decay",
    "state_space_momentum_candidate_top_k",
)

_SELECTION_FILENAMES = (
    "state_space_parameter_selection_manifest.json",
    "state_space_parameter_recommendation.csv",
    "state_space_selected_workflow_inputs.yml",
    "state_space_selected_workflow_inputs.yaml",
    "state_space_selected_cli_args.txt",
)


def load_state_space_parameter_selection(selection: str | Path) -> tuple[dict[str, object], Path]:
    """Return selected state-space parameters and the concrete source file used.

    ``selection`` may point at a parameter-selection output directory or directly at
    one of the files emitted by ``scripts/select_state_space_parameters.py``.
    Unknown columns are ignored by ``apply_state_space_parameter_selection`` so the
    loader remains forward-compatible with richer decision-table artifacts.
    """

    source = _resolve_selection_file(selection)
    suffix = source.suffix.lower()
    if suffix == ".json":
        parameters = _load_manifest(source)
    elif suffix == ".csv":
        parameters = _load_recommendation_csv(source)
    elif suffix in {".yml", ".yaml"}:
        parameters = _load_workflow_inputs(source)
    elif suffix == ".txt":
        parameters = _load_cli_args(source)
    else:
        raise ValueError(
            f"Unsupported state-space parameter selection file {source}. "
            "Expected .json, .csv, .yml, .yaml, or .txt."
        )
    return parameters, source


def apply_state_space_parameter_selection(args: SimpleNamespace | Any) -> dict[str, object]:
    """Override state-space benchmark args from a selection artifact if configured."""

    selection = getattr(args, "state_space_parameter_selection", None)
    if not selection:
        setattr(args, "state_space_parameter_selection_source", "")
        return {}

    parameters, source = load_state_space_parameter_selection(selection)
    applied: dict[str, object] = {}
    for column in STATE_SPACE_PARAMETER_COLUMNS:
        if column not in parameters:
            continue
        value = parameters[column]
        if _is_missing(value):
            continue
        coerced = _coerce_parameter(column, value)
        setattr(args, column, coerced)
        applied[column] = coerced

    if not applied:
        raise ValueError(f"{source} did not contain any recognized state-space parameters")
    setattr(args, "state_space_parameter_selection_source", str(source))
    return applied


def _resolve_selection_file(selection: str | Path) -> Path:
    path = Path(selection)
    if path.is_dir():
        for name in _SELECTION_FILENAMES:
            candidate = path / name
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            f"{path} does not contain a supported state-space parameter selection file: "
            + ", ".join(_SELECTION_FILENAMES)
        )
    if not path.exists():
        raise FileNotFoundError(f"State-space parameter selection file does not exist: {path}")
    return path


def _load_manifest(path: Path) -> dict[str, object]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    for key in ("selected_parameters", "recommendation"):
        value = manifest.get(key)
        if isinstance(value, dict) and value:
            return value
    if any(column in manifest for column in STATE_SPACE_PARAMETER_COLUMNS):
        return manifest
    return {}


def _load_recommendation_csv(path: Path) -> dict[str, object]:
    frame = pd.read_csv(path)
    if frame.empty:
        return {}
    return frame.iloc[0].to_dict()


def _load_workflow_inputs(path: Path) -> dict[str, object]:
    out: dict[str, object] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        if key:
            out[key] = value.strip()
    return out


def _load_cli_args(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8").replace("\\\n", " ")
    tokens = shlex.split(text, comments=True)
    out: dict[str, object] = {}
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if not token.startswith("--"):
            index += 1
            continue
        key = token[2:].replace("-", "_")
        if index + 1 >= len(tokens) or tokens[index + 1].startswith("--"):
            raise ValueError(f"Missing value for CLI argument {token} in {path}")
        out[key] = tokens[index + 1]
        index += 2
    return out


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "nan", "none", "null"}
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _coerce_parameter(column: str, value: object) -> int | float:
    if column == "state_space_momentum_candidate_top_k":
        return int(float(str(value).strip()))
    return float(str(value).strip())
