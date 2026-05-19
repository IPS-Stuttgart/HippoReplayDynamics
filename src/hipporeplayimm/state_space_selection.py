"""Load selected state-space hyperparameters into decoder configurations."""

from __future__ import annotations

import json
import shlex
from dataclasses import replace
from pathlib import Path
from typing import Mapping

import pandas as pd

from .state_space import StateSpaceDecoderConfig


STATE_SPACE_SELECTION_DYNAMIC_COLUMNS = (
    "state_space_diffusion_sigma_cm_sqrt_s",
    "state_space_momentum_sigma_cm_sqrt_s",
    "state_space_momentum_initial_sigma_cm_sqrt_s",
    "state_space_momentum_velocity_decay",
    "state_space_momentum_candidate_top_k",
)

STATE_SPACE_CONFIG_COLUMNS = (
    "state_space_stationary_sigma_cm",
    "state_space_diffusion_sigma_cm_sqrt_s",
    "state_space_max_step_sigma",
    "state_space_imm_mode_stickiness",
    "state_space_momentum_sigma_cm_sqrt_s",
    "state_space_momentum_initial_sigma_cm_sqrt_s",
    "state_space_momentum_velocity_decay",
    "state_space_momentum_candidate_top_k",
    "state_space_momentum_predicted_candidate_top_k",
)

_COLUMN_TO_CONFIG_FIELD = {
    "state_space_stationary_sigma_cm": "stationary_sigma_cm",
    "state_space_diffusion_sigma_cm_sqrt_s": "diffusion_sigma_cm_sqrt_s",
    "state_space_max_step_sigma": "max_step_sigma",
    "state_space_imm_mode_stickiness": "imm_mode_stickiness",
    "state_space_momentum_sigma_cm_sqrt_s": "momentum_sigma_cm_sqrt_s",
    "state_space_momentum_initial_sigma_cm_sqrt_s": "momentum_initial_sigma_cm_sqrt_s",
    "state_space_momentum_velocity_decay": "momentum_velocity_decay",
    "state_space_momentum_candidate_top_k": "momentum_candidate_top_k",
    "state_space_momentum_predicted_candidate_top_k": "momentum_predicted_candidate_top_k",
}

_INT_COLUMNS = {
    "state_space_momentum_candidate_top_k",
    "state_space_momentum_predicted_candidate_top_k",
}


def load_state_space_decoder_config(
    selection: str | Path,
    *,
    base: StateSpaceDecoderConfig | None = None,
) -> StateSpaceDecoderConfig:
    """Load a selected sweep row into a :class:`StateSpaceDecoderConfig`.

    ``selection`` may point at the output directory created by
    ``scripts/select_state_space_parameters.py`` or at one of its artifacts:
    the JSON manifest, recommendation CSV, workflow-input YAML, or CLI-args
    text file.  The selected artifact contains only the swept dynamic
    parameters, so non-swept fields such as stationary noise and IMM stickiness
    are inherited from ``base``.
    """

    parameters = load_state_space_parameter_values(selection)
    config = StateSpaceDecoderConfig() if base is None else base
    updated = apply_state_space_parameter_values(config, parameters)
    if updated == config and not any(column in parameters for column in STATE_SPACE_CONFIG_COLUMNS):
        raise ValueError(f"{selection} does not contain recognized state-space parameters")
    return updated


def load_state_space_parameter_values(selection: str | Path) -> dict[str, object]:
    """Return selected state-space parameter values from a selection artifact."""

    path = Path(selection)
    if path.is_dir():
        return _load_selection_directory(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist")
    suffix = path.suffix.lower()
    if suffix == ".json":
        return _load_json_file(path)
    if suffix == ".csv":
        return _load_csv_file(path)
    if suffix in {".yml", ".yaml"}:
        return _load_simple_yaml_file(path)
    if suffix == ".txt":
        return _load_cli_args_file(path)
    raise ValueError(
        f"unsupported state-space parameter artifact {path}; expected directory, JSON, CSV, YAML, or TXT"
    )


def apply_state_space_parameter_values(
    config: StateSpaceDecoderConfig,
    values: Mapping[str, object],
) -> StateSpaceDecoderConfig:
    """Apply recognized state-space parameter columns to an existing config."""

    kwargs: dict[str, float | int] = {}
    for column, field_name in _COLUMN_TO_CONFIG_FIELD.items():
        value = values.get(column)
        if not _has_value(value):
            continue
        kwargs[field_name] = _to_int(value) if column in _INT_COLUMNS else _to_float(value)
    return replace(config, **kwargs) if kwargs else config


def _load_selection_directory(path: Path) -> dict[str, object]:
    for name in (
        "state_space_parameter_selection_manifest.json",
        "state_space_parameter_recommendation.csv",
        "state_space_selected_workflow_inputs.yml",
        "state_space_selected_cli_args.txt",
    ):
        candidate = path / name
        if candidate.exists():
            return load_state_space_parameter_values(candidate)
    raise FileNotFoundError(
        f"{path} does not contain a state-space parameter selection artifact"
    )


def _load_json_file(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    for key in ("selected_parameters", "recommendation"):
        value = payload.get(key)
        if isinstance(value, dict):
            return value
    return payload


def _load_csv_file(path: Path) -> dict[str, object]:
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"{path} does not contain any parameter rows")
    if "recommendation_rank" in frame.columns:
        frame = frame.sort_values("recommendation_rank", kind="stable")
    return frame.iloc[0].to_dict()


def _load_simple_yaml_file(path: Path) -> dict[str, object]:
    values: dict[str, object] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ":" not in stripped:
            raise ValueError(f"cannot parse state-space workflow input line: {line!r}")
        key, value = stripped.split(":", 1)
        values[key.strip()] = _parse_scalar(value.strip())
    return values


def _load_cli_args_file(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8")
    text = text.replace("\\\r\n", " ").replace("\\\n", " ")
    tokens = shlex.split(text)
    values: dict[str, object] = {}
    idx = 0
    while idx < len(tokens):
        token = tokens[idx]
        if not token.startswith("--"):
            idx += 1
            continue
        if "=" in token:
            key, value = token[2:].split("=", 1)
        else:
            if idx + 1 >= len(tokens):
                raise ValueError(f"missing value for {token} in {path}")
            key = token[2:]
            idx += 1
            value = tokens[idx]
        values[key.replace("-", "_")] = _parse_scalar(value)
        idx += 1
    return values


def _parse_scalar(value: object) -> object:
    if not _has_value(value):
        return None
    text = str(value).strip()
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def _has_value(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        text = value.strip().lower()
        return bool(text) and text not in {"none", "null", "nan", "na"}
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return True
    if isinstance(missing, bool):
        return not missing
    return True


def _to_float(value: object) -> float:
    return float(value)


def _to_int(value: object) -> int:
    return int(float(value))
