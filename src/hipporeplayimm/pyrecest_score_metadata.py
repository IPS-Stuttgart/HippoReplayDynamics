from __future__ import annotations


import numpy as np
import pandas as pd

PYRECEST_INT_COLUMNS = {
    "pyrecest_particles": ("pyrecest_particles", "diagnostic_pyrecest_particles"),
}

PYRECEST_FLOAT_COLUMNS = {
    "pyrecest_alpha": ("pyrecest_alpha", "diagnostic_pyrecest_alpha"),
    "pyrecest_beta": ("pyrecest_beta", "diagnostic_pyrecest_beta"),
    "pyrecest_process_noise_sigma_cm_s": ("pyrecest_process_noise_sigma_cm_s", "diagnostic_pyrecest_process_noise_sigma_cm_s"),
    "pyrecest_position_jump_sigma_cm": ("pyrecest_position_jump_sigma_cm", "diagnostic_pyrecest_position_jump_sigma_cm"),
    "pyrecest_jump_probability": ("pyrecest_jump_probability", "diagnostic_pyrecest_jump_probability"),
    "pyrecest_goal_reset_probability": ("pyrecest_goal_reset_probability", "diagnostic_pyrecest_goal_reset_probability"),
    "pyrecest_position_proposal_probability": ("pyrecest_position_proposal_probability", "diagnostic_pyrecest_position_proposal_probability"),
    "pyrecest_initial_velocity_sigma_cm_s": ("pyrecest_initial_velocity_sigma_cm_s", "diagnostic_pyrecest_initial_velocity_sigma_cm_s"),
    "pyrecest_imm_mode_stickiness": ("pyrecest_imm_mode_stickiness", "diagnostic_pyrecest_imm_mode_stickiness"),
    "pyrecest_imm_stationary_velocity_decay": ("pyrecest_imm_stationary_velocity_decay", "diagnostic_pyrecest_imm_stationary_velocity_decay"),
    "pyrecest_imm_diffusion_velocity_decay": ("pyrecest_imm_diffusion_velocity_decay", "diagnostic_pyrecest_imm_diffusion_velocity_decay"),
    "pyrecest_imm_momentum_velocity_decay": ("pyrecest_imm_momentum_velocity_decay", "diagnostic_pyrecest_imm_momentum_velocity_decay"),
    "pyrecest_imm_jump_fraction": ("pyrecest_imm_jump_fraction", "diagnostic_pyrecest_imm_jump_fraction"),
    "pyrecest_imm_jump_velocity_decay": ("pyrecest_imm_jump_velocity_decay", "diagnostic_pyrecest_imm_jump_velocity_decay"),
}

PYRECEST_DEFAULTS = {
    "pyrecest_particles": 512,
    "pyrecest_alpha": 0.80,
    "pyrecest_beta": 1.00,
    "pyrecest_process_noise_sigma_cm_s": 60.0,
    "pyrecest_position_jump_sigma_cm": 25.0,
    "pyrecest_jump_probability": 0.03,
    "pyrecest_goal_reset_probability": 0.02,
    "pyrecest_position_proposal_probability": 0.0,
    "pyrecest_initial_velocity_sigma_cm_s": 120.0,
    "pyrecest_imm_mode_stickiness": 0.95,
    "pyrecest_imm_stationary_velocity_decay": 0.0,
    "pyrecest_imm_diffusion_velocity_decay": 0.0,
    "pyrecest_imm_momentum_velocity_decay": 0.95,
    "pyrecest_imm_jump_fraction": 0.9,
    "pyrecest_imm_jump_velocity_decay": 0.25,
}


def pyrecest_metadata_for_config(config: object) -> dict[str, int | float]:
    out: dict[str, int | float] = {}
    for name in PYRECEST_INT_COLUMNS:
        out[name] = int(getattr(config, name, PYRECEST_DEFAULTS[name]))
    for name in PYRECEST_FLOAT_COLUMNS:
        out[name] = float(getattr(config, name, PYRECEST_DEFAULTS[name]))
    return out


def pyrecest_config_kwargs_for_scores(scores: pd.DataFrame, defaults: dict[str, int | float] | None = None) -> dict[str, int | float]:
    base = dict(PYRECEST_DEFAULTS)
    if defaults:
        base.update(defaults)
    out: dict[str, int | float] = {}
    for name, columns in PYRECEST_INT_COLUMNS.items():
        out[name] = _unique_positive_int(scores, columns, int(base[name]))
    for name, columns in PYRECEST_FLOAT_COLUMNS.items():
        out[name] = _unique_float(scores, columns, float(base[name]))
    return out


def apply_pyrecest_score_metadata_patch() -> None:
    from . import benchmarks as bench
    from . import ground_truth as gt
    from .pyrecest_models import PyRecEstGoalParticleIMMModel, PyRecEstGoalParticleModel

    if getattr(gt, "_pyrecest_score_metadata_patch_applied", False):
        return

    base_metadata = bench._benchmark_config_metadata
    base_compare = gt.compare_scores_to_ground_truth

    def benchmark_config_metadata(config) -> dict[str, object]:
        metadata = dict(base_metadata(config))
        metadata.update(pyrecest_metadata_for_config(config))
        return metadata

    def compare_scores_to_ground_truth(root, scores, **kwargs) -> pd.DataFrame:
        frame = scores.copy() if isinstance(scores, pd.DataFrame) else pd.read_csv(scores)
        defaults = {k: v for k, v in kwargs.items() if k in PYRECEST_DEFAULTS}
        kwargs.update(pyrecest_config_kwargs_for_scores(frame, defaults))
        return base_compare(root, frame, **kwargs)

    def _score_with_metadata(base_score):
        def score_with_metadata(self, emissions, bin_centers):
            result = base_score(self, emissions, bin_centers)
            result.diagnostics.update(_model_diagnostics(self))
            return result

        score_with_metadata._pyrecest_metadata_wrapped = True
        return score_with_metadata

    bench._benchmark_config_metadata = benchmark_config_metadata
    gt.compare_scores_to_ground_truth = compare_scores_to_ground_truth
    if not getattr(PyRecEstGoalParticleModel.score, "_pyrecest_metadata_wrapped", False):
        PyRecEstGoalParticleModel.score = _score_with_metadata(PyRecEstGoalParticleModel.score)
    if not getattr(PyRecEstGoalParticleIMMModel.score, "_pyrecest_metadata_wrapped", False):
        PyRecEstGoalParticleIMMModel.score = _score_with_metadata(PyRecEstGoalParticleIMMModel.score)
    gt._pyrecest_score_metadata_patch_applied = True


def _model_diagnostics(model: object) -> dict[str, int | float]:
    out: dict[str, int | float] = {
        "pyrecest_particles": int(getattr(model, "n_particles")),
        "pyrecest_alpha": float(getattr(model, "alpha")),
        "pyrecest_beta": float(getattr(model, "beta")),
        "pyrecest_process_noise_sigma_cm_s": float(getattr(model, "process_noise_sigma_cm_s")),
        "pyrecest_position_jump_sigma_cm": float(getattr(model, "position_jump_sigma_cm")),
        "pyrecest_jump_probability": float(getattr(model, "jump_probability")),
        "pyrecest_goal_reset_probability": float(getattr(model, "goal_reset_probability")),
        "pyrecest_position_proposal_probability": float(getattr(model, "position_proposal_probability")),
        "pyrecest_initial_velocity_sigma_cm_s": float(getattr(model, "initial_velocity_sigma_cm_s")),
    }
    if hasattr(model, "mode_stickiness"):
        out.update(
            {
                "pyrecest_imm_mode_stickiness": float(getattr(model, "mode_stickiness")),
                "pyrecest_imm_stationary_velocity_decay": float(getattr(model, "stationary_velocity_decay")),
                "pyrecest_imm_diffusion_velocity_decay": float(getattr(model, "diffusion_velocity_decay")),
                "pyrecest_imm_momentum_velocity_decay": float(getattr(model, "momentum_velocity_decay")),
                "pyrecest_imm_jump_fraction": float(getattr(model, "jump_fraction")),
                "pyrecest_imm_jump_velocity_decay": float(getattr(model, "jump_velocity_decay")),
            }
        )
    return out


def _unique_int(frame: pd.DataFrame, columns: tuple[str, ...], default: int) -> int:
    numeric_values = _numeric_metadata_values(frame, columns)
    if not numeric_values:
        return int(default)
    values: list[int] = []
    for value in numeric_values:
        integer_value = int(round(value))
        if not np.isclose(value, integer_value, rtol=0.0, atol=1e-9):
            raise ValueError(f"{' / '.join(columns)} must be an integer")
        values.append(integer_value)
    if any(value != values[0] for value in values[1:]):
        raise ValueError(f"{' / '.join(columns)} contains multiple values")
    return values[0]


def _unique_positive_int(frame: pd.DataFrame, columns: tuple[str, ...], default: int) -> int:
    value = _unique_int(frame, columns, default)
    if value <= 0:
        raise ValueError(f"{' / '.join(columns)} must be positive")
    return value


def _unique_float(frame: pd.DataFrame, columns: tuple[str, ...], default: float) -> float:
    values = _numeric_metadata_values(frame, columns)
    if not values:
        return float(default)
    if any(not np.isclose(value, values[0]) for value in values[1:]):
        raise ValueError(f"{' / '.join(columns)} contains multiple values")
    return float(values[0])


_MISSING_METADATA_STRINGS = {"", "nan", "na", "n/a", "none", "null", "<na>"}


def _numeric_metadata_values(frame: pd.DataFrame, columns: tuple[str, ...]) -> list[float]:
    values: list[float] = []
    for column in columns:
        if column not in frame.columns:
            continue
        for value in frame[column]:
            parsed = _metadata_float_from_value(value, column)
            if parsed is not None:
                values.append(parsed)
    return values


def _metadata_float_from_value(value: object, column: str) -> float | None:
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    if isinstance(value, (bool, np.bool_)):
        raise _invalid_numeric_metadata(column, value)

    text = str(value).strip()
    if text.lower() in _MISSING_METADATA_STRINGS:
        return None

    try:
        numeric = float(text)
    except (TypeError, ValueError) as exc:
        raise _invalid_numeric_metadata(column, value) from exc
    if not np.isfinite(numeric):
        raise ValueError(f"{column} must be finite")
    return float(numeric)


def _invalid_numeric_metadata(column: str, value: object) -> ValueError:
    return ValueError(f"{column} must contain finite numeric metadata, got {value!r}")
