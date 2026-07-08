from functools import wraps

import numpy as np
import pytest

import hipporeplayimm
from hipporeplayimm.models import CandidateKinematicModel, DiffusionModel


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"sigma_cm": "12.0"}, "sigma_cm"),
        ({"max_step_sigma": np.asarray("3.0")}, "max_step_sigma"),
    ],
)
def test_diffusion_model_rejects_string_numeric_parameters(kwargs, message):
    hipporeplayimm.apply_runtime_patches()

    with pytest.raises(TypeError, match=message):
        DiffusionModel(**kwargs)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"stationary_sigma_cm": "2.0"}, "stationary_sigma_cm"),
        ({"diffusion_sigma_cm": np.asarray("12.0")}, "diffusion_sigma_cm"),
        ({"momentum_sigma_cm": "12.0"}, "momentum_sigma_cm"),
        ({"velocity_decay": "0.95"}, "velocity_decay"),
        ({"mode_stickiness": np.asarray("0.94")}, "mode_stickiness"),
    ],
)
def test_candidate_kinematic_model_rejects_string_numeric_parameters(kwargs, message):
    hipporeplayimm.apply_runtime_patches()

    with pytest.raises(TypeError, match=message):
        CandidateKinematicModel(**kwargs)


def test_model_numeric_parameters_still_accept_numeric_scalars():
    hipporeplayimm.apply_runtime_patches()

    diffusion = DiffusionModel(sigma_cm=np.float64(12.0), max_step_sigma=np.asarray(3.0))
    candidate = CandidateKinematicModel(
        stationary_sigma_cm=np.float64(2.0),
        diffusion_sigma_cm=np.asarray(12.0),
        momentum_sigma_cm=12.0,
        velocity_decay=0.95,
        mode_stickiness=np.float64(0.94),
    )

    assert diffusion.sigma_cm == np.float64(12.0)
    assert candidate.velocity_decay == 0.95


def _legacy_validate_positive_parameter(name: str, value: object) -> None:
    numeric = float(value)
    if not np.isfinite(numeric) or numeric <= 0.0:
        raise ValueError(f"{name} must be finite and positive")


def _legacy_validate_nonnegative_parameter(name: str, value: object) -> None:
    numeric = float(value)
    if not np.isfinite(numeric) or numeric < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")


def _legacy_validate_probability_parameter(name: str, value: object) -> None:
    numeric = float(value)
    if not np.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise ValueError(f"{name} must be finite and lie in [0, 1]")


def _stale_string_only_wrapper(original, marker: str):
    @wraps(original)
    def validator(name: str, value: object) -> None:
        if isinstance(value, (str, bytes, np.str_, np.bytes_)):
            raise TypeError(f"{name} must be a numeric scalar, not string")
        return original(name, value)

    setattr(validator, marker, True)
    setattr(validator, "__hipporeplayimm_original__", original)
    return validator


def test_runtime_patches_refresh_stale_model_numeric_validators(monkeypatch):
    hipporeplayimm.apply_runtime_patches()

    from hipporeplayimm import models

    monkeypatch.setattr(models, "_validate_positive_parameter", _legacy_validate_positive_parameter)
    monkeypatch.setattr(models, "_validate_nonnegative_parameter", _legacy_validate_nonnegative_parameter)
    monkeypatch.setattr(models, "_validate_probability_parameter", _legacy_validate_probability_parameter)

    hipporeplayimm.apply_runtime_patches()

    with pytest.raises(TypeError, match="sigma_cm"):
        DiffusionModel(sigma_cm=True)
    with pytest.raises(TypeError, match="sigma_cm"):
        DiffusionModel(sigma_cm="12.0")
    with pytest.raises(TypeError, match="velocity_decay"):
        CandidateKinematicModel(velocity_decay=True)
    with pytest.raises(ValueError, match="velocity_decay"):
        CandidateKinematicModel(velocity_decay=1.1)
    with pytest.raises(TypeError, match="mode_stickiness"):
        CandidateKinematicModel(mode_stickiness="0.95")


def test_runtime_patches_refresh_already_marked_stale_model_numeric_validators(monkeypatch):
    hipporeplayimm.apply_runtime_patches()

    from hipporeplayimm import model_numeric_string_validation, models

    marker = model_numeric_string_validation._PATCHED_FLAG
    monkeypatch.setattr(models, "_validate_positive_parameter", _stale_string_only_wrapper(_legacy_validate_positive_parameter, marker))
    monkeypatch.setattr(models, "_validate_nonnegative_parameter", _stale_string_only_wrapper(_legacy_validate_nonnegative_parameter, marker))
    monkeypatch.setattr(models, "_validate_probability_parameter", _stale_string_only_wrapper(_legacy_validate_probability_parameter, marker))

    hipporeplayimm.apply_runtime_patches()

    assert getattr(models._validate_positive_parameter, model_numeric_string_validation._PATCH_VERSION_ATTR) == model_numeric_string_validation._PATCH_VERSION

    with pytest.raises(TypeError, match="sigma_cm"):
        DiffusionModel(sigma_cm=True)
    with pytest.raises(TypeError, match="velocity_decay"):
        CandidateKinematicModel(velocity_decay=True)
    with pytest.raises(ValueError, match="velocity_decay"):
        CandidateKinematicModel(velocity_decay=1.1)

    current_positive_validator = models._validate_positive_parameter
    hipporeplayimm.apply_runtime_patches()
    assert models._validate_positive_parameter is current_positive_validator
