from __future__ import annotations

import sys
import types

import numpy as np
import pytest

import hipporeplayimm
from hipporeplayimm.models import CandidateKinematicModel, DiffusionModel


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sigma_cm": True},
        {"max_step_sigma": np.bool_(True)},
    ],
)
def test_diffusion_model_rejects_boolean_numeric_parameters(kwargs: dict[str, object]) -> None:
    hipporeplayimm.apply_runtime_patches()

    with pytest.raises(TypeError, match="not boolean"):
        DiffusionModel(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"stationary_sigma_cm": True},
        {"diffusion_sigma_cm": np.bool_(True)},
        {"momentum_sigma_cm": True},
        {"velocity_decay": False},
        {"mode_stickiness": np.bool_(False)},
    ],
)
def test_candidate_model_rejects_boolean_numeric_parameters(kwargs: dict[str, object]) -> None:
    hipporeplayimm.apply_runtime_patches()

    with pytest.raises(TypeError, match="not boolean"):
        CandidateKinematicModel(**kwargs)


def test_mode_transition_validation_patch_respects_package_namespace(monkeypatch: pytest.MonkeyPatch) -> None:
    from hipporeplayimm import model_parameter_validation, state_space_utils

    installed = state_space_utils._mode_transition_matrix
    original = getattr(installed, "__hipporeplayimm_original__", installed)

    # Force a fresh installation of the wrapper while making sure every real
    # package alias that can be rebound is restored by pytest afterwards.
    monkeypatch.setattr(state_space_utils, "_mode_transition_matrix", original)
    for module in tuple(sys.modules.values()):
        module_name = getattr(module, "__name__", "")
        if module is state_space_utils:
            continue
        if module_name != "hipporeplayimm" and not module_name.startswith("hipporeplayimm."):
            continue
        if getattr(module, "_mode_transition_matrix", None) is original:
            monkeypatch.setattr(module, "_mode_transition_matrix", original)

    external_module = types.ModuleType("hipporeplayimm_extension")
    external_module._mode_transition_matrix = original
    internal_module = types.ModuleType("hipporeplayimm.synthetic_mode_transition_user")
    internal_module._mode_transition_matrix = original
    monkeypatch.setitem(sys.modules, external_module.__name__, external_module)
    monkeypatch.setitem(sys.modules, internal_module.__name__, internal_module)

    model_parameter_validation._apply_state_space_mode_transition_validation_patch()

    assert external_module._mode_transition_matrix is original
    assert internal_module._mode_transition_matrix is state_space_utils._mode_transition_matrix
    assert internal_module._mode_transition_matrix is not original
