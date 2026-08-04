from __future__ import annotations

import warnings

import numpy as np
import pytest

import hipporeplayimm
from hipporeplayimm.models import CandidateKinematicModel, DiffusionModel
from hipporeplayimm.state_space import _mode_transition_matrix
from hipporeplayimm.state_space_utils import _coerce_unit_probability


@pytest.mark.parametrize(
    ("factory", "parameter"),
    [
        (lambda value: DiffusionModel(sigma_cm=value), "sigma_cm"),
        (lambda value: CandidateKinematicModel(mode_stickiness=value), "mode_stickiness"),
    ],
)
def test_models_reject_numpy_complex_numeric_parameters(factory, parameter):
    hipporeplayimm.apply_runtime_patches()

    with pytest.raises(TypeError, match=rf"{parameter}.*real numeric scalar"):
        factory(np.complex128(0.75 + 0.25j))


def test_state_space_probability_helper_rejects_object_wrapped_complex_scalar():
    hipporeplayimm.apply_runtime_patches()
    value = np.asarray(0.75 + 0.25j, dtype=object)

    with pytest.raises(ValueError, match=r"probability.*\[0, 1\]"):
        _coerce_unit_probability("probability", value)


def test_mode_transition_matrix_rejects_complex_stickiness_without_cast_warning():
    hipporeplayimm.apply_runtime_patches()

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match=r"mode_stickiness.*\[0, 1\]"):
            _mode_transition_matrix(3, np.complex128(0.9 + 0.2j))
