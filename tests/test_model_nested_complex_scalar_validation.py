from __future__ import annotations

import warnings

import numpy as np
import pytest

import hipporeplayimm
from hipporeplayimm.models import DiffusionModel
from hipporeplayimm.state_space_utils import _coerce_unit_probability


def _nested_object_scalar(value: object) -> np.ndarray:
    inner = np.empty((), dtype=object)
    inner[()] = value
    outer = np.empty((), dtype=object)
    outer[()] = inner
    return outer


def test_model_rejects_nested_object_wrapped_complex_parameter_without_cast_warning():
    hipporeplayimm.apply_runtime_patches()
    value = _nested_object_scalar(np.complex128(12.0 + 3.0j))

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(TypeError, match=r"sigma_cm.*real numeric scalar"):
            DiffusionModel(sigma_cm=value)


def test_state_space_probability_rejects_nested_object_wrapped_complex_without_cast_warning():
    hipporeplayimm.apply_runtime_patches()
    value = _nested_object_scalar(np.complex128(0.75 + 0.25j))

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match=r"probability.*\[0, 1\]"):
            _coerce_unit_probability("probability", value)
