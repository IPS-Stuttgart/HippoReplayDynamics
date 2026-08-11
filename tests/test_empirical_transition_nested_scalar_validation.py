from __future__ import annotations

import warnings

import numpy as np
import pytest

from hipporeplayimm.empirical_transition import _finite_real_scalar


def _object_scalar(value: object) -> np.ndarray:
    wrapper = np.empty((), dtype=object)
    wrapper[()] = value
    return wrapper


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("add_self_loop_count", _object_scalar(np.array(True)), "boolean"),
        ("min_speed_cm_s", _object_scalar(np.array(False)), "boolean"),
        ("teleport_probability", _object_scalar(np.array("0.25")), "text"),
    ],
)
def test_finite_real_scalar_rejects_nested_ambiguous_values(
    name: str,
    value: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(TypeError, match=message):
        _finite_real_scalar(name, value)


def test_finite_real_scalar_rejects_nested_non_scalar_without_numpy_warning() -> None:
    value = _object_scalar(np.array([0.25], dtype=float))

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        with pytest.raises(ValueError, match="teleport_probability must be a real numeric scalar"):
            _finite_real_scalar("teleport_probability", value)


def test_finite_real_scalar_accepts_nested_real_zero_dimensional_values() -> None:
    value = _object_scalar(_object_scalar(np.array(0.25, dtype=np.float64)))

    assert _finite_real_scalar("teleport_probability", value) == pytest.approx(0.25)


def test_finite_real_scalar_rejects_cyclic_zero_dimensional_wrappers() -> None:
    value = np.empty((), dtype=object)
    value[()] = value

    with pytest.raises(ValueError, match="add_self_loop_count must be a real numeric scalar"):
        _finite_real_scalar("add_self_loop_count", value)
