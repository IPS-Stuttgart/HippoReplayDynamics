from __future__ import annotations

import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from hipporeplayimm.state_space_trajectory_imm import (
    _trajectory_imm_mode_transition_matrices,
)


def _object_scalar(value: object) -> np.ndarray:
    wrapper = np.empty((), dtype=object)
    wrapper[()] = value
    return wrapper


@pytest.mark.parametrize(
    "tau_s",
    [
        True,
        np.bool_(True),
        _object_scalar(np.array(True)),
    ],
)
def test_continuous_time_trajectory_imm_preserves_tau_boolean_validation(
    tau_s: object,
) -> None:
    config = SimpleNamespace(
        imm_switch_tau_s=tau_s,
        trajectory_imm_momentum_switch_probability=None,
    )

    with pytest.raises(TypeError, match="imm_switch_tau_s"):
        _trajectory_imm_mode_transition_matrices(
            config,
            0.95,
            np.asarray([0.01]),
        )


def test_continuous_time_trajectory_imm_preserves_stickiness_validation() -> None:
    config = SimpleNamespace(
        imm_switch_tau_s=0.1,
        trajectory_imm_momentum_switch_probability=None,
    )

    with pytest.raises(TypeError, match="trajectory_imm_mode_stickiness"):
        _trajectory_imm_mode_transition_matrices(
            config,
            True,
            np.asarray([0.01]),
        )


def test_continuous_time_trajectory_imm_rejects_nonscalar_tau_without_deprecation() -> None:
    config = SimpleNamespace(
        imm_switch_tau_s=_object_scalar(np.array([0.1])),
        trajectory_imm_momentum_switch_probability=None,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        with pytest.raises(TypeError, match="imm_switch_tau_s"):
            _trajectory_imm_mode_transition_matrices(
                config,
                0.95,
                np.asarray([0.01]),
            )


def test_continuous_time_trajectory_imm_preserves_nested_numeric_tau() -> None:
    nested_config = SimpleNamespace(
        imm_switch_tau_s=_object_scalar(np.array(0.1)),
        trajectory_imm_momentum_switch_probability=None,
    )
    scalar_config = SimpleNamespace(
        imm_switch_tau_s=0.1,
        trajectory_imm_momentum_switch_probability=None,
    )
    durations = np.asarray([0.01, 0.02])

    nested = _trajectory_imm_mode_transition_matrices(
        nested_config,
        0.95,
        durations,
    )
    scalar = _trajectory_imm_mode_transition_matrices(
        scalar_config,
        0.95,
        durations,
    )

    assert len(nested) == len(scalar)
    for nested_matrix, scalar_matrix in zip(nested, scalar, strict=True):
        np.testing.assert_allclose(
            nested_matrix,
            scalar_matrix,
            rtol=1.0e-12,
            atol=1.0e-12,
        )
