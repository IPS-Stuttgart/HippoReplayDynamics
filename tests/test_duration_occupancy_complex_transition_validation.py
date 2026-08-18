from __future__ import annotations

import warnings

import numpy as np
import pytest

import hipporeplayimm.duration_occupancy as duration_occupancy
from hipporeplayimm import apply_runtime_patches


class _DummyStateSpace:
    @staticmethod
    def _mode_transition_matrix(n_modes: int, stickiness: float) -> np.ndarray:
        transition = np.full(
            (int(n_modes), int(n_modes)),
            (1.0 - float(stickiness)) / (int(n_modes) - 1),
            dtype=float,
        )
        np.fill_diagonal(transition, float(stickiness))
        return transition


def test_custom_duration_imm_mode_transition_rejects_complex_probabilities() -> None:
    apply_runtime_patches()
    transition = np.eye(3, dtype=np.complex128)
    transition[0, 0] = 1.0 + 2.0j

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match=r"real probabilities, not complex"):
            duration_occupancy._resolve_mode_transitions(
                _DummyStateSpace,
                3,
                0.9,
                [transition],
                1,
            )


def test_custom_duration_imm_mode_transition_rejects_complex_dtype_even_when_imaginary_part_is_zero() -> None:
    apply_runtime_patches()
    transition = np.eye(3, dtype=np.complex128)

    with pytest.raises(ValueError, match=r"real probabilities, not complex"):
        duration_occupancy._resolve_mode_transitions(
            _DummyStateSpace,
            3,
            0.9,
            [transition],
            1,
        )
