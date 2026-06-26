from __future__ import annotations

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


def _resolve(mode_transitions):
    apply_runtime_patches()
    return duration_occupancy._resolve_mode_transitions(
        _DummyStateSpace,
        3,
        0.9,
        mode_transitions,
        1,
    )


def test_custom_duration_imm_mode_transition_must_be_row_stochastic() -> None:
    bad_transition = np.eye(3, dtype=float)
    bad_transition[0, 0] = 0.5

    with pytest.raises(ValueError, match="rows must sum to 1"):
        _resolve([bad_transition])


@pytest.mark.parametrize(
    ("bad_value", "match"),
    [
        (np.nan, "finite probabilities"),
        (np.inf, "finite probabilities"),
        (-0.1, "nonnegative probabilities"),
    ],
)
def test_custom_duration_imm_mode_transition_rejects_invalid_probabilities(bad_value: float, match: str) -> None:
    bad_transition = np.eye(3, dtype=float)
    bad_transition[0, 0] = bad_value

    with pytest.raises(ValueError, match=match):
        _resolve([bad_transition])


def test_custom_duration_imm_mode_transition_rejects_boolean_masks() -> None:
    boolean_transition = np.eye(3, dtype=bool)

    with pytest.raises(ValueError, match="not booleans"):
        _resolve([boolean_transition])


def test_custom_duration_imm_mode_transition_accepts_valid_probability_matrix() -> None:
    transition = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.2, 0.7, 0.1],
            [0.25, 0.25, 0.5],
        ],
        dtype=float,
    )

    resolved = _resolve([transition])

    assert len(resolved) == 1
    np.testing.assert_allclose(resolved[0], transition)


def test_generated_duration_imm_mode_transitions_still_use_base_resolver() -> None:
    apply_runtime_patches()

    resolved = duration_occupancy._resolve_mode_transitions(
        _DummyStateSpace,
        3,
        0.9,
        None,
        2,
    )

    assert len(resolved) == 2
    np.testing.assert_allclose(resolved[0].sum(axis=1), np.ones(3))
