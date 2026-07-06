from __future__ import annotations

import numpy as np
import pytest

import hipporeplayimm
from hipporeplayimm.state_space import _augment_candidates_with_momentum_predictions
from hipporeplayimm.state_space_model import _transition_decay_at


def _tiny_candidates() -> list[np.ndarray]:
    return [np.array([0]), np.array([1]), np.array([2])]


def _tiny_centers() -> np.ndarray:
    return np.arange(3.0, dtype=float).reshape(-1, 1)


def test_momentum_prediction_rejects_nonfinite_velocity_decays() -> None:
    hipporeplayimm.apply_runtime_patches()

    with pytest.raises(ValueError, match=r"velocity_decays.*finite nonnegative"):
        _augment_candidates_with_momentum_predictions(
            _tiny_candidates(),
            _tiny_centers(),
            predicted_top_k=1,
            velocity_decay=0.9,
            velocity_decays=np.array([0.9, np.nan], dtype=float),
        )


def test_momentum_prediction_rejects_nonfinite_fallback_decay() -> None:
    hipporeplayimm.apply_runtime_patches()

    with pytest.raises(ValueError, match=r"velocity_decay.*finite nonnegative"):
        _augment_candidates_with_momentum_predictions(
            _tiny_candidates(),
            _tiny_centers(),
            predicted_top_k=1,
            velocity_decay=np.inf,
            velocity_decays=np.array([0.9], dtype=float),
        )


def test_transition_decay_guard_rejects_boolean_decay_values() -> None:
    hipporeplayimm.apply_runtime_patches()

    bad_cases = (
        (None, True),
        (None, np.bool_(False)),
        (np.array(True), 0.9),
        (np.array([True], dtype=bool), 0.9),
        (np.array([False], dtype=object), 0.9),
    )
    for values, fallback in bad_cases:
        with pytest.raises(ValueError, match="finite nonnegative"):
            _transition_decay_at(values, 0, fallback)


def test_transition_decay_guard_preserves_valid_values_and_fallback() -> None:
    hipporeplayimm.apply_runtime_patches()

    assert _transition_decay_at(np.array([0.25], dtype=float), 0, 0.9) == pytest.approx(0.25)
    assert _transition_decay_at(np.array([0.25], dtype=float), 1, 0.9) == pytest.approx(0.9)
    with pytest.raises(ValueError, match=r"velocity_decays.*finite nonnegative"):
        _transition_decay_at(np.array([-0.1], dtype=float), 0, 0.9)


def test_transition_decay_guard_rejects_noninteger_indices() -> None:
    hipporeplayimm.apply_runtime_patches()

    bad_indices = (0.5, np.nan, np.array([0], dtype=int), True)
    for transition_index in bad_indices:
        with pytest.raises(ValueError, match="transition_index"):
            _transition_decay_at(np.array([0.25], dtype=float), transition_index, 0.9)
