from __future__ import annotations

import numpy as np

import hipporeplayimm
from hipporeplayimm.pyrecest_models import _mode_diagnostics


class _ModeFilter:
    def __init__(self, probabilities: object, names: tuple[str, ...] = ("stationary", "momentum")) -> None:
        self.mode_probabilities = probabilities
        self.mode_names = names
        self.last_mode_transition_fraction = 0.25

    def most_likely_mode(self) -> str:
        return str(self.mode_names[int(np.argmax(np.asarray(self.mode_probabilities, dtype=float)))])


def test_pyrecest_mode_diagnostics_normalizes_valid_probabilities() -> None:
    hipporeplayimm.apply_runtime_patches()

    diagnostics = _mode_diagnostics(_ModeFilter(np.array([2.0, 6.0], dtype=float)))

    assert diagnostics["pyrecest_mode_stationary_probability"] == 0.25
    assert diagnostics["pyrecest_mode_momentum_probability"] == 0.75
    assert diagnostics["pyrecest_most_likely_mode"] == "momentum"
    assert diagnostics["pyrecest_last_mode_transition_fraction"] == 0.25


def test_pyrecest_mode_diagnostics_skips_invalid_probabilities() -> None:
    hipporeplayimm.apply_runtime_patches()
    invalid_probabilities = [
        np.array([np.nan, 1.0], dtype=float),
        np.array([0.0, -1.0], dtype=float),
        np.array([0.0, 0.0], dtype=float),
        np.array([[0.5, 0.5]], dtype=float),
        np.array([1.0], dtype=float),
    ]

    for probabilities in invalid_probabilities:
        assert _mode_diagnostics(_ModeFilter(probabilities)) == {}
