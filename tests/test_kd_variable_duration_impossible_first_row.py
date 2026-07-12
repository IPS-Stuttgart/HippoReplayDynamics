from __future__ import annotations

import numpy as np

from hipporeplayimm import kd_reference as kd


def _impossible_first_row() -> np.ndarray:
    return np.array(
        [
            [-np.inf, -np.inf, -np.inf, -np.inf],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=float,
    )


def test_variable_duration_diffusion_returns_negative_infinity_for_impossible_first_row() -> None:
    with np.errstate(divide="raise", invalid="raise"):
        result = kd.kd_diffusion_log_evidence_from_transition(
            _impossible_first_row(),
            2,
            2,
            [np.eye(2, dtype=float)],
        )

    assert np.isneginf(result)


def test_variable_duration_momentum_returns_negative_infinity_for_impossible_first_row() -> None:
    with np.errstate(divide="raise", invalid="raise"):
        result = kd.kd_momentum_log_evidence_from_transitions(
            _impossible_first_row(),
            2,
            np.eye(2, dtype=float),
            [],
        )

    assert np.isneginf(result)
