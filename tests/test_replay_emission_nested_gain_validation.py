from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.encoding import EncodingConfig, EncodingModel
from hipporeplayimm.replay_emission_calibration import apply_replay_cell_gains


def _two_cell_encoding() -> EncodingModel:
    return EncodingModel(
        x_edges=np.array([0.0, 1.0], dtype=float),
        y_edges=np.array([0.0, 1.0], dtype=float),
        bin_centers=np.array([[0.5, 0.5]], dtype=float),
        rates_hz=np.ones((2, 1), dtype=float),
        occupancy_s=np.array([1.0], dtype=float),
        cell_ids=np.array([1, 2], dtype=int),
        config=EncodingConfig(),
    )


def _nested_object_scalar(value: object) -> np.ndarray:
    inner = np.empty((), dtype=object)
    inner[()] = value
    outer = np.empty((), dtype=object)
    outer[()] = inner
    return outer


def _gain_vector(value: object) -> np.ndarray:
    gains = np.empty(2, dtype=object)
    gains[0] = 1.0
    gains[1] = _nested_object_scalar(value)
    return gains


@pytest.mark.parametrize(
    ("gains", "message"),
    [
        (_gain_vector(np.bool_(True)), "not boolean"),
        (_gain_vector("2.0"), "not text"),
        ({1: _nested_object_scalar(np.bool_(True))}, "not boolean"),
        ({2: _nested_object_scalar("3.0")}, "not text"),
    ],
)
def test_apply_replay_cell_gains_rejects_nested_non_numeric_wrappers(
    gains: object,
    message: str,
) -> None:
    with pytest.raises(TypeError, match=message):
        apply_replay_cell_gains(_two_cell_encoding(), gains)


def test_apply_replay_cell_gains_keeps_nested_real_scalar_compatibility() -> None:
    calibrated = apply_replay_cell_gains(
        _two_cell_encoding(),
        {2: _nested_object_scalar(np.float32(2.5))},
    )

    np.testing.assert_allclose(calibrated.rates_hz, np.array([[1.0], [2.5]]))
