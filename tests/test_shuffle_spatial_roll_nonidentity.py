from __future__ import annotations

import numpy as np
import pytest

import hipporeplayimm.shuffle_spike_time_order as shuffle_patch
from hipporeplayimm.encoding import EncodingConfig, EncodingModel
from hipporeplayimm.shuffle_controls import shuffled_encoding


class _PeriodicNoopThenChangeGenerator:
    """Yield a nonzero value-no-op roll before a genuinely changed roll."""

    def __init__(self) -> None:
        self._flat_shifts = iter((2, 1))
        self.calls = 0

    def integers(self, low: int, high: int) -> int:
        assert (low, high) == (1, 4)
        self.calls += 1
        return next(self._flat_shifts)


def _periodic_encoding() -> EncodingModel:
    return EncodingModel(
        x_edges=np.array([0.0, 1.0, 2.0], dtype=float),
        y_edges=np.array([0.0, 1.0, 2.0], dtype=float),
        bin_centers=np.array(
            [
                [0.5, 0.5],
                [0.5, 1.5],
                [1.5, 0.5],
                [1.5, 1.5],
            ],
            dtype=float,
        ),
        rates_hz=np.array(
            [
                [1.0, 2.0, 1.0, 2.0],
                [5.0, 5.0, 5.0, 5.0],
            ],
            dtype=float,
        ),
        occupancy_s=np.ones(4, dtype=float),
        cell_ids=np.array([10, 11], dtype=int),
        config=EncodingConfig(),
    )


def test_spatial_roll_retries_periodic_value_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoding = _periodic_encoding()
    generator = _PeriodicNoopThenChangeGenerator()
    monkeypatch.setattr(
        shuffle_patch.np.random,
        "default_rng",
        lambda _seed: generator,
    )

    control = shuffled_encoding(
        encoding,
        mode="spatial-roll",
        random_seed=0,
    )

    # Flat shift 2 maps to (1, 0), which is nonzero but leaves the periodic
    # first map unchanged. Flat shift 1 maps to (0, 1) and must be retried.
    np.testing.assert_array_equal(
        control.rates_hz[0],
        np.array([2.0, 1.0, 2.0, 1.0]),
    )
    # A constant map has no observably different circular shift.
    np.testing.assert_array_equal(control.rates_hz[1], encoding.rates_hz[1])
    assert generator.calls == 2
