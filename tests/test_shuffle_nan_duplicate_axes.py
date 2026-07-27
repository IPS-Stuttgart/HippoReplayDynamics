from __future__ import annotations

import numpy as np
import pytest

import hipporeplayimm.shuffle_spike_time_order as shuffle_patch
from hipporeplayimm.encoding import EncodingConfig, EncodingModel
from hipporeplayimm.shuffle_controls import shuffled_encoding


class _FailIfPermutationCalled:
    def permutation(self, _values):
        raise AssertionError("identical NaN-containing slices must not be permuted")


def _encoding(rates_hz: np.ndarray) -> EncodingModel:
    rates_hz = np.asarray(rates_hz, dtype=float)
    n_cells, n_bins = rates_hz.shape
    return EncodingModel(
        x_edges=np.arange(n_bins + 1, dtype=float),
        y_edges=np.array([0.0, 1.0], dtype=float),
        bin_centers=np.column_stack(
            (np.arange(n_bins, dtype=float) + 0.5, np.full(n_bins, 0.5))
        ),
        rates_hz=rates_hz,
        occupancy_s=np.ones(n_bins, dtype=float),
        cell_ids=np.arange(10, 10 + n_cells, dtype=int),
        config=EncodingConfig(),
    )


@pytest.mark.parametrize(
    ("mode", "rates_hz"),
    [
        (
            "cell-permutation",
            np.array([[np.nan, 1.0], [np.nan, 1.0]], dtype=float),
        ),
        (
            "spatial-permutation",
            np.array([[np.nan, np.nan], [1.0, 1.0]], dtype=float),
        ),
    ],
)
def test_duplicate_nan_slices_return_without_permutation(
    mode: str,
    rates_hz: np.ndarray,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = _FailIfPermutationCalled()
    monkeypatch.setattr(
        shuffle_patch.np.random,
        "default_rng",
        lambda _seed: generator,
    )
    encoding = _encoding(rates_hz)

    control = shuffled_encoding(encoding, mode=mode, random_seed=0)

    assert np.array_equal(control.rates_hz, encoding.rates_hz, equal_nan=True)
