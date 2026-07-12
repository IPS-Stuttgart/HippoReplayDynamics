from __future__ import annotations

import numpy as np
import pytest

import hipporeplayimm
from hipporeplayimm.accuracy_upgrades import ValidStateConfig, valid_state_mask_from_encoding
from hipporeplayimm.encoding import EncodingConfig, EncodingModel


def _encoding(rates_hz: np.ndarray) -> EncodingModel:
    return EncodingModel(
        x_edges=np.array([0.0, 1.0, 2.0, 3.0], dtype=float),
        y_edges=np.array([0.0, 1.0], dtype=float),
        bin_centers=np.array(
            [
                [0.5, 0.5],
                [1.5, 0.5],
                [2.5, 0.5],
            ],
            dtype=float,
        ),
        rates_hz=np.asarray(rates_hz, dtype=float),
        occupancy_s=np.array([10.0, 5.0, 1.0], dtype=float),
        cell_ids=np.array([1], dtype=int),
        config=EncodingConfig(),
    )


def test_valid_state_fallback_respects_finite_rate_requirement() -> None:
    hipporeplayimm.apply_runtime_patches()
    encoding = _encoding(np.array([[np.nan, 2.0, 3.0]], dtype=float))

    mask = valid_state_mask_from_encoding(
        encoding,
        ValidStateConfig(min_occupancy_s=20.0, require_finite_rates=True),
    )

    np.testing.assert_array_equal(mask, np.array([False, True, False]))
    assert np.all(np.isfinite(encoding.rates_hz[:, mask]))


def test_valid_state_fallback_rejects_encoding_without_eligible_bins() -> None:
    hipporeplayimm.apply_runtime_patches()
    encoding = _encoding(np.array([[np.nan, np.inf, -np.inf]], dtype=float))

    with pytest.raises(
        ValueError,
        match="no bins have finite occupancy and firing rates",
    ):
        valid_state_mask_from_encoding(
            encoding,
            ValidStateConfig(min_occupancy_s=20.0, require_finite_rates=True),
        )
