from __future__ import annotations

import warnings
from types import SimpleNamespace

import numpy as np
import pytest

import hipporeplayimm.clusterless_position_validation as validation


def test_clusterless_decode_entropy_handles_impossible_spatial_bins(monkeypatch) -> None:
    monkeypatch.setattr(
        validation,
        "build_clusterless_mark_emissions",
        lambda *args, **kwargs: SimpleNamespace(
            log_likelihood=np.array([[0.0, -np.inf]], dtype=float),
            n_spikes=0,
        ),
    )
    session = SimpleNamespace(session_id="RatX/OpenX")
    encoding = SimpleNamespace(
        bin_centers=np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float),
        n_bins=2,
        mark_likelihood="local-kde",
        spike_mark_source="test",
        n_features=1,
    )
    window = {
        "start_time": 0.0,
        "end_time": 1.0,
        "center_time": 0.5,
        "true_x": 0.0,
        "true_y": 0.0,
    }

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        row = validation._decode_clusterless_window(
            session,
            encoding,
            window,
            0,
            fold_index=0,
        )

    assert row["posterior_entropy"] == pytest.approx(0.0)


def test_clusterless_entropy_retains_finite_support_mass() -> None:
    log_posterior = np.array(
        [-np.log(2.0), -np.log(2.0), -np.inf],
        dtype=float,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        entropy = validation._posterior_entropy(log_posterior)

    assert entropy == pytest.approx(np.log(2.0))
