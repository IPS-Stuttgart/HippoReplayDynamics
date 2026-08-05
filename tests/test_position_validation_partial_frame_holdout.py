from pathlib import Path

import numpy as np
import pytest

from hipporeplayimm.data import ReplaySession
from hipporeplayimm.encoding import EncodingConfig
from hipporeplayimm.position_validation import (
    _fit_place_field_encoding_excluding_intervals,
)


def _session_with_spike(tmp_path: Path, spike_time: float) -> ReplaySession:
    times = np.array([0.0, 1.0, 2.0], dtype=float)
    return ReplaySession(
        rat="RatX",
        name="OpenX",
        path=tmp_path,
        position=np.column_stack(
            [times, times, np.zeros_like(times), np.zeros_like(times)]
        ),
        spikes=np.array([[spike_time, 7.0]], dtype=float),
        tetrode_cell_ids=np.array([[1, 7]], dtype=float),
        excitatory_neurons=np.array([7], dtype=int),
        inhibitory_neurons=np.empty(0, dtype=int),
        ripple_events=np.empty((0, 6), dtype=float),
        run_times=np.array([[0.0, 2.0]], dtype=float),
        sleep_box_immobile_times=np.empty((0, 2), dtype=float),
        sleep_times=np.empty((0, 2), dtype=float),
        rem_times=np.empty((0, 2), dtype=float),
        well_sequence=None,
        metadata={},
    )


def _encoding_config() -> EncodingConfig:
    return EncodingConfig(
        bin_size_cm=1.0,
        smoothing_sigma_bins=0.0,
        min_speed_cm_s=0.0,
        min_occupancy_s=1e-6,
        rate_floor_hz=1e-4,
        arena_padding_cm=0.0,
        use_excitatory=True,
    )


def _fit(tmp_path: Path, spike_time: float):
    return _fit_place_field_encoding_excluding_intervals(
        _session_with_spike(tmp_path, spike_time),
        np.array([True, True, False]),
        _encoding_config(),
        excluded_intervals=np.array([[0.5, 1.5]], dtype=float),
    )


def test_partial_frame_holdout_removes_exact_occupancy_and_spikes(tmp_path) -> None:
    encoding = _fit(tmp_path, 0.75)

    assert float(encoding.occupancy_s.sum()) == pytest.approx(1.0)
    np.testing.assert_allclose(
        encoding.rates_hz,
        _encoding_config().rate_floor_hz,
    )


def test_partial_frame_holdout_keeps_right_endpoint_spike(tmp_path) -> None:
    encoding = _fit(tmp_path, 1.5)

    assert float(encoding.occupancy_s.sum()) == pytest.approx(1.0)
    assert float(np.max(encoding.rates_hz)) > _encoding_config().rate_floor_hz
