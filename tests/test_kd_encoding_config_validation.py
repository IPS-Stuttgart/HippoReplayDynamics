from pathlib import Path

import numpy as np
import pytest

from hipporeplayimm.data import ReplaySession
from hipporeplayimm.kd_reference import KDEncodingConfig, fit_kd_place_field_encoding


def _minimal_session() -> ReplaySession:
    times = np.linspace(0.0, 1.0, 11)
    position = np.column_stack([times, np.linspace(0.0, 20.0, times.size), np.zeros_like(times)])
    return ReplaySession(
        rat="rat",
        name="session",
        path=Path("."),
        position=position,
        spikes=np.empty((0, 2)),
        tetrode_cell_ids=np.empty((0, 2), dtype=int),
        excitatory_neurons=np.empty(0, dtype=int),
        inhibitory_neurons=np.empty(0, dtype=int),
        ripple_events=np.empty((0, 6)),
        run_times=np.array([[0.0, 1.0]]),
        sleep_box_immobile_times=np.empty((0, 2)),
        sleep_times=np.empty((0, 2)),
        rem_times=np.empty((0, 2)),
        well_sequence=None,
        metadata={},
    )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"bin_size_cm": 0.0}, "bin_size_cm"),
        ({"n_bins_x": 0}, "n_bins_x"),
        ({"n_bins_y": 0}, "n_bins_y"),
        ({"smoothing_sigma_cm": -1.0}, "smoothing_sigma_cm"),
        ({"min_speed_cm_s": -0.1}, "min_speed_cm_s"),
        ({"min_occupancy_s": 0.0}, "min_occupancy_s"),
        ({"rate_floor_hz": 0.0}, "rate_floor_hz"),
        ({"min_peak_rate_hz": -0.1}, "min_peak_rate_hz"),
    ],
)
def test_fit_kd_place_field_encoding_rejects_invalid_config(kwargs, match):
    with pytest.raises(ValueError, match=match):
        fit_kd_place_field_encoding(_minimal_session(), KDEncodingConfig(**kwargs))


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"bin_size_cm": np.asarray([5.0])}, "bin_size_cm"),
        ({"smoothing_sigma_cm": np.asarray([0.0])}, "smoothing_sigma_cm"),
        ({"min_speed_cm_s": np.asarray([0.0])}, "min_speed_cm_s"),
        ({"min_occupancy_s": np.asarray([0.01])}, "min_occupancy_s"),
        ({"rate_floor_hz": np.asarray([1e-4])}, "rate_floor_hz"),
        ({"min_peak_rate_hz": np.asarray([0.0])}, "min_peak_rate_hz"),
    ],
)
def test_fit_kd_place_field_encoding_rejects_array_shaped_float_config(kwargs, match):
    with pytest.raises(TypeError, match=match):
        fit_kd_place_field_encoding(_minimal_session(), KDEncodingConfig(**kwargs))


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"bin_size_cm": True}, "bin_size_cm"),
        ({"smoothing_sigma_cm": False}, "smoothing_sigma_cm"),
        ({"min_speed_cm_s": False}, "min_speed_cm_s"),
        ({"min_occupancy_s": True}, "min_occupancy_s"),
        ({"rate_floor_hz": True}, "rate_floor_hz"),
        ({"min_peak_rate_hz": False}, "min_peak_rate_hz"),
    ],
)
def test_fit_kd_place_field_encoding_rejects_boolean_float_config(kwargs, match):
    with pytest.raises(TypeError, match=match):
        fit_kd_place_field_encoding(_minimal_session(), KDEncodingConfig(**kwargs))


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"bin_size_cm": "5.0"}, "bin_size_cm"),
        ({"smoothing_sigma_cm": "0.0"}, "smoothing_sigma_cm"),
        ({"min_speed_cm_s": b"0.0"}, "min_speed_cm_s"),
        ({"min_occupancy_s": np.asarray("0.01")}, "min_occupancy_s"),
        ({"rate_floor_hz": np.asarray(b"1e-4")}, "rate_floor_hz"),
        ({"min_peak_rate_hz": np.asarray("0.0", dtype=object)}, "min_peak_rate_hz"),
    ],
)
def test_fit_kd_place_field_encoding_rejects_text_float_config(kwargs, match):
    with pytest.raises(TypeError, match=match):
        fit_kd_place_field_encoding(_minimal_session(), KDEncodingConfig(**kwargs))


def test_fit_kd_place_field_encoding_rejects_non_integer_grid_size():
    with pytest.raises(TypeError, match="n_bins_x"):
        fit_kd_place_field_encoding(_minimal_session(), KDEncodingConfig(n_bins_x=10.0))  # type: ignore[arg-type]


def test_fit_kd_place_field_encoding_still_accepts_valid_config():
    encoding = fit_kd_place_field_encoding(
        _minimal_session(),
        KDEncodingConfig(
            bin_size_cm=5.0,
            n_bins_x=5,
            n_bins_y=2,
            smoothing_sigma_cm=0.0,
            min_speed_cm_s=0.0,
            min_occupancy_s=0.01,
            rate_floor_hz=1e-4,
            min_peak_rate_hz=0.0,
        ),
    )

    assert encoding.n_bins == 10
    assert encoding.rates_hz.shape == (0, 10)
