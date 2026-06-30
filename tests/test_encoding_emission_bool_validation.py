from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.data import ReplaySession
from hipporeplayimm.encoding import (
    EncodingConfig,
    _poisson_log_emissions,
    _time_bin_edges,
    fit_place_field_encoding,
)


def _minimal_session(tmp_path):
    times = np.array([0.0, 1.0, 2.0], dtype=float)
    position = np.column_stack(
        [
            times,
            [0.0, 5.0, 10.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    return ReplaySession(
        rat="RatX",
        name="OpenX",
        path=tmp_path,
        position=position,
        spikes=np.array([[0.5, 1.0]], dtype=float),
        tetrode_cell_ids=np.empty((0, 2), dtype=int),
        excitatory_neurons=np.array([], dtype=int),
        inhibitory_neurons=np.array([], dtype=int),
        ripple_events=np.empty((0, 6)),
        run_times=np.array([[0.0, 2.0]], dtype=float),
        sleep_box_immobile_times=np.empty((0, 2)),
        sleep_times=np.empty((0, 2)),
        rem_times=np.empty((0, 2)),
        well_sequence=None,
        metadata={},
    )


@pytest.mark.parametrize(
    "field_name",
    [
        "bin_size_cm",
        "smoothing_sigma_bins",
        "min_speed_cm_s",
        "min_occupancy_s",
        "rate_floor_hz",
        "arena_padding_cm",
    ],
)
def test_encoding_config_rejects_boolean_numeric_fields(tmp_path, field_name):
    kwargs = {"min_speed_cm_s": 0.0, field_name: True}
    config = EncodingConfig(**kwargs)

    with pytest.raises(TypeError, match=field_name):
        fit_place_field_encoding(_minimal_session(tmp_path), config)


@pytest.mark.parametrize(
    ("field_name", "valid_value"),
    [
        ("bin_size_cm", 4.0),
        ("smoothing_sigma_bins", 1.5),
        ("min_speed_cm_s", 0.0),
        ("min_occupancy_s", 0.02),
        ("rate_floor_hz", 1e-4),
        ("arena_padding_cm", 2.0),
    ],
)
def test_encoding_config_rejects_array_shaped_numeric_fields(
    tmp_path,
    field_name,
    valid_value,
):
    config = EncodingConfig(
        min_speed_cm_s=0.0,
        **{field_name: np.asarray([valid_value])},
    )

    with pytest.raises(TypeError, match=field_name):
        fit_place_field_encoding(_minimal_session(tmp_path), config)


@pytest.mark.parametrize("field_name", ["use_excitatory", "exclude_ripple_intervals"])
@pytest.mark.parametrize("value", ["False", 1, np.asarray([True])])
def test_encoding_config_rejects_non_boolean_flag_fields(tmp_path, field_name, value):
    config = EncodingConfig(min_speed_cm_s=0.0, **{field_name: value})

    with pytest.raises(TypeError, match=field_name):
        fit_place_field_encoding(_minimal_session(tmp_path), config)


@pytest.mark.parametrize("field_name", ["use_excitatory", "exclude_ripple_intervals"])
def test_encoding_config_accepts_numpy_boolean_flags(tmp_path, field_name):
    config = EncodingConfig(min_speed_cm_s=0.0, **{field_name: np.bool_(False)})

    fit_place_field_encoding(_minimal_session(tmp_path), config)


def test_emission_time_bin_width_rejects_boolean_scalar():
    with pytest.raises(TypeError, match="time_bin_s"):
        _time_bin_edges(0.0, 1.0, True)


@pytest.mark.parametrize(
    ("start", "end", "time_bin_s", "message"),
    [
        (np.asarray([0.0]), 1.0, 0.02, "ripple start"),
        (0.0, np.asarray([1.0]), 0.02, "ripple end"),
        (0.0, 1.0, np.asarray([0.02]), "time_bin_s"),
    ],
)
def test_emission_time_bin_edges_reject_array_shaped_scalars(
    start,
    end,
    time_bin_s,
    message,
):
    with pytest.raises(TypeError, match=message):
        _time_bin_edges(start, end, time_bin_s)


@pytest.mark.parametrize(
    ("start", "end", "message"),
    [
        (True, 1.0, "ripple start"),
        (0.0, False, "ripple end"),
    ],
)
def test_emission_time_bin_edges_reject_boolean_bounds(start, end, message):
    with pytest.raises(TypeError, match=message):
        _time_bin_edges(start, end, 0.02)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"dt": True}, "dt"),
        ({"spike_rate_scale": True}, "spike_rate_scale"),
        ({"likelihood_temperature": True}, "likelihood_temperature"),
        ({"negative_binomial_overdispersion": True}, "negative_binomial_overdispersion"),
        ({"cell_weights": [True]}, "cell_weights"),
    ],
)
def test_emission_numeric_parameters_reject_booleans(kwargs, message):
    counts = np.zeros((1, 1), dtype=int)
    rates = np.ones((1, 2), dtype=float)
    call_kwargs = dict(kwargs)
    dt = call_kwargs.pop("dt", 0.02)

    with pytest.raises(TypeError, match=message):
        _poisson_log_emissions(counts, rates, dt, **call_kwargs)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"spike_rate_scale": np.asarray([1.0])}, "spike_rate_scale"),
        ({"likelihood_temperature": np.asarray([1.0])}, "likelihood_temperature"),
        (
            {"negative_binomial_overdispersion": np.asarray([0.0])},
            "negative_binomial_overdispersion",
        ),
    ],
)
def test_emission_scalar_parameters_reject_array_shapes(kwargs, message):
    counts = np.zeros((1, 1), dtype=int)
    rates = np.ones((1, 2), dtype=float)

    with pytest.raises(TypeError, match=message):
        _poisson_log_emissions(counts, rates, 0.02, **kwargs)


def test_encoding_grid_patch_refreshes_stale_true_flag(monkeypatch):
    import hipporeplayimm
    import hipporeplayimm.encoding as encoding
    import hipporeplayimm.encoding_grid_extra_columns as patch

    def stale_make_grid(xy, config):
        raise AssertionError("stale _make_grid should have been refreshed")

    monkeypatch.setattr(encoding, "_make_grid", stale_make_grid)
    monkeypatch.setattr(encoding, "_grid_extra_columns_patch_applied", True, raising=False)

    hipporeplayimm.apply_runtime_patches()

    assert getattr(encoding._make_grid, patch._GRID_WRAPPER_MARKER, False)
    _x_edges, _y_edges, centers = encoding._make_grid(
        np.array([[0.0, 1.0, 99.0], [4.0, 5.0, 100.0]], dtype=float),
        EncodingConfig(),
    )
    assert centers.shape[1] == 2


def test_encoding_bool_patch_refreshes_stale_true_flag(monkeypatch):
    import hipporeplayimm
    import hipporeplayimm.encoding as encoding
    import hipporeplayimm.encoding_grid_extra_columns as patch

    def stale_time_bin_edges(start, end, time_bin_s):
        return np.array([float(start), float(end)], dtype=float)

    monkeypatch.setattr(encoding, "_time_bin_edges", stale_time_bin_edges)
    monkeypatch.setattr(encoding, "_encoding_bool_validation_patch_applied", True, raising=False)

    hipporeplayimm.apply_runtime_patches()

    assert getattr(encoding._time_bin_edges, patch._TIME_BIN_EDGES_WRAPPER_MARKER, False)
    with pytest.raises(TypeError, match="time_bin_s"):
        encoding._time_bin_edges(0.0, 1.0, True)
