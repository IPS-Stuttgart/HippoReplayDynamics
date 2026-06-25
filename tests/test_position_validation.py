import numpy as np
import pytest

from hipporeplayimm.data import ReplaySession
from hipporeplayimm.encoding import EncodingConfig
from hipporeplayimm.position_decoding_config_validation import _validated_position_decoding_config, _validated_train_frame_mask
from hipporeplayimm.position_validation import (
    PositionDecodingConfig,
    _spike_counts_for_window,
    fit_place_field_encoding_for_position_mask,
    summarize_position_decoding,
    validated_position_encoding_config,
    validate_session_position_decoding,
)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"decode_bin_s": 0.0}, "decode_bin_s"),
        ({"decode_bin_s": float("nan")}, "decode_bin_s"),
        ({"n_folds": 0}, "n_folds"),
        ({"n_folds": 1.5}, "n_folds"),
        ({"max_windows_per_session": 0}, "max_windows_per_session"),
        ({"max_windows_per_session": -1}, "max_windows_per_session"),
        ({"max_windows_per_session": 1.5}, "max_windows_per_session"),
        ({"min_spikes_per_window": -1}, "min_spikes_per_window"),
        ({"min_spikes_per_window": 0.5}, "min_spikes_per_window"),
    ],
)
def test_validate_session_position_decoding_rejects_invalid_config_values(kwargs, message):
    config = PositionDecodingConfig(**kwargs)

    with pytest.raises(ValueError, match=message):
        validate_session_position_decoding(object(), config)  # type: ignore[arg-type]


def test_position_decoding_config_validation_normalizes_accepted_values():
    config = PositionDecodingConfig(
        decode_bin_s="1.0",  # type: ignore[arg-type]
        n_folds="3.0",  # type: ignore[arg-type]
        max_windows_per_session="9.0",  # type: ignore[arg-type]
        min_spikes_per_window="0.0",  # type: ignore[arg-type]
    )

    normalized = _validated_position_decoding_config(config)

    assert normalized.decode_bin_s == 1.0
    assert normalized.n_folds == 3
    assert normalized.max_windows_per_session == 9
    assert normalized.min_spikes_per_window == 0


def test_position_train_frame_mask_validation_accepts_bool_and_binary_numeric_values():
    expected = np.array([True, False, True], dtype=bool)

    np.testing.assert_array_equal(_validated_train_frame_mask([True, False, True], 3), expected)
    np.testing.assert_array_equal(_validated_train_frame_mask(np.array([1.0, 0.0, 1.0]), 3), expected)


@pytest.mark.parametrize(
    "bad_mask",
    [
        np.array([1.0, np.nan, 0.0]),
        np.array([1.0, np.inf, 0.0]),
        np.array([1.0, 0.5, 0.0]),
        np.array(["yes", "no", "yes"], dtype=object),
        np.array([[1.0], [0.0], [1.0]]),
        np.array([1.0, 0.0]),
    ],
)
def test_position_train_frame_mask_validation_rejects_non_boolean_values(bad_mask):
    with pytest.raises(ValueError, match="train_frame_mask"):
        _validated_train_frame_mask(bad_mask, 3)


def test_position_mask_encoding_falls_back_to_all_spikes_without_excitatory_labels(tmp_path):
    times = np.array([0.0, 1.0, 2.0, 3.0], dtype=float)
    position = np.column_stack(
        [
            times,
            times,
            np.zeros_like(times),
            np.zeros_like(times),
        ]
    )
    session = ReplaySession(
        rat="RatX",
        name="OpenX",
        path=tmp_path,
        position=position,
        spikes=np.array([[1.5, 7.0]], dtype=float),
        tetrode_cell_ids=np.array([[1, 7]], dtype=float),
        excitatory_neurons=np.array([], dtype=int),
        inhibitory_neurons=np.array([], dtype=int),
        ripple_events=np.empty((0, 6), dtype=float),
        run_times=np.array([[0.0, 3.0]], dtype=float),
        sleep_box_immobile_times=np.empty((0, 2), dtype=float),
        sleep_times=np.empty((0, 2), dtype=float),
        rem_times=np.empty((0, 2), dtype=float),
        well_sequence=None,
        metadata={},
    )
    config = EncodingConfig(
        bin_size_cm=1.0,
        smoothing_sigma_bins=0.0,
        min_speed_cm_s=0.0,
        min_occupancy_s=1e-6,
        rate_floor_hz=1e-4,
        arena_padding_cm=0.0,
        use_excitatory=True,
    )

    encoding = fit_place_field_encoding_for_position_mask(
        session,
        np.ones(times.shape, dtype=bool),
        config,
    )

    assert encoding.cell_ids.tolist() == [7]
    assert encoding.n_cells == 1
    assert float(np.max(encoding.rates_hz[0])) > config.rate_floor_hz
    np.testing.assert_array_equal(
        _spike_counts_for_window(session, encoding, 1.0, 2.0),
        np.array([1], dtype=int),
    )


def test_position_mask_encoding_rejects_fractional_train_mask(tmp_path):
    times = np.array([0.0, 1.0, 2.0], dtype=float)
    position = np.column_stack([times, times, np.zeros_like(times), np.zeros_like(times)])
    session = ReplaySession(
        rat="RatX",
        name="OpenX",
        path=tmp_path,
        position=position,
        spikes=np.empty((0, 2), dtype=float),
        tetrode_cell_ids=np.empty((0, 2), dtype=float),
        excitatory_neurons=np.array([], dtype=int),
        inhibitory_neurons=np.array([], dtype=int),
        ripple_events=np.empty((0, 6), dtype=float),
        run_times=np.array([[0.0, 2.0]], dtype=float),
        sleep_box_immobile_times=np.empty((0, 2), dtype=float),
        sleep_times=np.empty((0, 2), dtype=float),
        rem_times=np.empty((0, 2), dtype=float),
        well_sequence=None,
        metadata={},
    )
    config = EncodingConfig(
        bin_size_cm=1.0,
        smoothing_sigma_bins=0.0,
        min_speed_cm_s=0.0,
        min_occupancy_s=1e-6,
        rate_floor_hz=1e-4,
        arena_padding_cm=0.0,
    )

    with pytest.raises(ValueError, match="train_frame_mask"):
        fit_place_field_encoding_for_position_mask(session, np.array([1.0, 0.5, 0.0]), config)


@pytest.mark.parametrize(
    ("config_kwargs", "message"),
    [
        ({"smoothing_sigma_bins": -1.0}, "smoothing_sigma_bins"),
        ({"min_occupancy_s": 0.0}, "min_occupancy_s"),
        ({"rate_floor_hz": float("nan")}, "rate_floor_hz"),
        ({"arena_padding_cm": -0.1}, "arena_padding_cm"),
    ],
)
def test_position_mask_encoding_rejects_invalid_encoding_config(tmp_path, config_kwargs, message):
    times = np.array([0.0, 1.0, 2.0], dtype=float)
    position = np.column_stack([times, times, np.zeros_like(times), np.zeros_like(times)])
    session = ReplaySession(
        rat="RatX",
        name="OpenX",
        path=tmp_path,
        position=position,
        spikes=np.empty((0, 2), dtype=float),
        tetrode_cell_ids=np.empty((0, 2), dtype=float),
        excitatory_neurons=np.array([], dtype=int),
        inhibitory_neurons=np.array([], dtype=int),
        ripple_events=np.empty((0, 6), dtype=float),
        run_times=np.array([[0.0, 2.0]], dtype=float),
        sleep_box_immobile_times=np.empty((0, 2), dtype=float),
        sleep_times=np.empty((0, 2), dtype=float),
        rem_times=np.empty((0, 2), dtype=float),
        well_sequence=None,
        metadata={},
    )
    config = EncodingConfig(
        bin_size_cm=1.0,
        min_speed_cm_s=0.0,
        **config_kwargs,
    )

    with pytest.raises(ValueError, match=message):
        fit_place_field_encoding_for_position_mask(session, np.ones(times.shape, dtype=bool), config)


def test_validate_session_position_decoding_returns_finite_cv_metrics(tmp_path):
    times = np.linspace(0.0, 20.0, 1001)
    x = np.linspace(0.0, 100.0, times.size)
    y = np.zeros_like(x)
    position = np.column_stack([times, x, y, np.zeros_like(x)])
    cell1_times = times[(x > 15.0) & (x < 35.0)][::10]
    cell2_times = times[(x > 65.0) & (x < 85.0)][::10]
    spikes = np.vstack(
        [
            np.column_stack([cell1_times, np.ones(cell1_times.shape)]),
            np.column_stack([cell2_times, np.full(cell2_times.shape, 2.0)]),
        ]
    )
    session = ReplaySession(
        rat="RatX",
        name="OpenX",
        path=tmp_path,
        position=position,
        spikes=spikes,
        tetrode_cell_ids=np.array([[1, 1], [1, 2]]),
        excitatory_neurons=np.array([1, 2]),
        inhibitory_neurons=np.array([]),
        ripple_events=np.empty((0, 6)),
        run_times=np.array([[0.0, 20.0]]),
        sleep_box_immobile_times=np.empty((0, 2)),
        sleep_times=np.empty((0, 2)),
        rem_times=np.empty((0, 2)),
        well_sequence=None,
        metadata={},
    )
    config = PositionDecodingConfig(
        encoding=EncodingConfig(bin_size_cm=10.0, smoothing_sigma_bins=0.5, min_speed_cm_s=0.0),
        decode_bin_s=1.0,
        n_folds=3,
        max_windows_per_session=9,
        random_seed=0,
    )

    samples = validate_session_position_decoding(session, config)
    summary = summarize_position_decoding(samples)

    assert len(samples) == 9
    assert np.isfinite(samples["posterior_mean_error_cm"]).all()
    assert np.isfinite(samples["map_error_cm"]).all()
    assert set(samples["observation_model"]) == {"sorted-spike-poisson"}
    assert set(samples["clusterless_mark_likelihood"]) == {"not_implemented"}
    assert summary.loc[0, "decode_windows"] == 9


def test_position_decoding_config_defaults_use_validated_settings():
    config = PositionDecodingConfig()
    encoding = validated_position_encoding_config()

    assert config.decode_bin_s == 1.0
    assert config.encoding == encoding
    assert encoding.bin_size_cm == 6.0
    assert encoding.smoothing_sigma_bins == 2.0
    assert encoding.min_speed_cm_s == 5.0
