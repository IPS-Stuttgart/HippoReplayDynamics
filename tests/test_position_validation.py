import numpy as np
import pytest

from hipporeplayimm.data import ReplaySession
from hipporeplayimm.encoding import EncodingConfig
from hipporeplayimm.position_validation import (
    PositionDecodingConfig,
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
