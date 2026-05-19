import numpy as np
import pandas as pd

from hipporeplayimm.data import ReplaySession
from hipporeplayimm.encoding import EncodingConfig
from hipporeplayimm.position_validation import (
    PositionDecodingConfig,
    select_position_validated_encoding_configs,
    summarize_position_decoding,
    validated_position_encoding_config,
    validate_session_position_decoding,
)


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
    assert summary.loc[0, "decode_bin_s"] == 1.0
    assert summary.loc[0, "bin_size_cm"] == 10.0
    assert summary.loc[0, "smoothing_sigma_bins"] == 0.5
    assert summary.loc[0, "min_speed_cm_s"] == 0.0


def test_position_decoding_config_defaults_use_validated_settings():
    config = PositionDecodingConfig()
    encoding = validated_position_encoding_config()

    assert config.decode_bin_s == 1.0
    assert config.encoding == encoding
    assert encoding.bin_size_cm == 6.0
    assert encoding.smoothing_sigma_bins == 2.0
    assert encoding.min_speed_cm_s == 5.0


def test_select_position_validated_encoding_configs_prefers_best_passing_row():
    frame = pd.DataFrame(
        [
            {
                "session": "RatX/Open1",
                "bin_size_cm": 8.0,
                "smoothing_sigma_bins": 1.0,
                "min_speed_cm_s": 4.0,
                "passes_smoke_gate": False,
                "median_posterior_mean_error_cm": 1.0,
            },
            {
                "session": "RatX/Open1",
                "bin_size_cm": 6.0,
                "smoothing_sigma_bins": 2.0,
                "min_speed_cm_s": 5.0,
                "passes_smoke_gate": True,
                "median_posterior_mean_error_cm": 10.0,
            },
            {
                "session": "RatX/Open1",
                "bin_size_cm": 4.0,
                "smoothing_sigma_bins": 2.5,
                "min_speed_cm_s": 6.0,
                "passes_smoke_gate": True,
                "median_posterior_mean_error_cm": 8.0,
            },
        ]
    )
    base = EncodingConfig(min_occupancy_s=0.5, rate_floor_hz=0.25)

    configs = select_position_validated_encoding_configs(frame, base_encoding=base)

    selected = configs["RatX/Open1"]
    assert selected.bin_size_cm == 4.0
    assert selected.smoothing_sigma_bins == 2.5
    assert selected.min_speed_cm_s == 6.0
    assert selected.min_occupancy_s == 0.5
    assert selected.rate_floor_hz == 0.25
