from __future__ import annotations

from pathlib import Path

import numpy as np

from hipporeplayimm.clusterless import ClusterlessMarkConfig, fit_clusterless_mark_encoding
from hipporeplayimm.data import ReplaySession, SpikeMarkData
from hipporeplayimm.encoding import EncodingConfig, fit_place_field_encoding
from hipporeplayimm.kd_reference import KDEncodingConfig, fit_kd_place_field_encoding


def _two_run_bout_session() -> ReplaySession:
    return ReplaySession(
        rat="RatX",
        name="TwoRunBouts",
        path=Path("RatX/TwoRunBouts"),
        position=np.array(
            [
                [0.0, 0.0, 0.0],
                [0.1, 1.0, 0.0],
                [10.0, 0.0, 0.0],
                [10.1, 1.0, 0.0],
            ],
            dtype=float,
        ),
        spikes=np.array([[0.1, 1.0], [10.0, 1.0]], dtype=float),
        tetrode_cell_ids=np.array([[1.0, 1.0]], dtype=float),
        excitatory_neurons=np.array([1], dtype=int),
        inhibitory_neurons=np.array([], dtype=int),
        ripple_events=np.empty((0, 6), dtype=float),
        run_times=np.array([[0.0, 0.1], [10.0, 10.1]], dtype=float),
        sleep_box_immobile_times=np.empty((0, 2), dtype=float),
        sleep_times=np.empty((0, 2), dtype=float),
        rem_times=np.empty((0, 2), dtype=float),
        well_sequence=None,
        metadata={},
        spike_marks=SpikeMarkData(
            times=np.array([0.1, 10.0], dtype=float),
            marks=np.array([[1.0], [2.0]], dtype=float),
            source_file="synthetic.mat",
            source_variable="marks",
            feature_names=("amplitude",),
            cell_ids=np.array([1, 1], dtype=int),
        ),
    )


def _assert_run_local_encoding(occupancy_s: np.ndarray, rates_hz: np.ndarray) -> None:
    np.testing.assert_allclose(occupancy_s, np.array([0.4]), atol=1e-12)
    np.testing.assert_allclose(rates_hz, np.array([[5.0]]), atol=1e-12)


def test_standard_place_field_kinematics_are_local_to_each_run_bout() -> None:
    encoding = fit_place_field_encoding(
        _two_run_bout_session(),
        EncodingConfig(
            bin_size_cm=10.0,
            smoothing_sigma_bins=0.0,
            min_speed_cm_s=5.0,
            min_occupancy_s=0.01,
            arena_padding_cm=0.0,
            exclude_ripple_intervals=False,
        ),
    )

    _assert_run_local_encoding(encoding.occupancy_s, encoding.rates_hz)


def test_kd_place_field_kinematics_are_local_to_each_run_bout() -> None:
    encoding = fit_kd_place_field_encoding(
        _two_run_bout_session(),
        KDEncodingConfig(
            bin_size_cm=10.0,
            n_bins_x=1,
            n_bins_y=1,
            smoothing_sigma_cm=0.0,
            min_speed_cm_s=5.0,
            min_occupancy_s=0.01,
            min_peak_rate_hz=0.0,
        ),
    )

    _assert_run_local_encoding(encoding.occupancy_s, encoding.rates_hz)


def test_clusterless_place_field_kinematics_are_local_to_each_run_bout() -> None:
    encoding = fit_clusterless_mark_encoding(
        _two_run_bout_session(),
        ClusterlessMarkConfig(
            encoding=EncodingConfig(
                bin_size_cm=10.0,
                smoothing_sigma_bins=0.0,
                min_speed_cm_s=5.0,
                min_occupancy_s=0.01,
                arena_padding_cm=0.0,
                exclude_ripple_intervals=False,
            ),
            mark_smoothing_sigma_bins=0.0,
            rate_floor_hz=1e-8,
            mark_likelihood="diagonal-gaussian",
            mark_group_by="none",
        ),
    )

    np.testing.assert_allclose(encoding.occupancy_s, np.array([0.4]), atol=1e-12)
    np.testing.assert_allclose(encoding.rate_hz, np.array([5.0]), atol=1e-12)
