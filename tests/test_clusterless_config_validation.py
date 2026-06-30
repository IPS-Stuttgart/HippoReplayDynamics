from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hipporeplayimm.benchmarks import BenchmarkConfig, _clusterless_mark_config
from hipporeplayimm.clusterless import ClusterlessMarkConfig, fit_clusterless_mark_encoding
from hipporeplayimm.clusterless_config_validation import _validate_clusterless_mark_config
from hipporeplayimm.data import ReplaySession, SpikeMarkData
from hipporeplayimm.encoding import EncodingConfig


def _marked_session() -> ReplaySession:
    spike_times = np.array([0.5], dtype=float)
    return ReplaySession(
        rat="RatX",
        name="OpenY",
        path=Path("unused"),
        position=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 10.0, 0.0],
            ],
            dtype=float,
        ),
        spikes=np.array([[0.5, 1.0]], dtype=float),
        tetrode_cell_ids=np.empty((0, 2), dtype=float),
        excitatory_neurons=np.array([1], dtype=int),
        inhibitory_neurons=np.array([], dtype=int),
        ripple_events=np.empty((0, 6), dtype=float),
        run_times=np.array([[0.0, 1.0]], dtype=float),
        sleep_box_immobile_times=np.empty((0, 2), dtype=float),
        sleep_times=np.empty((0, 2), dtype=float),
        rem_times=np.empty((0, 2), dtype=float),
        well_sequence=None,
        metadata={},
        spike_marks=SpikeMarkData(
            times=spike_times,
            marks=np.array([[10.0, 20.0]], dtype=float),
            source_file="Spike_Data.mat",
            source_variable="Spike_Marks",
            feature_names=("mark_0", "mark_1"),
            cell_ids=np.array([1], dtype=int),
        ),
    )


def test_clusterless_mark_encoding_validates_nested_encoding_config() -> None:
    session = _marked_session()
    config = ClusterlessMarkConfig(
        encoding=EncodingConfig(min_speed_cm_s=0.0, min_occupancy_s=0.0),
    )

    with pytest.raises(ValueError, match="min_occupancy_s"):
        fit_clusterless_mark_encoding(session, config)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("mark_smoothing_sigma_bins", True, "mark_smoothing_sigma_bins"),
        ("mark_prior_count", True, "mark_prior_count"),
        ("mark_variance_floor", True, "mark_variance_floor"),
        ("rate_floor_hz", True, "rate_floor_hz"),
        ("mark_kde_bandwidth", True, "mark_kde_bandwidth"),
        ("mark_kde_spatial_sigma_bins", False, "mark_kde_spatial_sigma_bins"),
        ("mark_kde_max_neighbors", np.bool_(True), "mark_kde_max_neighbors"),
    ],
)
def test_clusterless_mark_config_rejects_boolean_numeric_parameters(
    field: str,
    value: object,
    message: str,
) -> None:
    config = ClusterlessMarkConfig(**{field: value})

    with pytest.raises(ValueError, match=message):
        fit_clusterless_mark_encoding(object(), config)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("mark_smoothing_sigma_bins", np.array([1.0])),
        ("mark_prior_count", np.array([1.0])),
        ("mark_variance_floor", np.array([1.0])),
        ("rate_floor_hz", np.array([1.0])),
        ("mark_kde_bandwidth", np.array([1.0])),
        ("mark_kde_spatial_sigma_bins", np.array([1.0])),
        ("mark_kde_max_neighbors", np.array([16])),
        ("mark_smoothing_sigma_bins", np.array(True)),
        ("mark_kde_max_neighbors", np.array(True, dtype=object)),
    ],
)
def test_clusterless_mark_config_rejects_array_or_bool_scalars(field: str, value: object) -> None:
    config = ClusterlessMarkConfig(**{field: value})

    with pytest.raises(ValueError, match=field):
        _validate_clusterless_mark_config(config)


def test_clusterless_mark_config_accepts_zero_dimensional_numeric_arrays() -> None:
    config = ClusterlessMarkConfig(
        mark_smoothing_sigma_bins=np.array(1.0),
        mark_prior_count=np.array(1.0),
        mark_variance_floor=np.array(1.0),
        rate_floor_hz=np.array(1.0),
        mark_kde_bandwidth=np.array(1.0),
        mark_kde_spatial_sigma_bins=np.array(0.0),
        mark_kde_max_neighbors=np.array(16),
    )

    _validate_clusterless_mark_config(config)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("clusterless_mark_smoothing_sigma_bins", np.array([1.0]), "mark_smoothing_sigma_bins"),
        ("clusterless_mark_prior_count", np.array([1.0]), "mark_prior_count"),
        ("clusterless_mark_variance_floor", np.array([1.0]), "mark_variance_floor"),
        ("clusterless_rate_floor_hz", np.array([1.0]), "rate_floor_hz"),
        ("clusterless_mark_kde_bandwidth", np.array([1.0]), "mark_kde_bandwidth"),
        ("clusterless_mark_kde_spatial_sigma_bins", np.array([1.0]), "mark_kde_spatial_sigma_bins"),
        ("clusterless_mark_kde_max_neighbors", np.array([16]), "mark_kde_max_neighbors"),
        ("clusterless_mark_smoothing_sigma_bins", np.array(True), "mark_smoothing_sigma_bins"),
        ("clusterless_mark_kde_max_neighbors", np.array(True, dtype=object), "mark_kde_max_neighbors"),
    ],
)
def test_benchmark_clusterless_adapter_rejects_array_or_bool_scalars(
    field: str,
    value: object,
    message: str,
) -> None:
    config = BenchmarkConfig(**{field: value})

    with pytest.raises(ValueError, match=message):
        _clusterless_mark_config(config)
