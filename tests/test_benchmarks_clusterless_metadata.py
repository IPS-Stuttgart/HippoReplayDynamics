from types import SimpleNamespace

import numpy as np

from hipporeplayimm.benchmarks import (
    BenchmarkConfig,
    _benchmark_config_metadata,
    _clusterless_mark_config,
    _session_mark_diagnostics,
)
from hipporeplayimm.data import SpikeMarkData


def test_benchmark_clusterless_metadata_uses_configured_mark_likelihood():
    config = BenchmarkConfig(
        clusterless_mark_likelihood="diag",
        clusterless_mark_kde_bandwidth=0.25,
        clusterless_mark_kde_spatial_sigma_bins=2.0,
        clusterless_mark_kde_max_neighbors=17,
    )

    clusterless_config = _clusterless_mark_config(config)
    metadata = _benchmark_config_metadata(config)

    assert clusterless_config.mark_likelihood == "diagonal-gaussian"
    assert clusterless_config.mark_kde_bandwidth == 0.25
    assert clusterless_config.mark_kde_spatial_sigma_bins == 2.0
    assert clusterless_config.mark_kde_max_neighbors == 17
    assert metadata["clusterless_mark_likelihood"] == "diagonal-gaussian"
    assert metadata["clusterless_mark_kde_bandwidth"] == 0.25
    assert metadata["clusterless_mark_kde_spatial_sigma_bins"] == 2.0
    assert metadata["clusterless_mark_kde_max_neighbors"] == 17


def test_session_mark_diagnostics_does_not_override_clusterless_config_metadata():
    session = SimpleNamespace(
        spike_marks=SpikeMarkData(
            times=np.array([0.0]),
            marks=np.array([[1.0]]),
            source_file="Spike_Data.mat",
            source_variable="Spike_Amplitude_Marks",
            feature_names=("amp",),
            cell_ids=np.array([1]),
        )
    )

    diagnostics = _session_mark_diagnostics(session)

    assert diagnostics["clusterless_mark_likelihood_available"] is True
    assert "clusterless_mark_likelihood" not in diagnostics
