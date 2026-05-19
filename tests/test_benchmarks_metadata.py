from types import SimpleNamespace

from hipporeplayimm.benchmarks import (
    BenchmarkConfig,
    _benchmark_config_metadata,
    _clusterless_mark_config,
    _session_mark_diagnostics,
)


def test_benchmark_metadata_records_actual_clusterless_mark_likelihood_settings():
    config = BenchmarkConfig(
        clusterless_mark_likelihood="diagonal-gaussian",
        clusterless_mark_kde_bandwidth=2.5,
        clusterless_mark_kde_spatial_sigma_bins=3.5,
        clusterless_mark_kde_max_neighbors=17,
    )

    clusterless = _clusterless_mark_config(config)
    metadata = _benchmark_config_metadata(config)

    assert clusterless.mark_likelihood == "diagonal-gaussian"
    assert clusterless.mark_kde_bandwidth == 2.5
    assert clusterless.mark_kde_spatial_sigma_bins == 3.5
    assert clusterless.mark_kde_max_neighbors == 17
    assert metadata["clusterless_mark_likelihood"] == "diagonal-gaussian"
    assert metadata["clusterless_mark_kde_bandwidth"] == 2.5
    assert metadata["clusterless_mark_kde_spatial_sigma_bins"] == 3.5
    assert metadata["clusterless_mark_kde_max_neighbors"] == 17


def test_benchmark_metadata_normalizes_clusterless_mark_likelihood_alias():
    config = BenchmarkConfig(clusterless_mark_likelihood="kde")

    clusterless = _clusterless_mark_config(config)
    metadata = _benchmark_config_metadata(config)

    assert clusterless.mark_likelihood == "local-kde"
    assert metadata["clusterless_mark_likelihood"] == "local-kde"


def test_session_mark_diagnostics_do_not_hard_code_likelihood_model():
    session = SimpleNamespace(
        spike_marks=SimpleNamespace(
            n_features=4,
            source_file="Spike_Data.mat",
            source_variable="Spike_Amplitude_Marks",
        )
    )

    diagnostics = _session_mark_diagnostics(session)

    assert diagnostics["spike_mark_features"] == 4
    assert diagnostics["clusterless_mark_likelihood_available"] is True
    assert "clusterless_mark_likelihood" not in diagnostics
