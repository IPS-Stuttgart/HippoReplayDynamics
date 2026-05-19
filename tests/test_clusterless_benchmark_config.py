from types import SimpleNamespace

from hipporeplayimm.benchmarks import (
    BenchmarkConfig,
    _benchmark_config_metadata,
    _build_models,
    _clusterless_mark_config,
)
from hipporeplayimm.clusterless import ClusterlessStateSpaceReplayModel
from hipporeplayimm.encoding import EncodingConfig


def test_benchmark_config_propagates_clusterless_local_kde_options():
    config = BenchmarkConfig(
        models=("clusterless-state-space-momentum",),
        encoding=EncodingConfig(use_excitatory=False),
        clusterless_mark_likelihood="local-kde",
        clusterless_mark_kde_bandwidth=0.75,
        clusterless_mark_kde_spatial_sigma_bins=2.5,
        clusterless_mark_kde_max_neighbors=17,
    )

    mark_config = _clusterless_mark_config(config)
    metadata = _benchmark_config_metadata(config)
    models = _build_models(config)
    model = models["clusterless-state-space-momentum"]

    assert mark_config.mark_likelihood == "local-kde"
    assert mark_config.mark_kde_bandwidth == 0.75
    assert mark_config.mark_kde_spatial_sigma_bins == 2.5
    assert mark_config.mark_kde_max_neighbors == 17
    assert mark_config.use_excitatory is False
    assert metadata["clusterless_mark_likelihood"] == "local-kde"
    assert metadata["clusterless_mark_kde_bandwidth"] == "0.75"
    assert metadata["clusterless_mark_kde_spatial_sigma_bins"] == "2.5"
    assert metadata["clusterless_mark_kde_max_neighbors"] == 17
    assert isinstance(model, ClusterlessStateSpaceReplayModel)
    assert model.mark_likelihood == "local-kde"


def test_benchmark_model_evidence_cli_clusterless_config_accepts_kde_args():
    from scripts.benchmark_model_evidence import (
        _clusterless_mark_config as script_clusterless_mark_config,
    )

    args = SimpleNamespace(
        bin_size_cm=6.0,
        smoothing_sigma_bins=2.0,
        min_speed_cm_s=5.0,
        clusterless_mark_likelihood="local-kde",
        clusterless_mark_smoothing_sigma_bins=1.25,
        clusterless_mark_prior_count=0.5,
        clusterless_mark_variance_floor=0.2,
        clusterless_rate_floor_hz=1e-3,
        clusterless_mark_kde_bandwidth=0.9,
        clusterless_mark_kde_spatial_sigma_bins=3.0,
        clusterless_mark_kde_max_neighbors=31,
    )

    mark_config = script_clusterless_mark_config(args)

    assert mark_config.mark_likelihood == "local-kde"
    assert mark_config.mark_kde_bandwidth == 0.9
    assert mark_config.mark_kde_spatial_sigma_bins == 3.0
    assert mark_config.mark_kde_max_neighbors == 31
