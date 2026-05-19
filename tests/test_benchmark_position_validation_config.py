import pytest

from hipporeplayimm.benchmarks import BenchmarkConfig, _config_for_session
from hipporeplayimm.encoding import EncodingConfig


def test_benchmark_config_uses_position_validated_session_encoder():
    base = EncodingConfig(bin_size_cm=10.0, smoothing_sigma_bins=0.5)
    validated = EncodingConfig(bin_size_cm=6.0, smoothing_sigma_bins=2.0)
    config = BenchmarkConfig(
        encoding=base,
        position_validated_encoding_configs={"RatX/Open1": validated},
        position_validated_encoding_source="position_validation_matrix_best_by_session.csv",
    )

    selected = _config_for_session(config, "RatX/Open1")

    assert selected.encoding == validated
    assert config.encoding == base
    assert selected.position_validated_encoding_source.endswith("best_by_session.csv")


def test_benchmark_config_missing_position_validated_session_is_strict():
    config = BenchmarkConfig(
        position_validated_encoding_configs={"RatX/Open1": EncodingConfig(bin_size_cm=6.0)},
    )

    with pytest.raises(KeyError, match="RatX/Open2"):
        _config_for_session(config, "RatX/Open2")


def test_benchmark_config_can_fall_back_for_missing_position_validated_session():
    base = EncodingConfig(bin_size_cm=10.0)
    config = BenchmarkConfig(
        encoding=base,
        position_validated_encoding_configs={"RatX/Open1": EncodingConfig(bin_size_cm=6.0)},
        allow_missing_position_validated_encoding=True,
    )

    selected = _config_for_session(config, "RatX/Open2")

    assert selected is config
    assert selected.encoding == base
