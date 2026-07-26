import numpy as np
import pandas as pd
import pytest

from hipporeplayimm.benchmarks import BenchmarkConfig, _benchmark_config_metadata
from hipporeplayimm.encoding import EmissionConfig
from hipporeplayimm.ground_truth import _emission_config_for_scores
from hipporeplayimm.score_metadata import emission_config_for_scores



def test_emission_metadata_recovers_legacy_spike_rate_scale_column():
    scores = pd.DataFrame(
        {
            "time_bin_s": [0.015],
            "spike_rate_scale": [0.5],
        }
    )

    config = emission_config_for_scores(
        scores,
        EmissionConfig(time_bin_s=0.02, spike_rate_scale=1.0),
    )

    assert config.time_bin_s == pytest.approx(0.015)
    assert config.spike_rate_scale == pytest.approx(0.5)



def test_ground_truth_emission_metadata_recovers_canonical_spike_rate_scale_column():
    scores = pd.DataFrame(
        {
            "emission_time_bin_s": [0.01],
            "emission_spike_rate_scale": [2.5],
        }
    )

    config = _emission_config_for_scores(
        scores,
        EmissionConfig(time_bin_s=0.02, spike_rate_scale=1.0),
    )

    assert config.time_bin_s == pytest.approx(0.01)
    assert config.spike_rate_scale == pytest.approx(2.5)



def test_emission_metadata_rejects_conflicting_spike_rate_scale_aliases():
    scores = pd.DataFrame(
        {
            "emission_spike_rate_scale": [0.5],
            "spike_rate_scale": [0.75],
        }
    )

    with pytest.raises(ValueError, match="emission_spike_rate_scale"):
        emission_config_for_scores(scores, EmissionConfig())



def test_emission_metadata_accepts_float32_rounding_between_aliases():
    scores = pd.DataFrame(
        {
            "emission_time_bin_s": pd.Series([0.1], dtype=np.float32),
            "time_bin_s": pd.Series([0.1], dtype=np.float64),
        }
    )

    config = emission_config_for_scores(scores, EmissionConfig())

    assert config.time_bin_s == pytest.approx(0.1)



def test_emission_metadata_still_rejects_full_precision_alias_conflicts():
    scores = pd.DataFrame(
        {
            "emission_time_bin_s": [0.1],
            "time_bin_s": [0.10000000149011612],
        }
    )

    with pytest.raises(ValueError, match="emission_time_bin_s"):
        emission_config_for_scores(scores, EmissionConfig())



def test_emission_metadata_rejects_boolean_spike_rate_scale():
    scores = pd.DataFrame({"emission_spike_rate_scale": [True]})

    with pytest.raises(ValueError, match="finite numeric metadata"):
        emission_config_for_scores(scores, EmissionConfig())



def test_benchmark_metadata_writes_canonical_spike_rate_scale_column():
    metadata = _benchmark_config_metadata(
        BenchmarkConfig(emissions=EmissionConfig(time_bin_s=0.01, spike_rate_scale=0.5))
    )

    assert metadata["emission_time_bin_s"] == pytest.approx(0.01)
    assert metadata["emission_spike_rate_scale"] == pytest.approx(0.5)
