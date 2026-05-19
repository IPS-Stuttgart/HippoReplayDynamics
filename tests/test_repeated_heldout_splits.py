import pandas as pd
import pytest

from hipporeplayimm.benchmarks import (
    BenchmarkConfig,
    _add_relative_metrics,
    _benchmark_config_metadata,
    _heldout_split_seed,
    _validate_heldout_cell_splits,
)


def test_heldout_split_seed_preserves_historical_first_split():
    assert _heldout_split_seed(17, 0) == 17
    assert _heldout_split_seed(17, 1) == 18
    assert _heldout_split_seed(17, 3) == 20


def test_heldout_cell_splits_validation():
    assert _validate_heldout_cell_splits(1) == 1
    assert _validate_heldout_cell_splits(5) == 5

    with pytest.raises(ValueError, match="heldout_cell_splits"):
        _validate_heldout_cell_splits(0)


def test_benchmark_metadata_records_requested_heldout_cell_splits():
    metadata = _benchmark_config_metadata(BenchmarkConfig(heldout_cell_splits=7))

    assert metadata["benchmark_heldout_cell_splits"] == 7


def test_relative_metrics_use_split_specific_static_baseline():
    rows = pd.DataFrame(
        {
            "session": ["Rat1/Open1"] * 6,
            "event_index": [0] * 6,
            "heldout_split_index": [0, 0, 0, 1, 1, 1],
            "model": [
                "stationary",
                "diffusion",
                "imm",
                "stationary",
                "diffusion",
                "imm",
            ],
            "heldout_log_likelihood": [10.0, 9.0, 12.0, 0.0, 100.0, 95.0],
            "test_spikes": [10] * 6,
        }
    )

    scored = _add_relative_metrics(rows)
    deltas = scored.set_index(["heldout_split_index", "model"])["delta_vs_best_static"]

    assert deltas.loc[(0, "stationary")] == pytest.approx(0.0)
    assert deltas.loc[(0, "imm")] == pytest.approx(2.0)
    assert deltas.loc[(1, "diffusion")] == pytest.approx(0.0)
    assert deltas.loc[(1, "imm")] == pytest.approx(-5.0)
