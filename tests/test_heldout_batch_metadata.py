import numpy as np
import pytest

from hipporeplayimm.benchmarks import BenchmarkConfig
from hipporeplayimm.encoding import EmissionConfig
from scripts import heldout_batch


def test_heldout_batch_metadata_preserves_split_and_emission_config():
    metadata = heldout_batch._heldout_batch_metadata(
        BenchmarkConfig(
            test_cell_fraction=0.4,
            random_seed=7,
            emissions=EmissionConfig(time_bin_s=0.015, spike_rate_scale=2.5),
        ),
        train_cells=np.array([1, 3, 5]),
        test_cells=np.array([2, 4]),
    )

    assert metadata["train_cell_ids"] == "1,3,5"
    assert metadata["test_cell_ids"] == "2,4"
    assert metadata["benchmark_test_cell_fraction"] == pytest.approx(0.4)
    assert metadata["benchmark_random_seed"] == 7
    assert metadata["emission_time_bin_s"] == pytest.approx(0.015)
    assert metadata["emission_spike_rate_scale"] == pytest.approx(2.5)


def test_failure_row_keeps_metadata_for_posthoc_readers():
    metadata = {
        "train_cell_ids": "1,3,5",
        "test_cell_ids": "2,4",
        "benchmark_test_cell_fraction": 0.4,
        "benchmark_random_seed": 7,
        "emission_time_bin_s": 0.015,
        "emission_spike_rate_scale": 2.5,
    }

    row = heldout_batch.failure_row(
        "Rat1/Open1",
        3,
        "momentum",
        RuntimeError("boom"),
        metadata=metadata,
    )

    assert row["requested_model"] == "momentum"
    for key, value in metadata.items():
        assert row[key] == value
