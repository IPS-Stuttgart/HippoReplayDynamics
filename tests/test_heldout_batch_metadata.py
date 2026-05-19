import numpy as np
import pytest

from hipporeplayimm.benchmarks import BenchmarkConfig
from hipporeplayimm.encoding import EmissionConfig
from scripts.heldout_batch import _heldout_score_metadata


def test_heldout_score_metadata_records_exact_split_and_replay_config():
    config = BenchmarkConfig(
        emissions=EmissionConfig(time_bin_s=0.015, spike_rate_scale=0.5),
        test_cell_fraction=0.4,
        candidate_top_k=17,
        pyrecest_particles=33,
        random_seed=11,
    )

    metadata = _heldout_score_metadata(
        config,
        train_cells=np.asarray([1, 3]),
        test_cells=np.asarray([2]),
    )

    assert metadata["train_cell_ids"] == "1,3"
    assert metadata["test_cell_ids"] == "2"
    assert metadata["benchmark_test_cell_fraction"] == pytest.approx(0.4)
    assert metadata["benchmark_random_seed"] == 11
    assert metadata["emission_time_bin_s"] == pytest.approx(0.015)
    assert metadata["emission_spike_rate_scale"] == pytest.approx(0.5)
    assert metadata["candidate_top_k"] == 17
    assert metadata["pyrecest_particles"] == 33
