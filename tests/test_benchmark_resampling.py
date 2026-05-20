from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hipporeplayimm.benchmarks import (
    BenchmarkConfig,
    _add_relative_metrics,
    _event_indices,
    _n_cell_splits,
)
from hipporeplayimm.evidence_reporting import EXACT_EVIDENCE_SUPPORT


class _Session:
    ripple_count = 10

    def ripple_indices_in_run(self) -> np.ndarray:
        return np.arange(self.ripple_count, dtype=int)


def test_randomized_event_subset_is_reproducible_and_sorted() -> None:
    config = BenchmarkConfig(
        max_events_per_session=4,
        randomize_event_subset=True,
        event_subset_seed=123,
    )

    indices = _event_indices(_Session(), config, split_index=0)
    expected = np.sort(np.random.default_rng(123).choice(np.arange(10), size=4, replace=False))

    assert np.array_equal(indices, expected)
    assert np.array_equal(indices, np.sort(indices))


def test_n_cell_splits_rejects_non_positive_values() -> None:
    with pytest.raises(ValueError, match="n_cell_splits"):
        _n_cell_splits(BenchmarkConfig(n_cell_splits=0))


def test_relative_metrics_are_cell_split_aware() -> None:
    frame = pd.DataFrame(
        [
            _score_row(split=0, model="stationary", heldout=0.0),
            _score_row(split=0, model="imm", heldout=1.0),
            _score_row(split=1, model="stationary", heldout=100.0),
            _score_row(split=1, model="imm", heldout=101.0),
        ]
    )

    rows = _add_relative_metrics(frame)
    imm = rows[rows["model"] == "imm"].sort_values("benchmark_cell_split_index")

    assert np.allclose(imm["delta_vs_best_static"].to_numpy(float), [1.0, 1.0])


def _score_row(split: int, model: str, heldout: float) -> dict[str, object]:
    return {
        "session": "RatX/OpenY",
        "event_index": 0,
        "benchmark_cell_split_index": split,
        "model": model,
        "heldout_log_likelihood": heldout,
        "test_spikes": 1,
        "evidence_support": EXACT_EVIDENCE_SUPPORT,
        "status": "success",
    }
