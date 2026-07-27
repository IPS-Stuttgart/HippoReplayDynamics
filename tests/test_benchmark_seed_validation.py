from __future__ import annotations

from decimal import Decimal

import numpy as np
import pytest

import hipporeplayimm
from hipporeplayimm import benchmarks
from hipporeplayimm.benchmarks import BenchmarkConfig


class _Session:
    ripple_count = 4

    def ripple_indices_in_run(self) -> np.ndarray:
        return np.arange(self.ripple_count, dtype=int)


@pytest.mark.parametrize(
    "bad_seed",
    [
        True,
        False,
        np.bool_(True),
        1.5,
        "2.5",
        -1,
        float("nan"),
        float("inf"),
        np.array([1]),
    ],
)
def test_cell_split_seed_rejects_invalid_random_seeds(bad_seed) -> None:
    with pytest.raises(ValueError, match="random_seed"):
        benchmarks._cell_split_seed(bad_seed, 0)


@pytest.mark.parametrize("seed", [7, 7.0, "7", np.int64(7), np.array(7.0)])
def test_cell_split_seed_accepts_exact_integer_scalars(seed) -> None:
    assert benchmarks._cell_split_seed(seed, 3) == 10


def test_event_subset_seed_is_validated_before_random_sampling() -> None:
    config = BenchmarkConfig(
        max_events_per_session=2,
        randomize_event_subset=True,
        event_subset_seed=1.75,
    )

    with pytest.raises(ValueError, match="event_subset_seed"):
        benchmarks._event_indices(_Session(), config)


def test_benchmark_metadata_rejects_fractional_seed_aliases() -> None:
    with pytest.raises(ValueError, match="random_seed"):
        benchmarks._benchmark_config_metadata(BenchmarkConfig(random_seed=4.5))

    with pytest.raises(ValueError, match="event_subset_seed"):
        benchmarks._benchmark_config_metadata(
            BenchmarkConfig(random_seed=4, event_subset_seed="5.5")
        )


def test_benchmark_metadata_preserves_large_integer_seeds_exactly() -> None:
    large = 2**53 + 1
    config = BenchmarkConfig(
        random_seed=Decimal(large),
        event_subset_seed=np.array(large + 1, dtype=object),
    )

    metadata = benchmarks._benchmark_config_metadata(config)
    split_metadata = benchmarks._benchmark_split_metadata(config, 0)

    assert metadata["benchmark_random_seed"] == large
    assert metadata["benchmark_event_subset_base_seed"] == large + 1
    assert split_metadata["benchmark_cell_split_seed"] == large
    assert split_metadata["benchmark_event_subset_seed"] == large + 1


@pytest.mark.parametrize("random_seeds", [(), (1, 2.5), "1,2"])
def test_public_benchmark_runner_rejects_invalid_seed_sequences(
    monkeypatch: pytest.MonkeyPatch,
    random_seeds,
) -> None:
    monkeypatch.setattr(benchmarks, "load_open_field_sessions", lambda _root: [])

    with pytest.raises(ValueError, match="random_seeds"):
        hipporeplayimm.run_open_field_benchmark(
            "unused",
            BenchmarkConfig(random_seeds=random_seeds),
        )


def test_direct_cell_split_rejects_fractional_seed() -> None:
    with pytest.raises(ValueError, match="random_seed"):
        benchmarks._split_cells(np.array([1, 2, 3]), 0.25, 2.75)
