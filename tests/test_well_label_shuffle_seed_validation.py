from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hipporeplayimm import well_label_shuffle_patch
from hipporeplayimm.result_improvements import shuffle_well_labels


def _label_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event": [0, 1, 2],
            "true_well_id": ["A", "B", None],
            "true_well_x": [1.0, 2.0, np.nan],
            "true_well_y": [10.0, 20.0, np.nan],
        }
    )


@pytest.mark.parametrize(
    "seed",
    [True, np.bool_(False), 1.5, np.nan, -1, np.array([1])],
)
def test_shuffle_well_labels_rejects_invalid_random_seed(seed: object) -> None:
    with pytest.raises(ValueError, match="random_seed"):
        shuffle_well_labels(_label_frame(), random_seed=seed)  # type: ignore[arg-type]


@pytest.mark.parametrize("seed", ["2", b"2", np.str_("2"), np.bytes_("2")])
def test_shuffle_well_labels_rejects_text_random_seed(seed: object) -> None:
    with pytest.raises(ValueError, match="random_seed.*finite nonnegative integer"):
        shuffle_well_labels(_label_frame(), random_seed=seed)  # type: ignore[arg-type]


def test_shuffle_well_labels_accepts_integer_like_random_seed() -> None:
    shuffled = shuffle_well_labels(_label_frame(), random_seed=3.0)  # type: ignore[arg-type]

    assert shuffled.loc[:1, ["true_well_id", "true_well_x", "true_well_y"]].to_numpy().tolist() == [
        ["B", 2.0, 20.0],
        ["A", 1.0, 10.0],
    ]


@pytest.mark.parametrize("seed", [2**53 + 1, np.int64(2**53 + 1)])
def test_shuffle_well_labels_forwards_large_integer_seed_exactly(
    monkeypatch: pytest.MonkeyPatch,
    seed: object,
) -> None:
    observed_seeds: list[int] = []

    class IdentityRng:
        @staticmethod
        def permutation(size: int) -> np.ndarray:
            return np.arange(size)

    def capture_seed(value: int) -> IdentityRng:
        observed_seeds.append(value)
        return IdentityRng()

    monkeypatch.setattr(
        well_label_shuffle_patch.np.random,
        "default_rng",
        capture_seed,
    )

    shuffle_well_labels(_label_frame(), random_seed=seed)  # type: ignore[arg-type]

    assert observed_seeds == [2**53 + 1]
