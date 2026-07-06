from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.state_space import (
    _mass_retaining_candidate_indices,
    _top_candidate_indices,
)


@pytest.mark.parametrize(
    "top_k",
    [
        1.5,
        np.asarray(1.0),
        np.asarray([1]),
        np.asarray(True, dtype=object),
    ],
)
def test_top_candidate_indices_rejects_non_integer_scalar_counts(top_k: object) -> None:
    with pytest.raises(TypeError, match="top_k"):
        _top_candidate_indices(np.array([0.0, 1.0, 2.0]), top_k)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"top_k": 1.5},
        {"top_k": np.asarray(1.0)},
        {"top_k": np.asarray([1])},
        {"top_k": np.asarray(True, dtype=object)},
        {"min_k": 1.5},
        {"min_k": np.asarray(1.0)},
        {"min_k": np.asarray([1])},
        {"min_k": np.asarray(True, dtype=object)},
        {"max_k": 1.5},
        {"max_k": np.asarray(1.0)},
        {"max_k": np.asarray([1])},
        {"max_k": np.asarray(True, dtype=object)},
    ],
)
def test_mass_retaining_candidate_indices_rejects_non_integer_count_kwargs(kwargs: dict[str, object]) -> None:
    name = next(iter(kwargs))
    with pytest.raises(TypeError, match=name):
        _mass_retaining_candidate_indices(
            np.array([0.0, 1.0, 2.0]),
            mass_threshold=0.8,
            **kwargs,  # type: ignore[arg-type]
        )


def test_mass_retaining_candidate_indices_still_accepts_integer_numpy_counts() -> None:
    candidates = _mass_retaining_candidate_indices(
        np.array([0.0, 1.0, 2.0]),
        mass_threshold=0.8,
        top_k=np.asarray(1, dtype=np.int64),
        min_k=np.asarray(1, dtype=np.int64),
        max_k=np.asarray(2, dtype=np.int64),
    )
    assert candidates.tolist() == [2, 1]
