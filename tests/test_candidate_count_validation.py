from __future__ import annotations

import numpy as np
import pytest

import hipporeplayimm.state_space as state_space


@pytest.mark.parametrize("value", [True, False, np.bool_(True), 1.5, np.float64(2.0), "2", b"2", np.asarray([2])])
def test_top_candidate_indices_rejects_non_integer_counts(value):
    with pytest.raises(TypeError, match="top_k"):
        state_space._top_candidate_indices(np.asarray([0.0, 1.0, 2.0]), value)


@pytest.mark.parametrize("kwargs", [
    {"top_k": 1.5},
    {"top_k": "2"},
    {"top_k": np.asarray([2])},
    {"min_k": 1.5},
    {"min_k": "1"},
    {"min_k": np.asarray([1])},
    {"max_k": 2.5},
    {"max_k": "3"},
    {"max_k": np.asarray([3])},
])
def test_mass_retaining_candidate_indices_rejects_non_integer_counts(kwargs):
    with pytest.raises(TypeError, match="top_k|min_k|max_k"):
        state_space._mass_retaining_candidate_indices(
            np.asarray([0.0, 1.0, 2.0]),
            0.8,
            **kwargs,
        )


def test_candidate_count_validation_keeps_integer_behavior():
    values = np.asarray([0.0, 3.0, 1.0, 2.0])

    assert state_space._top_candidate_indices(values, 2).tolist() == [1, 3]
    assert state_space._mass_retaining_candidate_indices(
        values,
        0.8,
        top_k=1,
        min_k=1,
        max_k=3,
    ).tolist() == [1, 3]
