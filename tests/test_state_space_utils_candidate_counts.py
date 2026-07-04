"""Regression tests for candidate-support count validation."""

from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.state_space_utils import (
    _mass_retaining_candidate_indices,
    _top_candidate_indices,
)


def test_top_candidate_indices_rejects_non_integer_top_k() -> None:
    log_emission = np.asarray([0.0, 1.0, -1.0], dtype=float)

    with pytest.raises(TypeError, match="top_k must be an integer scalar"):
        _top_candidate_indices(log_emission, 1.5)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="top_k must be an integer count, not boolean"):
        _top_candidate_indices(log_emission, True)  # type: ignore[arg-type]


def test_mass_retaining_candidate_indices_rejects_non_integer_counts() -> None:
    log_emission = np.log(np.asarray([0.7, 0.2, 0.1], dtype=float))

    with pytest.raises(TypeError, match="top_k must be an integer scalar"):
        _mass_retaining_candidate_indices(log_emission, 0.8, top_k=1.0)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="min_k must be an integer scalar"):
        _mass_retaining_candidate_indices(log_emission, 0.8, min_k=1.2)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="max_k must be an integer scalar"):
        _mass_retaining_candidate_indices(log_emission, 0.8, max_k=2.1)  # type: ignore[arg-type]


def test_mass_retaining_candidate_indices_accepts_numpy_integer_counts() -> None:
    log_emission = np.log(np.asarray([0.7, 0.2, 0.1], dtype=float))

    result = _mass_retaining_candidate_indices(
        log_emission,
        0.8,
        top_k=np.int64(1),
        min_k=np.int64(1),
        max_k=np.int64(2),
    )

    assert result.tolist() == [0, 1]
