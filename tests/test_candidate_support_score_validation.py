from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.state_space_utils import (
    _mass_retaining_candidate_indices,
    _top_candidate_indices,
)


def test_top_candidate_indices_rejects_nan_and_positive_infinity() -> None:
    for values in (
        np.array([0.0, np.nan, -1.0], dtype=float),
        np.array([0.0, np.inf, -1.0], dtype=float),
    ):
        with pytest.raises(ValueError, match=r"NaN or \+inf"):
            _top_candidate_indices(values, 1)


def test_top_candidate_indices_require_one_finite_score() -> None:
    with pytest.raises(ValueError, match="at least one finite"):
        _top_candidate_indices(np.array([-np.inf, -np.inf], dtype=float), 1)


def test_mass_retaining_candidate_indices_rejects_invalid_scores() -> None:
    with pytest.raises(ValueError, match=r"NaN or \+inf"):
        _mass_retaining_candidate_indices(np.array([0.0, np.nan, -1.0], dtype=float), 0.9)
