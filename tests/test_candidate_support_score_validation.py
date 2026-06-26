from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.state_space_utils import (
    _mass_retaining_candidate_indices,
    _top_candidate_indices,
)


def test_top_candidate_indices_rejects_positive_infinity() -> None:
    with pytest.raises(ValueError, match=r"\+inf"):
        _top_candidate_indices(np.array([0.0, np.inf, -1.0], dtype=float), 1)


def test_top_candidate_indices_ignore_nan_scores() -> None:
    selected = _top_candidate_indices(np.array([0.0, np.nan, -1.0], dtype=float), 2)

    np.testing.assert_array_equal(selected, np.array([0, 2]))


def test_top_candidate_indices_require_one_finite_score() -> None:
    with pytest.raises(ValueError, match="at least one finite"):
        _top_candidate_indices(np.array([-np.inf, -np.inf], dtype=float), 1)


def test_mass_retaining_candidate_indices_rejects_invalid_scores() -> None:
    with pytest.raises(ValueError, match=r"\+inf"):
        _mass_retaining_candidate_indices(np.array([0.0, np.inf, -1.0], dtype=float), 0.9)
