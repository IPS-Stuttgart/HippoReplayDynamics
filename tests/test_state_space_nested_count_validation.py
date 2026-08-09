from __future__ import annotations

import warnings

import numpy as np
import pytest

from hipporeplayimm.state_space_utils import (
    _mass_retaining_candidate_indices,
    _top_candidate_indices,
    _uniform_probabilities,
)


def _object_scalar(value):
    wrapper = np.empty((), dtype=object)
    wrapper[()] = value
    return wrapper


def test_state_space_counts_reject_nested_boolean_scalars():
    nested_boolean = _object_scalar(np.array(True))
    scores = np.array([0.0, 2.0, 1.0], dtype=float)

    with pytest.raises(TypeError, match="top_k must be an integer"):
        _top_candidate_indices(scores, nested_boolean)
    with pytest.raises(TypeError, match="min_k must be an integer"):
        _mass_retaining_candidate_indices(scores, mass_threshold=0.0, min_k=nested_boolean)
    with pytest.raises(ValueError, match="n_bins must be a positive integer"):
        _uniform_probabilities(nested_boolean)


def test_state_space_counts_reject_nested_nonscalar_arrays_without_deprecation():
    nested_singleton = _object_scalar(np.array([2]))
    scores = np.array([0.0, 2.0, 1.0], dtype=float)

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        with pytest.raises(TypeError, match="top_k must be an integer"):
            _top_candidate_indices(scores, nested_singleton)


def test_state_space_counts_reject_cyclic_object_scalars():
    cyclic = np.empty((), dtype=object)
    cyclic[()] = cyclic

    with pytest.raises(TypeError, match="top_k must be an integer"):
        _top_candidate_indices(np.array([0.0, 1.0], dtype=float), cyclic)


def test_state_space_counts_preserve_nested_integral_numeric_scalars():
    nested_two = _object_scalar(np.array(2.0))
    nested_three = _object_scalar(np.array(3))

    candidates = _top_candidate_indices(np.array([0.0, 2.0, 1.0], dtype=float), nested_two)
    assert candidates.size == 2
    np.testing.assert_allclose(_uniform_probabilities(nested_three), np.full(3, 1.0 / 3.0))
