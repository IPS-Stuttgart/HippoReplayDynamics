from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm import apply_runtime_patches
from hipporeplayimm import state_space as public_state_space
from hipporeplayimm import state_space_model, state_space_utils


def test_candidate_count_helpers_reject_fractional_counts() -> None:
    log_emission = np.array([0.0, 1.0, 2.0], dtype=float)

    with pytest.raises(ValueError, match="top_k.*integer"):
        public_state_space._top_candidate_indices(log_emission, 1.5)

    with pytest.raises(ValueError, match="min_k.*integer"):
        public_state_space._mass_retaining_candidate_indices(
            log_emission,
            0.9,
            min_k=1.5,
        )

    with pytest.raises(ValueError, match="max_k.*integer"):
        public_state_space._mass_retaining_candidate_indices(
            log_emission,
            0.9,
            max_k=2.5,
        )


def test_candidate_count_helpers_reject_boolean_counts() -> None:
    log_emission = np.array([0.0, 1.0, 2.0], dtype=float)

    with pytest.raises(TypeError, match="top_k.*not boolean"):
        public_state_space._top_candidate_indices(log_emission, True)

    with pytest.raises(TypeError, match="min_k.*not boolean"):
        public_state_space._mass_retaining_candidate_indices(
            log_emission,
            0.9,
            min_k=np.bool_(True),
        )


def test_candidate_count_helpers_accept_integer_valued_scalar_counts() -> None:
    log_emission = np.array([0.0, 2.0, 1.0], dtype=float)

    np.testing.assert_array_equal(
        public_state_space._top_candidate_indices(log_emission, np.float64(2.0)),
        np.array([1, 2]),
    )
    selected = public_state_space._mass_retaining_candidate_indices(
        log_emission,
        0.8,
        top_k=np.asarray(1),
        min_k=np.float64(2.0),
        max_k=3.0,
    )
    assert selected.size >= 2


def test_candidate_count_validation_refreshes_imported_aliases() -> None:
    apply_runtime_patches()

    assert state_space_model._top_candidate_indices is state_space_utils._top_candidate_indices
    assert public_state_space._top_candidate_indices is state_space_utils._top_candidate_indices
    assert state_space_model._mass_retaining_candidate_indices is state_space_utils._mass_retaining_candidate_indices
    assert public_state_space._mass_retaining_candidate_indices is state_space_utils._mass_retaining_candidate_indices
