import numpy as np
import pytest

from hipporeplayimm.state_space import _mass_retaining_candidate_indices


def test_mass_retaining_candidate_support_rejects_noninteger_count_bounds():
    log_emission = np.log(np.array([0.50, 0.30, 0.15, 0.05], dtype=float))

    invalid_bounds = (
        {"top_k": 1.5},
        {"top_k": True},
        {"min_k": 2.5},
        {"min_k": np.bool_(True)},
        {"max_k": 2.5},
        {"max_k": np.array([2])},
    )
    for kwargs in invalid_bounds:
        with pytest.raises(TypeError, match="must be an integer"):
            _mass_retaining_candidate_indices(log_emission, 0.95, **kwargs)


def test_mass_retaining_candidate_support_accepts_integer_valued_count_bounds():
    log_emission = np.log(np.array([0.50, 0.30, 0.15, 0.05], dtype=float))

    selected = _mass_retaining_candidate_indices(
        log_emission,
        0.95,
        top_k=1.0,
        min_k=2.0,
        max_k=3.0,
    )

    assert list(selected) == [0, 1, 2]
