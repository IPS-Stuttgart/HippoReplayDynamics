import numpy as np

from hipporeplayimm.state_space_utils import _restrict_candidates_to_valid_bins


def test_restrict_candidates_to_valid_bins_preserves_candidate_order() -> None:
    log_likelihood = np.array([[0.0, 4.0, 3.0, 2.0, 5.0]])
    candidates = [np.array([4, 2, 3, 1])]
    valid_mask = np.array([False, True, True, False, True])

    restricted = _restrict_candidates_to_valid_bins(candidates, log_likelihood, valid_mask)

    assert restricted[0].tolist() == [4, 2, 1]


def test_restrict_candidates_to_valid_bins_fallback_keeps_best_valid_bin() -> None:
    log_likelihood = np.array([[0.0, 4.0, 3.0, 2.0, 5.0]])
    candidates = [np.array([0, 3])]
    valid_mask = np.array([False, True, True, False, False])

    restricted = _restrict_candidates_to_valid_bins(candidates, log_likelihood, valid_mask)

    assert restricted[0].tolist() == [1]
