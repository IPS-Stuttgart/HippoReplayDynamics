import numpy as np

from hipporeplayimm.ground_truth_candidate_support import _score_joint_for_ground_truth


def test_ground_truth_candidate_support_passes_bin_centers_to_modern_models():
    train_emissions = object()
    joint_emissions = object()
    bin_centers = np.array([[0.0, 0.0], [1.0, 0.0]])
    sentinel_score = object()

    class ModernCandidateModel:
        def __init__(self):
            self.seen = {}

        def candidate_indices(self, emissions, bin_centers_arg):
            self.seen["candidate_emissions"] = emissions
            self.seen["candidate_bin_centers"] = bin_centers_arg
            return [np.array([0]), np.array([1])]

        def score(self, emissions, bin_centers_arg, candidate_indices=None):
            self.seen["score_emissions"] = emissions
            self.seen["score_bin_centers"] = bin_centers_arg
            self.seen["candidate_indices"] = candidate_indices
            return sentinel_score

    model = ModernCandidateModel()

    score = _score_joint_for_ground_truth(
        model,
        train_emissions,
        joint_emissions,
        bin_centers,
    )

    assert score is sentinel_score
    assert model.seen["candidate_emissions"] is train_emissions
    assert model.seen["candidate_bin_centers"] is bin_centers
    assert model.seen["score_emissions"] is joint_emissions
    assert model.seen["score_bin_centers"] is bin_centers
    assert [arr.tolist() for arr in model.seen["candidate_indices"]] == [[0], [1]]


def test_ground_truth_candidate_support_preserves_legacy_one_arg_models():
    train_emissions = object()
    joint_emissions = object()
    bin_centers = np.array([[0.0, 0.0], [1.0, 0.0]])
    sentinel_score = object()

    class LegacyCandidateModel:
        def __init__(self):
            self.seen = {}

        def candidate_indices(self, emissions):
            self.seen["candidate_emissions"] = emissions
            return [np.array([0])]

        def score(self, emissions, bin_centers_arg, candidate_indices=None):
            self.seen["score_emissions"] = emissions
            self.seen["score_bin_centers"] = bin_centers_arg
            self.seen["candidate_indices"] = candidate_indices
            return sentinel_score

    model = LegacyCandidateModel()

    score = _score_joint_for_ground_truth(
        model,
        train_emissions,
        joint_emissions,
        bin_centers,
    )

    assert score is sentinel_score
    assert model.seen["candidate_emissions"] is train_emissions
    assert model.seen["score_emissions"] is joint_emissions
    assert model.seen["score_bin_centers"] is bin_centers
    assert [arr.tolist() for arr in model.seen["candidate_indices"]] == [[0]]
