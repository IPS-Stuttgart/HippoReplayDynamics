"""Keep post-hoc ground-truth decoding candidate support benchmark-consistent."""

from __future__ import annotations

import numpy as np

from . import ground_truth as _ground_truth
from .benchmarks import (
    _call_score_with_optional_occupancy,
    _candidate_indices_for_model,
    _state_space_uses_candidate_support,
)
from .state_space import StateSpaceReplayModel


def _score_joint_for_ground_truth(
    model,
    train_emissions,
    joint_emissions,
    bin_centers: np.ndarray,
    *,
    occupancy_s: np.ndarray | None = None,
):
    """Score held-out ground-truth decodes with benchmark-matched candidates.

    Held-out benchmark scoring derives pruned candidate supports from the train
    emissions and, for state-space momentum/IMM decoders, also passes the grid
    bin centers so dynamically plausible predicted states can be included.  The
    post-hoc behavioral ground-truth decoder must use the same support; otherwise
    endpoint metrics can be decoded from a different posterior than the benchmark
    evidence rows.
    """

    if isinstance(model, StateSpaceReplayModel):
        candidates = (
            _candidate_indices_for_model(model, train_emissions, bin_centers)
            if _state_space_uses_candidate_support(model)
            else None
        )
        return _score_state_space_joint_for_ground_truth(
            model,
            joint_emissions,
            bin_centers,
            candidates,
            occupancy_s,
        )
    if hasattr(model, "candidate_indices"):
        candidates = _candidate_indices_for_model(model, train_emissions, bin_centers)
        return model.score(joint_emissions, bin_centers, candidate_indices=candidates)
    return model.score(joint_emissions, bin_centers)


def _score_state_space_joint_for_ground_truth(
    model: StateSpaceReplayModel,
    joint_emissions,
    bin_centers: np.ndarray,
    candidate_indices,
    occupancy_s: np.ndarray | None = None,
):
    kwargs: dict[str, object] = {"candidate_indices": candidate_indices}
    return _call_score_with_optional_occupancy(
        model.score,
        joint_emissions,
        bin_centers,
        kwargs,
        occupancy_s,
    )


def apply_ground_truth_candidate_support_patch() -> None:
    """Install bin-center-aware held-out ground-truth scoring."""

    _ground_truth._score_joint_for_ground_truth = _score_joint_for_ground_truth


__all__ = ["apply_ground_truth_candidate_support_patch"]
