"""Validate replay-wrapper return_trajectory controls without bool() coercion."""

from __future__ import annotations

from functools import wraps

import numpy as np

_PATCHED_APPLIER_FLAG = "_return_trajectory_validation_applier_wrapped"
_PATCHED_SCORE_FLAG = "_return_trajectory_validation_score_wrapped"
_PATCHED_MODULE_FLAG = "_return_trajectory_validation_patch_applied"


def _normalize_return_trajectory(return_trajectory: object) -> bool | None:
    if return_trajectory is None:
        return None
    if isinstance(return_trajectory, (bool, np.bool_)):
        return bool(return_trajectory)
    raise TypeError("return_trajectory must be a boolean or None")


def _patch_wrapper_score(wrapper_cls: object) -> None:
    current = getattr(wrapper_cls, "score", None)
    if current is None or getattr(current, _PATCHED_SCORE_FLAG, False):
        return

    @wraps(current)
    def score(self, emissions, bin_centers, *, occupancy_s=None, candidate_indices=None, return_trajectory=None):
        return current(
            self,
            emissions,
            bin_centers,
            occupancy_s=occupancy_s,
            candidate_indices=candidate_indices,
            return_trajectory=_normalize_return_trajectory(return_trajectory),
        )

    setattr(score, _PATCHED_SCORE_FLAG, True)
    setattr(score, "__hipporeplayimm_original__", current)
    setattr(wrapper_cls, "score", score)


def _patch_current_wrapper_scores() -> None:
    from . import result_improvement_extensions as compat
    from . import reverse_models as direct

    for module in (compat, direct):
        _patch_wrapper_score(module.ReverseTimeReplayModel)
        _patch_wrapper_score(module.BidirectionalReplayModel)
        setattr(module, _PATCHED_MODULE_FLAG, True)


def apply_return_trajectory_validation_patch() -> None:
    """Install validation now and after bidirectional wrappers refresh."""

    from . import bidirectional_infinite_evidence_patch as bidirectional_patch

    current = bidirectional_patch.apply_bidirectional_infinite_evidence_patch
    if not getattr(current, _PATCHED_APPLIER_FLAG, False):

        @wraps(current)
        def apply_bidirectional_infinite_evidence_patch() -> None:
            current()
            _patch_current_wrapper_scores()

        setattr(apply_bidirectional_infinite_evidence_patch, _PATCHED_APPLIER_FLAG, True)
        setattr(apply_bidirectional_infinite_evidence_patch, "__hipporeplayimm_original__", current)
        bidirectional_patch.apply_bidirectional_infinite_evidence_patch = apply_bidirectional_infinite_evidence_patch

    _patch_current_wrapper_scores()


__all__ = ["apply_return_trajectory_validation_patch"]
