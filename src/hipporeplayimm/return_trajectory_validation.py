"""Validate replay-wrapper return_trajectory controls without bool() coercion."""

from __future__ import annotations

from functools import wraps

import numpy as np

_PATCHED_APPLIER_FLAG = "_return_trajectory_validation_applier_wrapped"
_PATCHED_WRAPPER_APPLIER_FLAG = "_return_trajectory_validation_wrapper_applier_wrapped"
_PATCHED_COMPAT_FLAG = "_return_trajectory_validation_score_replay_model_compat_wrapped"
_PATCHED_SCORE_FLAG = "_return_trajectory_validation_score_wrapped"
_PATCHED_MODULE_FLAG = "_return_trajectory_validation_patch_applied"


def _normalize_return_trajectory(return_trajectory: object) -> bool | None:
    if return_trajectory is None:
        return None
    if isinstance(return_trajectory, (bool, np.bool_)):
        return bool(return_trajectory)
    raise TypeError("return_trajectory must be a boolean or None")


def _patch_score_replay_model_compat(module: object) -> None:
    current = getattr(module, "score_replay_model_compat", None)
    if current is None or getattr(current, _PATCHED_COMPAT_FLAG, False):
        return

    @wraps(current)
    def score_replay_model_compat(model, emissions, bin_centers, *, occupancy_s=None, candidate_indices=None, return_trajectory=None):
        return current(
            model,
            emissions,
            bin_centers,
            occupancy_s=occupancy_s,
            candidate_indices=candidate_indices,
            return_trajectory=_normalize_return_trajectory(return_trajectory),
        )

    setattr(score_replay_model_compat, _PATCHED_COMPAT_FLAG, True)
    setattr(score_replay_model_compat, "__hipporeplayimm_original__", current)
    setattr(module, "score_replay_model_compat", score_replay_model_compat)


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

    _patch_score_replay_model_compat(compat)
    for module in (compat, direct):
        _patch_wrapper_score(module.ReverseTimeReplayModel)
        _patch_wrapper_score(module.BidirectionalReplayModel)
        setattr(module, _PATCHED_MODULE_FLAG, True)


def apply_return_trajectory_validation_patch() -> None:
    """Install validation now and after replay wrapper refreshes."""

    from . import bidirectional_infinite_evidence_patch as bidirectional_patch
    from . import wrapper_return_trajectory as wrapper_patch

    current_wrapper_patch = wrapper_patch.apply_wrapper_return_trajectory_patch
    if not getattr(current_wrapper_patch, _PATCHED_WRAPPER_APPLIER_FLAG, False):

        @wraps(current_wrapper_patch)
        def apply_wrapper_return_trajectory_patch() -> None:
            current_wrapper_patch()
            _patch_current_wrapper_scores()

        setattr(apply_wrapper_return_trajectory_patch, _PATCHED_WRAPPER_APPLIER_FLAG, True)
        setattr(apply_wrapper_return_trajectory_patch, "__hipporeplayimm_original__", current_wrapper_patch)
        wrapper_patch.apply_wrapper_return_trajectory_patch = apply_wrapper_return_trajectory_patch

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
