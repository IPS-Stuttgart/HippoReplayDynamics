"""Validate replay-wrapper return_trajectory controls without bool() coercion."""

from __future__ import annotations

from functools import wraps

import numpy as np

_PATCHED_APPLIER_FLAG = "_return_trajectory_validation_applier_wrapped"
_PATCHED_COMPAT_SCORE_FLAG = "_return_trajectory_validation_compat_score_wrapped"
_PATCHED_SCORE_FLAG = "_return_trajectory_validation_score_wrapped"
_PATCHED_MODULE_FLAG = "_return_trajectory_validation_patch_applied"


def _normalize_return_trajectory(value: object) -> bool | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    raise TypeError("return_trajectory must be a boolean or None")


def _patch_score_replay_model_compat(module: object) -> None:
    original = getattr(module, "score_replay_model_compat", None)
    if original is None or getattr(original, _PATCHED_COMPAT_SCORE_FLAG, False):
        return

    @wraps(original)
    def score_replay_model_compat(model, emissions, bin_centers, *, occupancy_s=None, candidate_indices=None, return_trajectory=None):
        return original(
            model,
            emissions,
            bin_centers,
            occupancy_s=occupancy_s,
            candidate_indices=candidate_indices,
            return_trajectory=_normalize_return_trajectory(return_trajectory),
        )

    setattr(score_replay_model_compat, _PATCHED_COMPAT_SCORE_FLAG, True)
    setattr(module, "score_replay_model_compat", score_replay_model_compat)


def _patch_wrapper_score(wrapper_cls: object) -> None:
    original = getattr(wrapper_cls, "score", None)
    if original is None or getattr(original, _PATCHED_SCORE_FLAG, False):
        return

    @wraps(original)
    def score(self, emissions, bin_centers, *, occupancy_s=None, candidate_indices=None, return_trajectory=None):
        return original(
            self,
            emissions,
            bin_centers,
            occupancy_s=occupancy_s,
            candidate_indices=candidate_indices,
            return_trajectory=_normalize_return_trajectory(return_trajectory),
        )

    setattr(score, _PATCHED_SCORE_FLAG, True)
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
    """Install validation now and after bidirectional wrappers refresh."""

    from . import bidirectional_infinite_evidence_patch as bidirectional_patch

    original_applier = bidirectional_patch.apply_bidirectional_infinite_evidence_patch
    if not getattr(original_applier, _PATCHED_APPLIER_FLAG, False):

        @wraps(original_applier)
        def apply_bidirectional_infinite_evidence_patch() -> None:
            original_applier()
            _patch_current_wrapper_scores()

        setattr(apply_bidirectional_infinite_evidence_patch, _PATCHED_APPLIER_FLAG, True)
        bidirectional_patch.apply_bidirectional_infinite_evidence_patch = apply_bidirectional_infinite_evidence_patch

    _patch_current_wrapper_scores()


__all__ = ["apply_return_trajectory_validation_patch"]
