"""Validate state-space candidate mass thresholds before coercion.

``StateSpaceReplayModel.candidate_indices`` historically converted
``momentum_candidate_mass_threshold`` with ``float(...)`` before the shared
probability helper saw it.  That erased boolean type information, so ``True``
was silently treated as a threshold of ``1.0`` and ``False`` as ``0.0``.  The
same malformed configuration could bypass candidate construction entirely when
callers supplied candidate support directly to ``score``.
"""

from __future__ import annotations

from functools import wraps

import numpy as np

_CANDIDATE_INDICES_PATCHED_FLAG = "_candidate_mass_threshold_indices_validation_patch_applied"
_SCORE_PATCHED_FLAG = "_candidate_mass_threshold_score_validation_patch_applied"
_ORIGINAL_ATTR = "__hipporeplayimm_original__"


def _wrapper_chain_has_marker(function: object, marker: str) -> bool:
    """Return whether a runtime-wrapper chain already contains ``marker``."""

    seen: set[int] = set()
    current = function
    while callable(current) and id(current) not in seen:
        seen.add(id(current))
        if getattr(current, marker, False):
            return True
        current = getattr(current, _ORIGINAL_ATTR, None)
    return False


def _validate_candidate_mass_threshold(config: object) -> None:
    """Reject coercive threshold types while preserving disable sentinels."""

    value = getattr(config, "momentum_candidate_mass_threshold", None)
    if value is None:
        return

    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "momentum_candidate_mass_threshold must be a real numeric scalar"
        ) from exc
    if raw.ndim != 0:
        raise TypeError(
            "momentum_candidate_mass_threshold must be a real numeric scalar"
        )
    if np.issubdtype(raw.dtype, np.bool_):
        raise TypeError(
            "momentum_candidate_mass_threshold must be a real numeric scalar, not boolean"
        )
    if raw.dtype.kind in {"S", "U"}:
        raise TypeError(
            "momentum_candidate_mass_threshold must be a real numeric scalar, not string"
        )
    if np.issubdtype(raw.dtype, np.complexfloating):
        raise TypeError(
            "momentum_candidate_mass_threshold must be a real numeric scalar, not complex"
        )

    item = raw.item()
    if isinstance(item, (bool, np.bool_)):
        raise TypeError(
            "momentum_candidate_mass_threshold must be a real numeric scalar, not boolean"
        )
    if isinstance(item, (str, bytes, np.str_, np.bytes_)):
        raise TypeError(
            "momentum_candidate_mass_threshold must be a real numeric scalar, not string"
        )
    if isinstance(item, (complex, np.complexfloating)):
        raise TypeError(
            "momentum_candidate_mass_threshold must be a real numeric scalar, not complex"
        )
    try:
        numeric = float(item)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(
            "momentum_candidate_mass_threshold must be a real numeric scalar"
        ) from exc
    if np.isfinite(numeric) and numeric > 1.0:
        raise ValueError(
            "momentum_candidate_mass_threshold must not exceed 1"
        )


def _patch_candidate_indices(state_space_model: object) -> None:
    current = state_space_model.StateSpaceReplayModel.candidate_indices
    if _wrapper_chain_has_marker(current, _CANDIDATE_INDICES_PATCHED_FLAG):
        return

    @wraps(current)
    def candidate_indices(self, emissions, bin_centers=None, valid_bin_mask=None):
        config = getattr(self, "config", None)
        if config is not None:
            _validate_candidate_mass_threshold(config)
        return current(
            self,
            emissions,
            bin_centers=bin_centers,
            valid_bin_mask=valid_bin_mask,
        )

    setattr(candidate_indices, _CANDIDATE_INDICES_PATCHED_FLAG, True)
    setattr(candidate_indices, _ORIGINAL_ATTR, current)
    state_space_model.StateSpaceReplayModel.candidate_indices = candidate_indices


def _patch_score(state_space_model: object) -> None:
    current = state_space_model.StateSpaceReplayModel.score
    if _wrapper_chain_has_marker(current, _SCORE_PATCHED_FLAG):
        return

    @wraps(current)
    def score(
        self,
        emissions,
        bin_centers,
        candidate_indices=None,
        *,
        occupancy_s=None,
        return_trajectory: bool = True,
    ):
        config = getattr(self, "config", None)
        if config is not None:
            _validate_candidate_mass_threshold(config)
        return current(
            self,
            emissions,
            bin_centers,
            candidate_indices=candidate_indices,
            occupancy_s=occupancy_s,
            return_trajectory=return_trajectory,
        )

    setattr(score, _SCORE_PATCHED_FLAG, True)
    setattr(score, _ORIGINAL_ATTR, current)
    if getattr(current, "_native_duration_occupancy_aware", False):
        setattr(score, "_native_duration_occupancy_aware", True)
    state_space_model.StateSpaceReplayModel.score = score


def apply_state_space_candidate_mass_threshold_validation_patch() -> None:
    """Install threshold validation on candidate construction and scoring."""

    from . import state_space_model

    _patch_candidate_indices(state_space_model)
    _patch_score(state_space_model)


__all__ = ["apply_state_space_candidate_mass_threshold_validation_patch"]
