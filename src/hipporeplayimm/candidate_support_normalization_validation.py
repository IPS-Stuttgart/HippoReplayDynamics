"""Validate log-row normalization used for candidate support selection.

Posterior-derived candidate beams normalize log support rows before selecting the
highest-scoring spatial bins. A row with no finite active-support likelihood has
zero probability mass and must fail fast instead of becoming NaN after
``-inf - -inf`` normalization.  Emission-derived candidate beams need the same
finite-score validation before top-k support selection, because NumPy otherwise
orders NaN/+inf values as if they were valid high-likelihood spatial bins.

Candidate evidence-support diagnostics also share the same valid-bin mask
contract.  They must reject malformed textual, complex, or non-binary numeric
masks instead of treating arbitrary truthy values as valid spatial bins.

State-space adaptive candidate beams must validate the raw mass-threshold config
before ``StateSpaceReplayModel.candidate_indices`` coerces it through ``float``;
otherwise boolean values such as ``True`` are silently interpreted as a full-mass
threshold of ``1.0``.
"""

from __future__ import annotations

from functools import wraps

import numpy as np
from scipy.special import logsumexp

_PATCHED_FLAG = "_candidate_support_normalization_validation_patch_applied"
_SCORE_PATCHED_FLAG = "_candidate_support_score_validation_patch_applied"
_EVIDENCE_MASK_PATCHED_FLAG = "_candidate_evidence_mask_validation_patch_applied"
_CANDIDATE_INDICES_PATCHED_FLAG = "_candidate_mass_threshold_candidate_indices_validation_patch_applied"
_EVIDENCE_MASK_WRAPPER_MARKER = "_candidate_evidence_mask_validation_wrapper"
_CANDIDATE_INDICES_WRAPPER_MARKER = "_candidate_mass_threshold_candidate_indices_validation_wrapper"
_ORIGINAL_ATTR = "__hipporeplayimm_original__"


def _is_boolean_scalar(value: object) -> bool:
    """Return True for Python, NumPy, and object-wrapped boolean scalars."""

    if isinstance(value, (bool, np.bool_)):
        return True
    arr = np.asarray(value)
    if arr.ndim != 0:
        return False
    if np.issubdtype(arr.dtype, np.bool_):
        return True
    if arr.dtype == object:
        try:
            return isinstance(arr.item(), (bool, np.bool_))
        except ValueError:
            return False
    return False


def _reject_array_shaped_mass_threshold(mass_threshold: object) -> None:
    """Reject values that NumPy/Python might coerce from an array to a scalar."""

    try:
        arr = np.asarray(mass_threshold)
    except ValueError as exc:
        raise TypeError("mass_threshold must be a numeric scalar") from exc
    if arr.ndim != 0:
        raise TypeError("mass_threshold must be a numeric scalar")


def _reject_boolean_mass_threshold(mass_threshold: object) -> None:
    _reject_array_shaped_mass_threshold(mass_threshold)
    if _is_boolean_scalar(mass_threshold):
        raise TypeError("mass_threshold must be numeric, not boolean")


def _normalize_log_rows(values: np.ndarray) -> np.ndarray:
    """Normalize finite-mass log rows and reject empty active support rows."""

    out = np.asarray(values, dtype=float).copy()
    if out.ndim != 2:
        raise ValueError("values must be two-dimensional")
    row_norm = logsumexp(out, axis=1)
    invalid = ~np.isfinite(row_norm)
    if np.any(invalid):
        first = int(np.flatnonzero(invalid)[0])
        raise ValueError(f"log support row {first} must contain at least one finite value")
    out -= row_norm[:, None]
    return out


def _candidate_support_scores(log_emission: np.ndarray) -> np.ndarray:
    """Return a finite-mass one-dimensional log-score vector for support selection."""

    scores = np.asarray(log_emission, dtype=float)
    if scores.ndim != 1:
        raise ValueError("log_emission must be one-dimensional")
    if scores.size == 0:
        raise ValueError("log_emission must contain at least one spatial bin")
    if np.any(scores == np.inf):
        raise ValueError("log_emission must not contain +inf")
    scores = np.where(np.isnan(scores), -np.inf, scores)
    if not np.any(np.isfinite(scores)):
        raise ValueError("log_emission must contain at least one finite value")
    return scores


def _candidate_evidence_mask_patch_current(state_space_model: object) -> bool:
    return bool(
        getattr(
            getattr(state_space_model, "_full_candidate_index_set", None),
            _EVIDENCE_MASK_WRAPPER_MARKER,
            False,
        )
    )


def _candidate_indices_patch_current(state_space_model: object) -> bool:
    return bool(
        getattr(
            getattr(state_space_model.StateSpaceReplayModel, "candidate_indices", None),
            _CANDIDATE_INDICES_WRAPPER_MARKER,
            False,
        )
    )


def apply_candidate_support_normalization_validation_patch() -> None:
    """Install finite-mass validation for posterior and emission candidate supports."""

    from . import models, state_space, state_space_model, state_space_utils

    if not getattr(state_space_model, _PATCHED_FLAG, False):
        state_space_model._normalize_log_rows = _normalize_log_rows
        setattr(state_space_model, _PATCHED_FLAG, True)

    if not _candidate_evidence_mask_patch_current(state_space_model):
        original_full_candidate_index_set = state_space_model._full_candidate_index_set

        def _full_candidate_index_set(n_bins: int, valid_bin_mask: np.ndarray | None) -> np.ndarray:
            if valid_bin_mask is None:
                return original_full_candidate_index_set(n_bins, None)
            valid_mask = state_space_utils._coerce_valid_bin_mask(valid_bin_mask, n_bins)
            assert valid_mask is not None
            return original_full_candidate_index_set(n_bins, valid_mask)

        setattr(_full_candidate_index_set, _EVIDENCE_MASK_WRAPPER_MARKER, True)
        setattr(_full_candidate_index_set, _ORIGINAL_ATTR, original_full_candidate_index_set)
        state_space_model._full_candidate_index_set = _full_candidate_index_set
    setattr(state_space_model, _EVIDENCE_MASK_PATCHED_FLAG, True)

    if not _candidate_indices_patch_current(state_space_model):
        original_candidate_indices = state_space_model.StateSpaceReplayModel.candidate_indices

        @wraps(original_candidate_indices)
        def candidate_indices(self, emissions, bin_centers=None, valid_bin_mask=None):
            config = getattr(self, "config", None)
            if config is not None:
                mass_threshold = getattr(config, "momentum_candidate_mass_threshold", None)
                if mass_threshold is not None:
                    _reject_boolean_mass_threshold(mass_threshold)
            return original_candidate_indices(
                self,
                emissions,
                bin_centers=bin_centers,
                valid_bin_mask=valid_bin_mask,
            )

        setattr(candidate_indices, _CANDIDATE_INDICES_WRAPPER_MARKER, True)
        setattr(candidate_indices, _ORIGINAL_ATTR, original_candidate_indices)
        state_space_model.StateSpaceReplayModel.candidate_indices = candidate_indices
    setattr(state_space_model, _CANDIDATE_INDICES_PATCHED_FLAG, True)

    if not getattr(state_space_utils, _SCORE_PATCHED_FLAG, False):
        original_top_candidate_indices = state_space_utils._top_candidate_indices
        original_mass_retaining_candidate_indices = state_space_utils._mass_retaining_candidate_indices

        def _top_candidate_indices(log_emission: np.ndarray, top_k: int) -> np.ndarray:
            return original_top_candidate_indices(_candidate_support_scores(log_emission), top_k)

        def _mass_retaining_candidate_indices(
            log_emission: np.ndarray,
            mass_threshold: float | None = None,
            *,
            top_k: int | None = None,
            min_k: int = 1,
            max_k: int = 0,
        ) -> np.ndarray:
            if mass_threshold is not None:
                _reject_boolean_mass_threshold(mass_threshold)
            return original_mass_retaining_candidate_indices(
                _candidate_support_scores(log_emission),
                mass_threshold,
                top_k=top_k,
                min_k=min_k,
                max_k=max_k,
            )

        for module in (state_space_utils, state_space, state_space_model):
            if getattr(module, "_top_candidate_indices", None) is original_top_candidate_indices:
                module._top_candidate_indices = _top_candidate_indices
            if getattr(module, "_mass_retaining_candidate_indices", None) is original_mass_retaining_candidate_indices:
                module._mass_retaining_candidate_indices = _mass_retaining_candidate_indices
        setattr(state_space_utils, _SCORE_PATCHED_FLAG, True)

    if not getattr(models, _SCORE_PATCHED_FLAG, False):
        original_model_top_candidate_indices = models._top_candidate_indices

        def _model_top_candidate_indices(log_emission: np.ndarray, top_k: int) -> np.ndarray:
            return original_model_top_candidate_indices(_candidate_support_scores(log_emission), top_k)

        models._top_candidate_indices = _model_top_candidate_indices
        setattr(models, _SCORE_PATCHED_FLAG, True)


__all__ = ["apply_candidate_support_normalization_validation_patch"]
