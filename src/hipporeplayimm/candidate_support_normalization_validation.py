"""Validate log-row normalization used for candidate support selection.

Posterior-derived candidate beams normalize log support rows before selecting the
highest-scoring spatial bins. A row with no finite active-support likelihood has
zero probability mass and must fail fast instead of becoming NaN after
``-inf - -inf`` normalization.  Emission-derived candidate beams need the same
finite-score validation before top-k support selection, because NumPy otherwise
orders NaN/+inf values as if they were valid high-likelihood spatial bins.
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

_PATCHED_FLAG = "_candidate_support_normalization_validation_patch_applied"
_SCORE_PATCHED_FLAG = "_candidate_support_score_validation_patch_applied"


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


def _reject_boolean_mass_threshold(mass_threshold: object) -> None:
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


def apply_candidate_support_normalization_validation_patch() -> None:
    """Install finite-mass validation for posterior and emission candidate supports."""

    from . import models, state_space, state_space_model, state_space_utils

    if not getattr(state_space_model, _PATCHED_FLAG, False):
        state_space_model._normalize_log_rows = _normalize_log_rows
        setattr(state_space_model, _PATCHED_FLAG, True)

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
