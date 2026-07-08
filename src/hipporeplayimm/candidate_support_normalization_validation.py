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
_RESTRICT_CANDIDATES_PATCHED_FLAG = "_candidate_restriction_active_support_validation_patch_applied"
_EVIDENCE_MASK_WRAPPER_MARKER = "_candidate_evidence_mask_validation_wrapper"
_CANDIDATE_INDICES_WRAPPER_MARKER = "_candidate_mass_threshold_candidate_indices_validation_wrapper"
_TOP_CANDIDATE_WRAPPER_MARKER = "_candidate_support_top_candidate_scores_wrapper"
_MASS_RETAINING_WRAPPER_MARKER = "_candidate_support_mass_retaining_scores_wrapper"
_RESTRICT_CANDIDATES_WRAPPER_MARKER = "_candidate_restriction_active_support_wrapper"
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


def _validate_restricted_candidate_active_support(
    log_likelihood: np.ndarray,
    valid_bin_mask: np.ndarray | None,
    *,
    state_space_utils,
) -> None:
    """Reject candidate restriction when the active support has no finite score."""

    if valid_bin_mask is None:
        return
    values = np.asarray(log_likelihood, dtype=float)
    if values.ndim != 2:
        return
    valid_mask = state_space_utils._coerce_valid_bin_mask(
        valid_bin_mask,
        values.shape[1],
    )
    if valid_mask is None:
        return
    finite_rows = np.any(np.isfinite(values) & valid_mask[None, :], axis=1)
    if not np.all(finite_rows):
        first = int(np.flatnonzero(~finite_rows)[0])
        raise ValueError(
            f"row {first} must contain at least one finite value on the active support"
        )


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


def _is_top_candidate_scores_wrapper(func: object) -> bool:
    return bool(getattr(func, _TOP_CANDIDATE_WRAPPER_MARKER, False))


def _is_mass_retaining_scores_wrapper(func: object) -> bool:
    return bool(getattr(func, _MASS_RETAINING_WRAPPER_MARKER, False))


def _is_restrict_candidates_wrapper(func: object) -> bool:
    return bool(getattr(func, _RESTRICT_CANDIDATES_WRAPPER_MARKER, False))


def _wrap_top_candidate_indices(func):
    @wraps(func)
    def _top_candidate_indices(log_emission: np.ndarray, top_k: int) -> np.ndarray:
        return func(_candidate_support_scores(log_emission), top_k)

    setattr(_top_candidate_indices, _TOP_CANDIDATE_WRAPPER_MARKER, True)
    setattr(_top_candidate_indices, _ORIGINAL_ATTR, func)
    return _top_candidate_indices


def _wrap_mass_retaining_candidate_indices(func):
    @wraps(func)
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
        return func(
            _candidate_support_scores(log_emission),
            mass_threshold,
            top_k=top_k,
            min_k=min_k,
            max_k=max_k,
        )

    setattr(_mass_retaining_candidate_indices, _MASS_RETAINING_WRAPPER_MARKER, True)
    setattr(_mass_retaining_candidate_indices, _ORIGINAL_ATTR, func)
    return _mass_retaining_candidate_indices


def _wrap_restrict_candidates_to_valid_bins(func, *, state_space_utils):
    @wraps(func)
    def _restrict_candidates_to_valid_bins(
        candidates: list[np.ndarray],
        log_likelihood: np.ndarray,
        valid_bin_mask: np.ndarray | None,
    ) -> list[np.ndarray]:
        _validate_restricted_candidate_active_support(
            log_likelihood,
            valid_bin_mask,
            state_space_utils=state_space_utils,
        )
        return func(candidates, log_likelihood, valid_bin_mask)

    setattr(_restrict_candidates_to_valid_bins, _RESTRICT_CANDIDATES_WRAPPER_MARKER, True)
    setattr(_restrict_candidates_to_valid_bins, _ORIGINAL_ATTR, func)
    return _restrict_candidates_to_valid_bins


def _refresh_candidate_score_helpers(*, state_space_utils, state_space, state_space_model, models) -> None:
    """Install finite-score guards on every current candidate-helper alias."""

    for module in (state_space_utils, state_space, state_space_model):
        current_top_candidate_indices = getattr(module, "_top_candidate_indices", None)
        if current_top_candidate_indices is not None and not _is_top_candidate_scores_wrapper(current_top_candidate_indices):
            module._top_candidate_indices = _wrap_top_candidate_indices(current_top_candidate_indices)

        current_mass_retaining_candidate_indices = getattr(module, "_mass_retaining_candidate_indices", None)
        if current_mass_retaining_candidate_indices is not None and not _is_mass_retaining_scores_wrapper(current_mass_retaining_candidate_indices):
            module._mass_retaining_candidate_indices = _wrap_mass_retaining_candidate_indices(current_mass_retaining_candidate_indices)

        current_restrict_candidates = getattr(module, "_restrict_candidates_to_valid_bins", None)
        if current_restrict_candidates is not None and not _is_restrict_candidates_wrapper(current_restrict_candidates):
            module._restrict_candidates_to_valid_bins = _wrap_restrict_candidates_to_valid_bins(
                current_restrict_candidates,
                state_space_utils=state_space_utils,
            )

    current_model_top_candidate_indices = getattr(models, "_top_candidate_indices", None)
    if current_model_top_candidate_indices is not None and not _is_top_candidate_scores_wrapper(current_model_top_candidate_indices):
        models._top_candidate_indices = _wrap_top_candidate_indices(current_model_top_candidate_indices)

    setattr(state_space_utils, _SCORE_PATCHED_FLAG, True)
    setattr(state_space_utils, _RESTRICT_CANDIDATES_PATCHED_FLAG, True)
    setattr(models, _SCORE_PATCHED_FLAG, True)


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

    _refresh_candidate_score_helpers(
        state_space_utils=state_space_utils,
        state_space=state_space,
        state_space_model=state_space_model,
        models=models,
    )


__all__ = ["apply_candidate_support_normalization_validation_patch"]
