"""Validate candidate support and preserve exact posterior support."""

from __future__ import annotations

import sys
from functools import wraps

import numpy as np
from scipy.sparse import csr_matrix
from scipy.special import logsumexp

_PATCHED_FLAG = "_candidate_active_support_validation_patch_applied"
_PAIR_POSTERIOR_PATCHED_FLAG = "_candidate_pair_posterior_exact_support_patch_applied"
_PAIR_POSTERIOR_WRAPPER_FLAG = "_candidate_pair_posterior_exact_support_wrapper"
_SPARSE_MATVEC_WRAPPER_FLAG = "_sparse_diffusion_exact_support_wrapper"
_GAUSSIAN_TRANSITION_WRAPPER_FLAG = "_stable_gaussian_transition_weights_wrapper"
_SPARSE_GAUSSIAN_ROW_WRAPPER_FLAG = "_stable_sparse_gaussian_row_weights_wrapper"


def _validate_active_support_rows(values: np.ndarray) -> None:
    rows = np.asarray(values, dtype=float)
    if rows.ndim != 2:
        raise ValueError("log_likelihood must be two-dimensional")
    finite_rows = np.any(np.isfinite(rows), axis=1)
    if not np.all(finite_rows):
        row = int(np.flatnonzero(~finite_rows)[0])
        raise ValueError(f"row {row} must contain at least one finite value on the active support")


def apply_candidate_active_support_validation_patch() -> None:
    """Install active-support validation and exact candidate posterior support."""

    from . import state_space_model

    current = state_space_model._masked_candidate_support_log_values
    if not getattr(current, _PATCHED_FLAG, False):

        @wraps(current)
        def masked_candidate_support_log_values(log_likelihood, valid_bin_mask):
            masked = current(log_likelihood, valid_bin_mask)
            _validate_active_support_rows(masked)
            return masked

        setattr(masked_candidate_support_log_values, _PATCHED_FLAG, True)
        setattr(masked_candidate_support_log_values, "__hipporeplayimm_original__", current)
        state_space_model._masked_candidate_support_log_values = masked_candidate_support_log_values

    _patch_candidate_pair_posteriors()
    _patch_sparse_diffusion_exact_support()
    _patch_stable_gaussian_transition_weights()


def _patch_candidate_pair_posteriors() -> None:
    """Represent bins outside a candidate marginal with exact zero probability."""

    from . import models

    current_terminal = models._pair_terminal_posterior
    if not getattr(current_terminal, _PAIR_POSTERIOR_WRAPPER_FLAG, False):

        @wraps(current_terminal)
        def pair_terminal_posterior(log_pair_or_modes, current_indices, n_bins):
            posterior = current_terminal(log_pair_or_modes, current_indices, n_bins)
            return _restrict_log_posterior_to_candidates(
                posterior,
                current_indices,
                n_bins,
            )

        setattr(pair_terminal_posterior, _PAIR_POSTERIOR_WRAPPER_FLAG, True)
        setattr(pair_terminal_posterior, "__hipporeplayimm_original__", current_terminal)
        models._pair_terminal_posterior = pair_terminal_posterior

    current_previous = models._pair_previous_posterior
    if not getattr(current_previous, _PAIR_POSTERIOR_WRAPPER_FLAG, False):

        @wraps(current_previous)
        def pair_previous_posterior(log_pair_or_modes, previous_indices, n_bins):
            posterior = current_previous(log_pair_or_modes, previous_indices, n_bins)
            return _restrict_log_posterior_to_candidates(
                posterior,
                previous_indices,
                n_bins,
            )

        setattr(pair_previous_posterior, _PAIR_POSTERIOR_WRAPPER_FLAG, True)
        setattr(pair_previous_posterior, "__hipporeplayimm_original__", current_previous)
        models._pair_previous_posterior = pair_previous_posterior

    setattr(models, _PAIR_POSTERIOR_PATCHED_FLAG, True)


def _patch_sparse_diffusion_exact_support() -> None:
    """Keep unreachable sparse-diffusion states at exact zero probability."""

    from . import models

    current = models._log_sparse_matvec
    if getattr(current, _SPARSE_MATVEC_WRAPPER_FLAG, False):
        return

    @wraps(current)
    def log_sparse_matvec(log_alpha, transition):
        log_alpha = np.asarray(log_alpha, dtype=float)
        result = np.full(log_alpha.shape, -np.inf, dtype=float)
        for src, (dst_indices, log_weights) in enumerate(transition):
            values = log_alpha[src] + np.asarray(log_weights, dtype=float)
            for dst, value in zip(
                np.asarray(dst_indices, dtype=int),
                values,
                strict=True,
            ):
                result[int(dst)] = np.logaddexp(result[int(dst)], value)
        return result

    setattr(log_sparse_matvec, _SPARSE_MATVEC_WRAPPER_FLAG, True)
    setattr(log_sparse_matvec, "__hipporeplayimm_original__", current)
    models._log_sparse_matvec = log_sparse_matvec


def _stable_gaussian_weights(dist2: np.ndarray, sigma_cm: float) -> np.ndarray:
    """Normalize Gaussian weights without underflowing every candidate to zero."""

    distances = np.asarray(dist2, dtype=float)
    if distances.ndim != 1 or distances.size == 0:
        raise ValueError("dist2 must be a nonempty one-dimensional array")
    minimum = float(np.min(distances))
    nearest = distances == minimum
    sigma = abs(float(sigma_cm))
    if not np.isfinite(minimum) or not np.isfinite(sigma) or sigma <= 0.0:
        return nearest.astype(float) / float(np.sum(nearest))

    shifted = np.maximum(distances - minimum, 0.0)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        standardized = np.sqrt(shifted) / sigma
        weights = np.exp(-0.5 * standardized * standardized)
    total = float(weights.sum())
    if total <= 0.0 or not np.isfinite(total):
        return nearest.astype(float) / float(np.sum(nearest))
    return weights / total


def _patch_stable_gaussian_transition_weights() -> None:
    """Preserve relative Gaussian transition mass when raw kernels underflow."""

    from . import state_space_sparse_momentum, state_space_utils

    current_dense = state_space_utils._gaussian_transition_matrix
    if not getattr(current_dense, _GAUSSIAN_TRANSITION_WRAPPER_FLAG, False):

        @wraps(current_dense)
        def gaussian_transition_matrix(
            bin_centers,
            sigma_cm,
            max_step_sigma,
            valid_bin_mask=None,
        ):
            sigma = float(sigma_cm)
            max_step = float(max_step_sigma)
            if not np.isfinite(sigma) or sigma <= 0.0:
                raise ValueError("sigma_cm must be finite and positive")
            if not np.isfinite(max_step) or max_step <= 0.0:
                raise ValueError("max_step_sigma must be finite and positive")
            centers = state_space_utils._as_finite_2d_points(bin_centers, "bin_centers")
            n_bins = centers.shape[0]
            valid_mask = state_space_utils._coerce_valid_bin_mask(valid_bin_mask, n_bins)
            allowed = np.arange(n_bins, dtype=int) if valid_mask is None else np.flatnonzero(valid_mask)
            rows: list[int] = []
            cols: list[int] = []
            data: list[float] = []
            for src, center in enumerate(centers):
                with np.errstate(over="ignore", invalid="ignore"):
                    standardized_delta = (centers - center[None, :]) / sigma
                    standardized_distance = np.hypot.reduce(standardized_delta, axis=1)
                keep = standardized_distance <= max_step
                if valid_mask is not None:
                    keep &= valid_mask
                if not np.any(keep):
                    keep[int(allowed[int(np.argmin(standardized_distance[allowed]))])] = True
                dst = np.flatnonzero(keep)
                with np.errstate(over="ignore", invalid="ignore"):
                    standardized_dist2 = np.square(standardized_distance[dst])
                weights = _stable_gaussian_weights(standardized_dist2, 1.0)
                rows.extend(int(index) for index in dst)
                cols.extend([src] * len(dst))
                data.extend(float(value) for value in weights)
            return csr_matrix((data, (rows, cols)), shape=(n_bins, n_bins))

        setattr(gaussian_transition_matrix, _GAUSSIAN_TRANSITION_WRAPPER_FLAG, True)
        setattr(gaussian_transition_matrix, "__hipporeplayimm_original__", current_dense)
        state_space_utils._gaussian_transition_matrix = gaussian_transition_matrix
        _synchronize_transition_aliases(
            "_gaussian_transition_matrix",
            current_dense,
            gaussian_transition_matrix,
        )

    current_sparse = state_space_sparse_momentum._finite_gaussian_row
    if not getattr(current_sparse, _SPARSE_GAUSSIAN_ROW_WRAPPER_FLAG, False):

        @wraps(current_sparse)
        def finite_gaussian_row(
            centers,
            valid_indices,
            tree,
            predicted,
            *,
            sigma_cm,
            max_step_sigma,
        ):
            sigma = float(sigma_cm)
            if not np.isfinite(sigma) or sigma <= 0.0:
                raise ValueError("sigma_cm must be finite and positive")
            points = np.asarray(centers, dtype=float)
            prediction = np.asarray(predicted, dtype=float).reshape(points.shape[1])
            radius = max(sigma * float(max_step_sigma), np.finfo(float).eps)
            local = tree.query_ball_point(prediction, radius)
            if len(local) == 0:
                _, nearest = tree.query(prediction, k=1)
                local = [int(nearest)]
            valid = np.asarray(valid_indices, dtype=int)
            dst = valid[np.asarray(local, dtype=int)]
            dist2 = np.sum((points[dst] - prediction[None, :]) ** 2, axis=1)
            weights = _stable_gaussian_weights(dist2, sigma)
            return dst.astype(int), weights

        setattr(finite_gaussian_row, _SPARSE_GAUSSIAN_ROW_WRAPPER_FLAG, True)
        setattr(finite_gaussian_row, "__hipporeplayimm_original__", current_sparse)
        state_space_sparse_momentum._finite_gaussian_row = finite_gaussian_row
        _synchronize_transition_aliases(
            "_finite_gaussian_row",
            current_sparse,
            finite_gaussian_row,
        )


def _synchronize_transition_aliases(name: str, original: object, replacement: object) -> None:
    """Refresh package-local imports of a patched transition helper."""

    for module in list(sys.modules.values()):
        module_name = getattr(module, "__name__", "")
        if module_name.startswith("hipporeplayimm") and getattr(module, name, None) is original:
            setattr(module, name, replacement)


def _restrict_log_posterior_to_candidates(
    log_posterior: np.ndarray,
    candidate_indices: np.ndarray,
    n_bins: int,
) -> np.ndarray:
    """Set excluded bins to ``-inf`` and renormalize the retained support."""

    out = np.asarray(log_posterior, dtype=float).copy()
    active = np.zeros(int(n_bins), dtype=bool)
    active[np.asarray(candidate_indices, dtype=int)] = True
    out[~active] = -np.inf
    normalizer = logsumexp(out[active])
    if np.isfinite(normalizer):
        out[active] -= normalizer
    return out


__all__ = ["apply_candidate_active_support_validation_patch"]
