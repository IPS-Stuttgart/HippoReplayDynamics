"""Validate candidate support and preserve exact posterior support."""

from __future__ import annotations

from functools import wraps

import numpy as np
from scipy.special import logsumexp

_PATCHED_FLAG = "_candidate_active_support_validation_patch_applied"
_PAIR_POSTERIOR_PATCHED_FLAG = "_candidate_pair_posterior_exact_support_patch_applied"
_PAIR_POSTERIOR_WRAPPER_FLAG = "_candidate_pair_posterior_exact_support_wrapper"
_SPARSE_MATVEC_WRAPPER_FLAG = "_sparse_diffusion_exact_support_wrapper"


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
