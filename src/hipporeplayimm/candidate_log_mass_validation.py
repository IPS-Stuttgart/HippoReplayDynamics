"""Validate candidate-path retained emission masses before scoring.

Candidate-pruned path recursions normalize posterior marginals after retaining
only a spatial support subset. If a retained support has no finite emission mass,
the path evidence is zero and the posterior cannot be normalized. Guard that
case before dynamic-program arrays can propagate NaN or LOG_ZERO-normalized
fallbacks.
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

_PATCHED_FLAG = "_candidate_log_mass_validation_patch_applied"


def _candidate_log_masses(log_likelihood: np.ndarray, candidates: list[np.ndarray]) -> list[float]:
    """Return finite retained emission masses for candidate supports.

    The retained-mass diagnostic is also an early consistency check for
    candidate-pruned recursions: every emission row must contain finite mass on
    the full support and on the supplied candidate support.
    """

    values = np.asarray(log_likelihood, dtype=float)
    if values.ndim != 2:
        raise ValueError("log_likelihood must be two-dimensional")
    if np.any(np.isnan(values)) or np.any(values == np.inf):
        raise ValueError("log_likelihood must not contain NaN or +inf")
    if len(candidates) != values.shape[0]:
        raise ValueError("candidate_indices must contain one array per emission time bin")

    masses: list[float] = []
    for time_index, curr in enumerate(candidates):
        current = np.asarray(curr)
        if current.ndim != 1:
            raise ValueError(f"candidate_indices[{time_index}] must be one-dimensional")
        if current.size == 0:
            raise ValueError(f"candidate_indices[{time_index}] must not be empty")
        if not np.issubdtype(current.dtype, np.integer):
            raise TypeError(f"candidate_indices[{time_index}] must contain integer bin indices")
        current = current.astype(np.intp, copy=False)
        if np.any((current < 0) | (current >= values.shape[1])):
            raise ValueError(f"candidate_indices[{time_index}] contains an out-of-range bin")
        if np.unique(current).size != current.size:
            raise ValueError(f"candidate_indices[{time_index}] contains duplicate bins")

        row = values[time_index]
        total = logsumexp(row)
        if not np.isfinite(total):
            raise ValueError(f"log_likelihood row {time_index} must contain at least one finite spatial-bin value")

        selected = logsumexp(row[current])
        if not np.isfinite(selected):
            raise ValueError(f"candidate_indices[{time_index}] select no finite likelihood mass")

        masses.append(float(selected - total))
    return masses


def _candidate_log_masses_for_model(emissions, candidates: list[np.ndarray]) -> list[float]:
    """Compatibility wrapper for legacy CandidateKinematicModel helpers."""

    return _candidate_log_masses(emissions.log_likelihood, candidates)


def _current_patch_installed(
    models,
    state_space,
    state_space_candidates,
    state_space_candidates_momentum,
    state_space_utils,
) -> bool:
    return (
        getattr(state_space, _PATCHED_FLAG, False)
        and getattr(state_space_utils, "_candidate_log_masses", None) is _candidate_log_masses
        and getattr(state_space, "_candidate_log_masses", None) is _candidate_log_masses
        and getattr(state_space_candidates, "_candidate_log_masses", None) is _candidate_log_masses
        and getattr(state_space_candidates_momentum, "_candidate_log_masses", None) is _candidate_log_masses
        and getattr(models, "_candidate_log_masses", None) is _candidate_log_masses_for_model
    )


def apply_candidate_log_mass_validation_patch() -> None:
    """Install finite retained-mass validation on candidate-pruned scorers."""

    from . import models
    from . import state_space
    from . import state_space_candidates
    from . import state_space_candidates_momentum
    from . import state_space_utils

    if _current_patch_installed(
        models,
        state_space,
        state_space_candidates,
        state_space_candidates_momentum,
        state_space_utils,
    ):
        return

    state_space_utils._candidate_log_masses = _candidate_log_masses
    state_space._candidate_log_masses = _candidate_log_masses
    state_space_candidates._candidate_log_masses = _candidate_log_masses
    state_space_candidates_momentum._candidate_log_masses = _candidate_log_masses
    models._candidate_log_masses = _candidate_log_masses_for_model
    setattr(state_space, _PATCHED_FLAG, True)


__all__ = ["apply_candidate_log_mass_validation_patch"]
