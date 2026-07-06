"""Runtime validation for candidate-support integer count parameters."""

from __future__ import annotations

from functools import wraps

import numpy as np

_PATCH_ATTR = "_candidate_integer_count_validation_patch"
_ORIGINAL_ATTR = "_candidate_integer_count_validation_original"


def _coerce_optional_count(name: str, value: object) -> int | None:
    if value is None:
        return None
    from .state_space_utils import _coerce_integer_count

    return _coerce_integer_count(name, value)


def apply_candidate_integer_count_validation_patch() -> None:
    """Reject float-like candidate-count parameters before support selection."""

    from . import state_space_utils
    from . import state_space

    current_top = state_space_utils._top_candidate_indices
    if not getattr(current_top, _PATCH_ATTR, False):

        @wraps(current_top)
        def top_candidate_indices(log_emission: np.ndarray, top_k: int) -> np.ndarray:
            return current_top(log_emission, _coerce_optional_count("top_k", top_k))

        setattr(top_candidate_indices, _PATCH_ATTR, True)
        setattr(top_candidate_indices, _ORIGINAL_ATTR, current_top)
        state_space_utils._top_candidate_indices = top_candidate_indices
        state_space._top_candidate_indices = top_candidate_indices

    current_mass = state_space_utils._mass_retaining_candidate_indices
    if not getattr(current_mass, _PATCH_ATTR, False):

        @wraps(current_mass)
        def mass_retaining_candidate_indices(
            log_emission: np.ndarray,
            mass_threshold: float | None = None,
            *,
            top_k: int | None = None,
            min_k: int = 1,
            max_k: int = 0,
        ) -> np.ndarray:
            return current_mass(
                log_emission,
                mass_threshold,
                top_k=_coerce_optional_count("top_k", top_k),
                min_k=_coerce_optional_count("min_k", min_k),
                max_k=_coerce_optional_count("max_k", max_k),
            )

        setattr(mass_retaining_candidate_indices, _PATCH_ATTR, True)
        setattr(mass_retaining_candidate_indices, _ORIGINAL_ATTR, current_mass)
        state_space_utils._mass_retaining_candidate_indices = mass_retaining_candidate_indices
        state_space._mass_retaining_candidate_indices = mass_retaining_candidate_indices
