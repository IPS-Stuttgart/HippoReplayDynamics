"""Runtime guards for duration-aware occupancy helper edge cases."""

from __future__ import annotations

from dataclasses import replace
from functools import wraps
from typing import Any

import numpy as np


def apply_duration_occupancy_guard_patch() -> None:
    """Install small validation and metadata-isolation guards."""

    from . import duration_occupancy as duration

    if not getattr(duration, "_candidate_selection_metadata_guard_applied", False):
        original_candidate_selection = duration._candidate_selection_emissions

        @wraps(original_candidate_selection)
        def candidate_selection_emissions_with_metadata_copy(
            emissions: Any,
            valid_bin_mask: Any,
        ) -> Any:
            restricted = original_candidate_selection(emissions, valid_bin_mask)
            if restricted is emissions:
                return restricted
            metadata = dict(getattr(restricted, "metadata", {}))
            return replace(restricted, metadata=metadata)

        duration._candidate_selection_emissions = candidate_selection_emissions_with_metadata_copy
        duration._candidate_selection_metadata_guard_applied = True

    if not getattr(duration, "_uniform_probability_size_guard_applied", False):
        original_uniform_probabilities = duration._uniform_probabilities

        @wraps(original_uniform_probabilities)
        def uniform_probabilities_with_size_guard(
            n_bins: int,
            valid_bin_mask: Any = None,
        ) -> np.ndarray:
            n_bins = int(n_bins)
            if n_bins <= 0:
                raise ValueError("n_bins must be positive")
            return original_uniform_probabilities(n_bins, valid_bin_mask)

        duration._uniform_probabilities = uniform_probabilities_with_size_guard
        duration._uniform_probability_size_guard_applied = True


__all__ = ["apply_duration_occupancy_guard_patch"]
