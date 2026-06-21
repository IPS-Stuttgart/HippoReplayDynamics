"""Runtime guard for occupancy-masked emission metadata isolation."""

from __future__ import annotations

from dataclasses import replace


def apply_duration_occupancy_metadata_guard_patch() -> None:
    """Ensure occupancy candidate-selection emissions do not share metadata.

    The duration/occupancy scorer derives a temporary emission tensor whose
    likelihood rows mask invalid occupancy bins before candidate selection.  A
    bare dataclass ``replace`` keeps the original mutable metadata mapping by
    reference, so any downstream metadata annotation on the temporary tensor can
    mutate the caller's emission object.  Keep the original helper's validation
    and masking semantics, but copy metadata whenever a derived tensor is
    returned.
    """

    from . import duration_occupancy as _duration_occupancy
    from . import transition_duration_validation as _transition_duration_validation

    _transition_duration_validation.apply_transition_duration_validation_patch()

    if getattr(_duration_occupancy, "_metadata_guard_patch_applied", False):
        return

    previous_candidate_selection = _duration_occupancy._candidate_selection_emissions
    previous_uniform_probabilities = _duration_occupancy._uniform_probabilities

    def _candidate_selection_emissions(emissions, valid_bin_mask):
        restricted = previous_candidate_selection(emissions, valid_bin_mask)
        if restricted is emissions:
            return restricted
        metadata = dict(getattr(restricted, "metadata", {}))
        return replace(restricted, metadata=metadata)

    def _uniform_probabilities(n_bins: int, valid_bin_mask=None):
        n_bins = int(n_bins)
        if n_bins <= 0:
            raise ValueError("n_bins must be positive")
        return previous_uniform_probabilities(n_bins, valid_bin_mask)

    _candidate_selection_emissions.__name__ = previous_candidate_selection.__name__
    _candidate_selection_emissions.__doc__ = previous_candidate_selection.__doc__
    _uniform_probabilities.__name__ = previous_uniform_probabilities.__name__
    _uniform_probabilities.__doc__ = previous_uniform_probabilities.__doc__

    _duration_occupancy._candidate_selection_emissions = _candidate_selection_emissions
    _duration_occupancy._uniform_probabilities = _uniform_probabilities
    _duration_occupancy._metadata_guard_patch_applied = True
