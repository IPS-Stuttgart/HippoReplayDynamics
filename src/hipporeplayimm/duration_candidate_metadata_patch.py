"""Keep candidate-emission metadata isolated during occupancy masking."""

from __future__ import annotations


def apply_duration_candidate_metadata_patch() -> None:
    """Patch internal candidate-emission masking to avoid metadata aliasing.

    ``duration_occupancy._candidate_selection_emissions`` creates a derived
    ``LogEmissionTensor`` with invalid occupancy bins masked out before internal
    candidate selection.  The dataclass ``replace`` call used there carries the
    source metadata dictionary by reference, unlike the public masking helper.
    Keep metadata immutable from the caller's perspective by copying it whenever
    a derived tensor is returned.
    """

    from . import duration_occupancy

    if getattr(duration_occupancy, "_candidate_emission_metadata_patch_applied", False):
        return

    original_candidate_selection_emissions = duration_occupancy._candidate_selection_emissions

    def _candidate_selection_emissions_with_metadata_copy(emissions, valid_bin_mask):
        restricted = original_candidate_selection_emissions(emissions, valid_bin_mask)
        if restricted is emissions:
            return restricted
        metadata = getattr(restricted, "metadata", None)
        if metadata is None:
            return restricted
        restricted.metadata = dict(metadata)
        return restricted

    _candidate_selection_emissions_with_metadata_copy.__name__ = (
        original_candidate_selection_emissions.__name__
    )
    _candidate_selection_emissions_with_metadata_copy.__doc__ = (
        original_candidate_selection_emissions.__doc__
    )
    duration_occupancy._candidate_selection_emissions = _candidate_selection_emissions_with_metadata_copy
    duration_occupancy._candidate_emission_metadata_patch_applied = True
