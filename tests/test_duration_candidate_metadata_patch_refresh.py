from __future__ import annotations

from types import SimpleNamespace

import hipporeplayimm.duration_candidate_metadata_patch as patch
import hipporeplayimm.duration_occupancy as duration_occupancy


def test_candidate_metadata_patch_refreshes_stale_selection_helper(monkeypatch):
    def lossy_candidate_selection_emissions(emissions, valid_bin_mask):
        del valid_bin_mask
        return SimpleNamespace(metadata=emissions.metadata)

    emissions = SimpleNamespace(metadata={"source": "original"})
    monkeypatch.setattr(
        duration_occupancy,
        "_candidate_selection_emissions",
        lossy_candidate_selection_emissions,
    )
    monkeypatch.setattr(
        duration_occupancy,
        patch._CANDIDATE_METADATA_PATCHED_FLAG,
        True,
        raising=False,
    )

    patch.apply_duration_candidate_metadata_patch()

    assert getattr(
        duration_occupancy._candidate_selection_emissions,
        patch._CANDIDATE_METADATA_SELECTION_WRAPPED_FLAG,
        False,
    )
    restricted = duration_occupancy._candidate_selection_emissions(emissions, object())
    assert restricted.metadata == emissions.metadata
    assert restricted.metadata is not emissions.metadata

    restricted.metadata["source"] = "restricted"
    assert emissions.metadata["source"] == "original"
