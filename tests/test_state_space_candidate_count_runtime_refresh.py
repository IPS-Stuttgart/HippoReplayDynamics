from __future__ import annotations

import importlib

import hipporeplayimm
import hipporeplayimm.state_space_model as state_space_model
from hipporeplayimm.state_space_candidate_bin_center_validation import (
    _PATCHED_FLAG as _BIN_CENTER_PATCHED_FLAG,
)
from hipporeplayimm.state_space_candidate_count_validation import (
    _PATCHED_FLAG,
    _SCORE_PATCHED_FLAG,
    _wrapper_chain_has_marker,
)


def test_runtime_patches_restore_candidate_count_validation_after_model_reload() -> None:
    module = importlib.reload(state_space_model)

    assert not _wrapper_chain_has_marker(
        module.StateSpaceReplayModel.candidate_indices,
        _PATCHED_FLAG,
    )
    assert not _wrapper_chain_has_marker(
        module.StateSpaceReplayModel.candidate_indices,
        _BIN_CENTER_PATCHED_FLAG,
    )
    assert not _wrapper_chain_has_marker(
        module.StateSpaceReplayModel.score,
        _SCORE_PATCHED_FLAG,
    )

    hipporeplayimm.apply_runtime_patches()

    assert _wrapper_chain_has_marker(
        module.StateSpaceReplayModel.candidate_indices,
        _PATCHED_FLAG,
    )
    assert _wrapper_chain_has_marker(
        module.StateSpaceReplayModel.candidate_indices,
        _BIN_CENTER_PATCHED_FLAG,
    )
    assert _wrapper_chain_has_marker(
        module.StateSpaceReplayModel.score,
        _SCORE_PATCHED_FLAG,
    )
