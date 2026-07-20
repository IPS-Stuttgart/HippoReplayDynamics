from __future__ import annotations


def test_runtime_patch_refresh_does_not_stack_candidate_index_wrappers() -> None:
    import hipporeplayimm
    from hipporeplayimm import state_space_model

    candidate_indices = state_space_model.StateSpaceReplayModel.candidate_indices

    hipporeplayimm.apply_runtime_patches()
    hipporeplayimm.apply_runtime_patches()

    assert state_space_model.StateSpaceReplayModel.candidate_indices is candidate_indices
