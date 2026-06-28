from hipporeplayimm.duration_dynamics import apply_duration_dynamics_patch
from hipporeplayimm.state_space import StateSpaceReplayModel


def test_state_space_score_is_marked_duration_occupancy_aware():
    assert getattr(StateSpaceReplayModel.score, "_native_duration_occupancy_aware", False)


def test_legacy_duration_patch_preserves_native_state_space_score():
    score_before = StateSpaceReplayModel.score

    apply_duration_dynamics_patch()

    assert StateSpaceReplayModel.score is score_before
