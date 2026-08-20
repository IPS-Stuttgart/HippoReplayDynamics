from __future__ import annotations

from hipporeplayimm import state_space_model
from hipporeplayimm.state_space_candidate_bin_center_validation import (
    _ORIGINAL_ATTR,
    _PATCHED_FLAG,
    apply_state_space_candidate_bin_center_validation_patch,
)


def _marker_count(function: object) -> int:
    seen: set[int] = set()
    current = function
    count = 0
    while callable(current) and id(current) not in seen:
        seen.add(id(current))
        count += int(bool(getattr(current, _PATCHED_FLAG, False)))
        current = getattr(current, _ORIGINAL_ATTR, None)
    return count


def test_candidate_bin_center_wrapper_is_installed_once() -> None:
    assert _marker_count(state_space_model.StateSpaceReplayModel.candidate_indices) == 1


def test_candidate_bin_center_patch_detects_marker_below_outer_wrapper(monkeypatch) -> None:
    current = state_space_model.StateSpaceReplayModel.candidate_indices

    def outer(self, *args, **kwargs):
        return current(self, *args, **kwargs)

    setattr(outer, _ORIGINAL_ATTR, current)
    monkeypatch.setattr(
        state_space_model.StateSpaceReplayModel,
        "candidate_indices",
        outer,
    )

    apply_state_space_candidate_bin_center_validation_patch()

    assert state_space_model.StateSpaceReplayModel.candidate_indices is outer
    assert _marker_count(outer) == 1
