from __future__ import annotations

import importlib

import hipporeplayimm
import hipporeplayimm.ground_truth as ground_truth


def test_runtime_patches_refresh_public_ground_truth_compare_alias_after_reload() -> None:
    stale_public = hipporeplayimm.compare_scores_to_ground_truth

    importlib.reload(ground_truth)

    assert hipporeplayimm.compare_scores_to_ground_truth is stale_public
    assert ground_truth.compare_scores_to_ground_truth is not stale_public

    hipporeplayimm.apply_runtime_patches()

    assert (
        hipporeplayimm.compare_scores_to_ground_truth
        is ground_truth.compare_scores_to_ground_truth
    )
    assert getattr(
        ground_truth.compare_scores_to_ground_truth,
        "_ground_truth_window_scope_wrapped",
        False,
    )
