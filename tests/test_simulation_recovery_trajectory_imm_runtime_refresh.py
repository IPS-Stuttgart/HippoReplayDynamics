from __future__ import annotations

import importlib

import hipporeplayimm
from hipporeplayimm import simulation_recovery
from hipporeplayimm.simulation_recovery_trajectory_imm import (
    _BUILD_WRAPPER_MARKER,
    _SCORE_WRAPPER_MARKER,
    _TRAJECTORY_IMM_ALIASES,
)


def test_runtime_patches_restore_trajectory_imm_recovery_after_reload() -> None:
    recovery = importlib.reload(simulation_recovery)
    try:
        # ``reload`` keeps arbitrary module attributes, so the legacy sentinels
        # survive even though the source-defined registry and functions reset.
        assert getattr(recovery, "_trajectory_imm_recovery_patch_applied", False)
        assert getattr(
            recovery,
            "_trajectory_imm_recovery_evidence_only_patch_applied",
            False,
        )
        assert _TRAJECTORY_IMM_ALIASES.isdisjoint(recovery._TRAJECTORY)
        assert not getattr(recovery.build_scoring_models, _BUILD_WRAPPER_MARKER, False)
        assert not getattr(recovery._score_recovery_model, _SCORE_WRAPPER_MARKER, False)

        hipporeplayimm.apply_runtime_patches()

        assert _TRAJECTORY_IMM_ALIASES.issubset(recovery._TRAJECTORY)
        assert getattr(recovery.build_scoring_models, _BUILD_WRAPPER_MARKER, False)
        assert getattr(recovery._score_recovery_model, _SCORE_WRAPPER_MARKER, False)

        build_wrapper = recovery.build_scoring_models
        score_wrapper = recovery._score_recovery_model
        hipporeplayimm.apply_runtime_patches()
        assert recovery.build_scoring_models is build_wrapper
        assert recovery._score_recovery_model is score_wrapper
    finally:
        hipporeplayimm.apply_runtime_patches()
