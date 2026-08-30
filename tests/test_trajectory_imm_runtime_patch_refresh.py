from __future__ import annotations

import importlib

import numpy as np

import hipporeplayimm
from hipporeplayimm import state_space_trajectory_imm
from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.evidence_reporting import DEGENERATE_SINGLE_BIN_EVIDENCE_SUPPORT
from hipporeplayimm.state_space import StateSpaceDecoderConfig, StateSpaceReplayModel


def test_runtime_patches_restore_trajectory_imm_single_bin_diagnostics_after_reload() -> None:
    trajectory_imm = importlib.reload(state_space_trajectory_imm)
    assert not getattr(
        trajectory_imm._score_trajectory_imm_exact_sparse,
        "_trajectory_imm_single_bin_diagnostics_patch_applied",
        False,
    )

    hipporeplayimm.apply_runtime_patches()

    assert getattr(
        trajectory_imm._score_trajectory_imm_exact_sparse,
        "_trajectory_imm_single_bin_diagnostics_patch_applied",
        False,
    )

    emissions = LogEmissionTensor(
        log_likelihood=np.log(np.array([[0.6, 0.4]], dtype=float)),
        spike_counts=np.zeros((1, 1), dtype=int),
        times=np.array([0.0], dtype=float),
        dt=0.003,
        cell_ids=np.array([1], dtype=int),
        n_spikes=0,
    )
    centers = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)
    model = StateSpaceReplayModel(
        mode="trajectory-imm-exact-sparse",
        config=StateSpaceDecoderConfig(mode="trajectory-imm-exact-sparse"),
    )

    score = model.score(emissions, centers)

    assert score.diagnostics["state_space_trajectory_imm_evidence_support"] == (
        DEGENERATE_SINGLE_BIN_EVIDENCE_SUPPORT
    )
    assert score.diagnostics["state_space_momentum_evidence_support"] == (
        DEGENERATE_SINGLE_BIN_EVIDENCE_SUPPORT
    )
