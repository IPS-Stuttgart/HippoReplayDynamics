from __future__ import annotations

import numpy as np
import pytest
from scipy.special import logsumexp

from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.evidence_reporting import (
    DEGENERATE_SINGLE_BIN_EVIDENCE_SUPPORT,
    EVIDENCE_COMPARISON_DEGENERATE,
    evidence_comparison_from_support,
)
from hipporeplayimm.state_space import StateSpaceDecoderConfig, StateSpaceReplayModel


def test_trajectory_imm_single_bin_evidence_is_reported_degenerate() -> None:
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

    expected_logp = logsumexp(emissions.log_likelihood[0]) - np.log(emissions.n_bins)
    assert score.log_likelihood == pytest.approx(expected_logp)
    assert score.trajectory_log_posterior is not None
    assert score.diagnostics["state_space_trajectory_imm_evidence_support"] == (
        DEGENERATE_SINGLE_BIN_EVIDENCE_SUPPORT
    )
    assert score.diagnostics["state_space_momentum_evidence_support"] == (
        DEGENERATE_SINGLE_BIN_EVIDENCE_SUPPORT
    )
    assert score.diagnostics["state_space_trajectory_imm_degenerate_reason"] == (
        "single_time_bin_fragmented_marginal"
    )
    assert score.diagnostics["state_space_trajectory_imm_required_min_time_bins"] == 2
    assert evidence_comparison_from_support(
        score.diagnostics["state_space_trajectory_imm_evidence_support"]
    ) == EVIDENCE_COMPARISON_DEGENERATE
