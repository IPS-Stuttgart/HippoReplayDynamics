import numpy as np
import pytest
from scipy.special import logsumexp

from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.evidence_reporting import DEGENERATE_SINGLE_BIN_EVIDENCE_SUPPORT
from hipporeplayimm.state_space import StateSpaceDecoderConfig, StateSpaceReplayModel


def _single_bin_emissions() -> LogEmissionTensor:
    return LogEmissionTensor(
        log_likelihood=np.log(np.array([[0.6, 0.4]], dtype=float)),
        spike_counts=np.zeros((1, 1), dtype=int),
        times=np.array([0.0]),
        dt=0.003,
        cell_ids=np.array([1]),
        n_spikes=0,
    )


def test_exact_sparse_momentum_single_bin_reports_complete_evidence_diagnostics():
    emissions = _single_bin_emissions()
    centers = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)
    model = StateSpaceReplayModel(
        mode="momentum-exact-sparse",
        config=StateSpaceDecoderConfig(mode="momentum-exact-sparse"),
    )

    full = model.score(emissions, centers, return_trajectory=True)
    evidence_only = model.score(emissions, centers, return_trajectory=False)

    assert np.isfinite(full.log_likelihood)
    assert evidence_only.log_likelihood == pytest.approx(full.log_likelihood, abs=1e-12)
    assert full.trajectory_log_posterior is not None
    assert evidence_only.trajectory_log_posterior is None
    assert evidence_only.terminal_log_posterior is not None
    assert np.allclose(logsumexp(evidence_only.terminal_log_posterior), 0.0)

    assert full.diagnostics["state_space_sparse_momentum_evidence_support"] == DEGENERATE_SINGLE_BIN_EVIDENCE_SUPPORT
    assert full.diagnostics["state_space_sparse_momentum_evidence_mode"] == "single_bin_fragmented_fallback"
    assert full.diagnostics["state_space_sparse_momentum_evidence_only"] == 0
    assert full.diagnostics["state_space_sparse_momentum_backward_transition_rows"] == "none_single_bin"
    assert full.diagnostics["state_space_momentum_trajectory_posterior"] == "single_bin_fragmented_fallback"

    assert evidence_only.diagnostics["state_space_sparse_momentum_evidence_support"] == DEGENERATE_SINGLE_BIN_EVIDENCE_SUPPORT
    assert evidence_only.diagnostics["state_space_sparse_momentum_evidence_mode"] == "evidence_only"
    assert evidence_only.diagnostics["state_space_sparse_momentum_evidence_only"] == 1
    assert evidence_only.diagnostics["state_space_sparse_momentum_backward_transition_rows"] == "skipped_evidence_only"
    assert evidence_only.diagnostics["state_space_momentum_trajectory_posterior"] == "not_returned_evidence_only"
