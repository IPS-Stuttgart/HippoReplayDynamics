from __future__ import annotations

import warnings

import numpy as np

from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.models import CandidateKinematicModel


_POSTERIOR_DIAGNOSTIC_KEYS = {
    "decoded_endpoint_x",
    "decoded_endpoint_y",
    "decoded_map_x",
    "decoded_map_y",
    "decoded_map_bin",
    "terminal_posterior_entropy",
}


def _emissions(log_likelihood: np.ndarray) -> LogEmissionTensor:
    values = np.asarray(log_likelihood, dtype=float)
    return LogEmissionTensor(
        log_likelihood=values,
        spike_counts=np.zeros((values.shape[0], 1), dtype=int),
        times=np.arange(values.shape[0], dtype=float),
        dt=1.0,
        cell_ids=np.array([1], dtype=int),
        n_spikes=0,
    )


def _extreme_bin_centers() -> np.ndarray:
    return np.array([[-1.0e308], [1.0e308]], dtype=float)


def test_candidate_model_clears_undefined_disconnected_path_posterior() -> None:
    emissions = _emissions(
        np.array(
            [
                [0.0, -np.inf],
                [-np.inf, 0.0],
            ]
        )
    )
    candidates = [
        np.array([0], dtype=int),
        np.array([1], dtype=int),
    ]

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        score = CandidateKinematicModel(
            mode="diffusion",
            top_k=1,
            diffusion_sigma_cm=1.0,
        ).score(
            emissions,
            _extreme_bin_centers(),
            candidate_indices=candidates,
        )

    assert np.isneginf(score.log_likelihood)
    assert score.terminal_log_posterior is None
    assert score.trajectory_log_posterior is None
    assert score.diagnostics["candidate_path_support"] == "no_finite_path"
    assert score.diagnostics["candidate_evidence_support"] == "truncated_full_grid"
    assert _POSTERIOR_DIAGNOSTIC_KEYS.isdisjoint(score.diagnostics)


def test_candidate_model_keeps_connected_path_posterior() -> None:
    emissions = _emissions(
        np.array(
            [
                [0.0, -np.inf],
                [0.0, -np.inf],
            ]
        )
    )
    candidates = [
        np.array([0], dtype=int),
        np.array([0], dtype=int),
    ]

    score = CandidateKinematicModel(
        mode="diffusion",
        top_k=1,
        diffusion_sigma_cm=1.0,
    ).score(
        emissions,
        _extreme_bin_centers(),
        candidate_indices=candidates,
    )

    assert np.isfinite(score.log_likelihood)
    assert score.terminal_log_posterior is not None
    assert score.trajectory_log_posterior is not None
    assert "candidate_path_support" not in score.diagnostics
    assert score.diagnostics["decoded_map_bin"] == 0
    np.testing.assert_array_equal(
        score.terminal_log_posterior,
        np.array([0.0, -np.inf]),
    )
