from __future__ import annotations

import importlib.util
import itertools
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
from scipy.special import logsumexp

from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.evidence_reporting import EXACT_EVIDENCE_SUPPORT
from hipporeplayimm.sorted_spike_state_space import SortedSpikeStateSpaceReplayModel
from hipporeplayimm.state_space import StateSpaceDecoderConfig, StateSpaceReplayModel


EXACT_CORE_MODES = (
    "stationary",
    "diffusion",
    "fragmented",
    "first-order-imm",
    "momentum-exact-sparse",
)


def test_exact_core_state_space_models_accept_1d_bin_centers() -> None:
    emissions = _tiny_1d_emissions()
    centers = _tiny_1d_centers(emissions.n_bins)

    assert centers.shape == (emissions.n_bins, 1)
    for mode in EXACT_CORE_MODES:
        model = SortedSpikeStateSpaceReplayModel(mode=mode, config=_config(mode))

        score = model.score(emissions, centers)

        assert np.isfinite(score.log_likelihood), mode
        assert score.model_name == f"sorted-spike-state-space-{mode}"
        assert score.n_time == emissions.n_time
        assert score.n_spikes == emissions.n_spikes
        assert score.terminal_log_posterior is not None
        assert score.terminal_log_posterior.shape == (emissions.n_bins,)
        assert score.trajectory_log_posterior is not None
        assert score.trajectory_log_posterior.shape == (emissions.n_time, emissions.n_bins)
        np.testing.assert_allclose(logsumexp(score.trajectory_log_posterior, axis=1), 0.0, atol=1e-12)
        assert score.diagnostics["state_space_observation_model"] == "sorted-spike-poisson"
        assert score.diagnostics["clusterless_mark_likelihood"] == "not_implemented"


def test_exact_core_state_space_models_support_1d_evidence_only_scoring() -> None:
    emissions = _tiny_1d_emissions()
    centers = _tiny_1d_centers(emissions.n_bins)

    for mode in EXACT_CORE_MODES:
        model = SortedSpikeStateSpaceReplayModel(mode=mode, config=_config(mode))

        full = model.score(emissions, centers, return_trajectory=True)
        evidence_only = model.score(emissions, centers, return_trajectory=False)

        assert evidence_only.log_likelihood == pytest.approx(full.log_likelihood, abs=1e-10), mode
        assert evidence_only.trajectory_log_posterior is None
        assert evidence_only.terminal_log_posterior is not None
        assert evidence_only.terminal_log_posterior.shape == (emissions.n_bins,)
        assert evidence_only.diagnostics["state_space_trajectory_posterior"] == 0


def test_exact_sparse_momentum_matches_1d_bruteforce_tiny_grid() -> None:
    emissions = LogEmissionTensor(
        log_likelihood=np.log(
            np.array(
                [
                    [0.7, 0.3],
                    [0.2, 0.8],
                    [0.4, 0.6],
                    [0.45, 0.55],
                ],
                dtype=float,
            )
        ),
        spike_counts=np.zeros((4, 1), dtype=int),
        times=np.array([0.0, 1.0, 2.0, 3.0]),
        dt=1.0,
        cell_ids=np.array([1]),
        n_spikes=0,
    )
    centers = np.array([[0.0], [1.0]], dtype=float)
    config = StateSpaceDecoderConfig(
        mode="momentum-exact-sparse",
        momentum_sigma_cm_sqrt_s=1.0,
        momentum_initial_sigma_cm_sqrt_s=1.0,
        momentum_velocity_decay=0.9,
        max_step_sigma=10.0,
    )

    score = StateSpaceReplayModel(mode="momentum-exact-sparse", config=config).score(emissions, centers)

    def kernel_log(predicted: np.ndarray, dst: int, sigma: float) -> float:
        weights = np.exp(-0.5 * np.sum((centers - predicted[None, :]) ** 2, axis=1) / (sigma * sigma))
        return float(np.log(weights[dst] / weights.sum()))

    brute_terms = []
    for x0, x1, x2, x3 in itertools.product(range(2), repeat=4):
        predicted = centers[x1] + config.momentum_velocity_decay * (centers[x1] - centers[x0])
        predicted_next = centers[x2] + config.momentum_velocity_decay * (centers[x2] - centers[x1])
        brute_terms.append(
            -np.log(2.0)
            + emissions.log_likelihood[0, x0]
            + kernel_log(centers[x0], x1, 1.0)
            + emissions.log_likelihood[1, x1]
            + kernel_log(predicted, x2, 1.0)
            + emissions.log_likelihood[2, x2]
            + kernel_log(predicted_next, x3, 1.0)
            + emissions.log_likelihood[3, x3]
        )

    assert score.log_likelihood == pytest.approx(logsumexp(brute_terms), abs=1e-10)
    assert score.diagnostics["state_space_sparse_momentum_evidence_support"] == EXACT_EVIDENCE_SUPPORT
    assert score.diagnostics["state_space_sparse_momentum_max_pair_count"] == 4


def test_1d_exact_core_benchmark_rows_keep_family_and_evidence_flags() -> None:
    benchmark = _load_benchmark_module()
    emissions = _tiny_1d_emissions()
    centers = _tiny_1d_centers(emissions.n_bins)
    rows = []
    for mode in EXACT_CORE_MODES:
        model = SortedSpikeStateSpaceReplayModel(mode=mode, config=_config(mode))
        score = model.score(emissions, centers, return_trajectory=False)
        row = {
            "status": "success",
            "session": "Synthetic1D/Tiny",
            "event_index": 0,
            "model": score.model_name,
            "requested_model": score.model_name,
            "model_family": benchmark._family(score.model_name),
            "log_evidence": float(score.log_likelihood),
            "n_time": int(score.n_time),
            "n_spikes": int(score.n_spikes),
            "runtime_s": 0.0,
            "error": "",
        }
        row.update({f"diagnostic_{key}": value for key, value in score.diagnostics.items()})
        rows.append(row)

    table = benchmark._postprocess_evidence_scores(pd.DataFrame(rows))

    assert set(table["model"]) == {f"sorted-spike-state-space-{mode}" for mode in EXACT_CORE_MODES}
    assert set(table["evidence_support"]) == {EXACT_EVIDENCE_SUPPORT}
    assert table["evidence_comparable"].map(bool).all()
    family_by_model = dict(zip(table["model"], table["model_family"]))
    assert family_by_model["sorted-spike-state-space-stationary"] == "nontrajectory"
    for mode in EXACT_CORE_MODES:
        if mode != "stationary":
            assert family_by_model[f"sorted-spike-state-space-{mode}"] == "trajectory"
    assert table["model_probability"].notna().all()
    assert table["model_probability"].sum() == pytest.approx(1.0)
    assert table["is_best_model"].map(bool).sum() == 1


def _tiny_1d_emissions() -> LogEmissionTensor:
    return LogEmissionTensor(
        log_likelihood=np.log(
            np.array(
                [
                    [0.70, 0.20, 0.08, 0.02],
                    [0.15, 0.65, 0.15, 0.05],
                    [0.05, 0.15, 0.65, 0.15],
                    [0.03, 0.12, 0.25, 0.60],
                ],
                dtype=float,
            )
        ),
        spike_counts=np.zeros((4, 1), dtype=int),
        times=np.array([0.0, 1.0, 2.0, 3.0]),
        dt=1.0,
        cell_ids=np.array([1]),
        n_spikes=0,
    )


def _tiny_1d_centers(n_bins: int) -> np.ndarray:
    return np.arange(float(n_bins), dtype=float).reshape(-1, 1)


def _config(mode: str) -> StateSpaceDecoderConfig:
    return StateSpaceDecoderConfig(
        mode=mode,
        stationary_sigma_cm=1.0,
        diffusion_sigma_cm_sqrt_s=1.0,
        momentum_sigma_cm_sqrt_s=1.0,
        momentum_initial_sigma_cm_sqrt_s=1.0,
        momentum_velocity_decay=0.9,
        max_step_sigma=10.0,
        momentum_candidate_top_k=4,
        momentum_predicted_candidate_top_k=4,
        imm_mode_stickiness=0.8,
    )


def _load_benchmark_module():
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root / "src") not in sys.path:
        sys.path.insert(0, str(repo_root / "src"))
    module_path = repo_root / "scripts" / "benchmark_model_evidence.py"
    spec = importlib.util.spec_from_file_location("benchmark_model_evidence", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
