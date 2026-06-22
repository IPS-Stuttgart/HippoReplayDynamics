from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

import hipporeplayimm.benchmarks as benchmarks
import hipporeplayimm.ground_truth as ground_truth
from hipporeplayimm.simulation_recovery import (
    SimulationRecoveryConfig,
    _score_recovery_model,
    build_scoring_models,
    model_family,
)


TRAJECTORY_IMM_MODEL = "sorted-spike-state-space-trajectory-imm-exact-sparse"
TRAJECTORY_IMM_SHORT_MODEL = "trajectory-imm-exact-sparse"


def test_simulation_recovery_registers_trajectory_imm_exact_sparse() -> None:
    config = SimulationRecoveryConfig(scoring_models=(TRAJECTORY_IMM_MODEL,))

    models = build_scoring_models(config)

    assert list(models) == [TRAJECTORY_IMM_MODEL]
    assert models[TRAJECTORY_IMM_MODEL].mode == "trajectory-imm-exact-sparse"
    assert models[TRAJECTORY_IMM_MODEL].name == TRAJECTORY_IMM_MODEL
    assert model_family(TRAJECTORY_IMM_MODEL) == "trajectory"


def test_simulation_recovery_accepts_short_trajectory_imm_exact_sparse_alias() -> None:
    config = SimulationRecoveryConfig(scoring_models=(TRAJECTORY_IMM_SHORT_MODEL,))

    models = build_scoring_models(config)

    assert list(models) == [TRAJECTORY_IMM_SHORT_MODEL]
    assert models[TRAJECTORY_IMM_SHORT_MODEL].mode == "trajectory-imm-exact-sparse"
    assert models[TRAJECTORY_IMM_SHORT_MODEL].name == TRAJECTORY_IMM_SHORT_MODEL
    assert model_family(TRAJECTORY_IMM_SHORT_MODEL) == "trajectory"


def test_simulation_recovery_preserves_model_order_with_trajectory_imm() -> None:
    config = SimulationRecoveryConfig(
        scoring_models=(
            "random",
            TRAJECTORY_IMM_MODEL,
            TRAJECTORY_IMM_SHORT_MODEL,
            "sorted-spike-state-space-diffusion",
        )
    )

    models = build_scoring_models(config)

    assert list(models) == [
        "random",
        TRAJECTORY_IMM_MODEL,
        TRAJECTORY_IMM_SHORT_MODEL,
        "sorted-spike-state-space-diffusion",
    ]


def test_trajectory_imm_recovery_scores_evidence_only(monkeypatch) -> None:
    model = build_scoring_models(
        SimulationRecoveryConfig(scoring_models=(TRAJECTORY_IMM_MODEL,))
    )[TRAJECTORY_IMM_MODEL]
    observed: dict[str, object] = {}

    def fake_score(emissions, bin_centers, **kwargs):
        observed["emissions"] = emissions
        observed["bin_centers"] = bin_centers
        observed.update(kwargs)
        return SimpleNamespace(log_likelihood=0.0, diagnostics={})

    monkeypatch.setattr(model, "score", fake_score)
    encoding = SimpleNamespace(
        bin_centers=np.zeros((1, 2), dtype=float),
        occupancy_s=np.ones(1, dtype=float),
    )
    emissions = object()

    score = _score_recovery_model(model, emissions, encoding, score_with_occupancy=True)

    assert score.log_likelihood == 0.0
    assert observed["emissions"] is emissions
    np.testing.assert_array_equal(observed["bin_centers"], encoding.bin_centers)
    assert observed["return_trajectory"] is False
    np.testing.assert_array_equal(observed["occupancy_s"], encoding.occupancy_s)


def test_saved_split_strategy_controls_direct_ground_truth_helper() -> None:
    class EncodingStub:
        cell_ids = np.array([1, 2, 3, 4], dtype=int)
        rates_hz = np.array([[1.0, 1.0], [2.0, 1.0], [3.0, 1.0], [20.0, 1.0]])
        occupancy_s = np.ones(2, dtype=float)

    encoding = EncodingStub()
    rows = pd.DataFrame(
        {
            "benchmark_test_cell_fraction": [0.5],
            "benchmark_random_seed": [1],
            "benchmark_cell_split_seed": [3],
            "benchmark_cell_split_strategy": ["peak-rate"],
            "benchmark_cell_split_strata": [2],
        }
    )

    train, test = ground_truth._cell_split_for_score_rows(
        rows,
        encoding,
        benchmarks.BenchmarkConfig(cell_split_strategy="random", cell_split_strata=4),
    )
    expected_train, expected_test = benchmarks.stratified_cell_split(
        encoding.cell_ids,
        benchmarks._cell_split_scores_from_encoding(encoding, "peak-rate"),
        0.5,
        3,
        n_strata=2,
    )

    np.testing.assert_array_equal(train, expected_train)
    np.testing.assert_array_equal(test, expected_test)
