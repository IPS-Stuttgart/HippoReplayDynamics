from __future__ import annotations

from hipporeplayimm.simulation_recovery import (
    SimulationRecoveryConfig,
    build_scoring_models,
    model_family,
)


TRAJECTORY_IMM_MODEL = "sorted-spike-state-space-trajectory-imm-exact-sparse"


def test_simulation_recovery_registers_trajectory_imm_exact_sparse() -> None:
    config = SimulationRecoveryConfig(scoring_models=(TRAJECTORY_IMM_MODEL,))

    models = build_scoring_models(config)

    assert list(models) == [TRAJECTORY_IMM_MODEL]
    assert models[TRAJECTORY_IMM_MODEL].mode == "trajectory-imm-exact-sparse"
    assert models[TRAJECTORY_IMM_MODEL].name == TRAJECTORY_IMM_MODEL
    assert model_family(TRAJECTORY_IMM_MODEL) == "trajectory"


def test_simulation_recovery_preserves_model_order_with_trajectory_imm() -> None:
    config = SimulationRecoveryConfig(
        scoring_models=(
            "random",
            TRAJECTORY_IMM_MODEL,
            "sorted-spike-state-space-diffusion",
        )
    )

    models = build_scoring_models(config)

    assert list(models) == [
        "random",
        TRAJECTORY_IMM_MODEL,
        "sorted-spike-state-space-diffusion",
    ]
