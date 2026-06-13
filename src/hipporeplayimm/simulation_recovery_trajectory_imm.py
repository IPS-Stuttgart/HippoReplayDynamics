"""Register trajectory-IMM scoring models for simulation recovery.

The state-space decoder already implements ``trajectory-imm-exact-sparse``.
This compatibility patch makes the implemented sorted-spike model selectable via
``SimulationRecoveryConfig.scoring_models`` and classifies it as a trajectory
family model in recovery summaries.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from .sorted_spike_state_space import SortedSpikeStateSpaceReplayModel

_SORTED_SPIKE_TRAJECTORY_IMM = "sorted-spike-state-space-trajectory-imm-exact-sparse"
_TRAJECTORY_IMM_MODE = "trajectory-imm-exact-sparse"


def apply_trajectory_imm_recovery_patch() -> None:
    """Make the implemented trajectory-IMM model available to recovery runs."""

    import hipporeplayimm.simulation_recovery as recovery

    if getattr(recovery, "_trajectory_imm_recovery_patch_applied", False):
        return

    recovery._TRAJECTORY = set(getattr(recovery, "_TRAJECTORY", set())) | {
        _TRAJECTORY_IMM_MODE,
        _SORTED_SPIKE_TRAJECTORY_IMM,
    }

    previous_build_scoring_models = recovery.build_scoring_models

    def build_scoring_models_with_trajectory_imm(config: Any) -> dict[str, object]:
        names = recovery.parse_model_list(config.scoring_models)
        if _SORTED_SPIKE_TRAJECTORY_IMM not in names:
            return previous_build_scoring_models(config)

        non_trajectory_names = tuple(
            name for name in names if name != _SORTED_SPIKE_TRAJECTORY_IMM
        )
        models: dict[str, object] = {}
        if non_trajectory_names:
            base_config = replace(config, scoring_models=non_trajectory_names)
            models.update(previous_build_scoring_models(base_config))

        state_space_config = recovery._recovery_state_space_config(config)
        trajectory_model = SortedSpikeStateSpaceReplayModel(
            mode=_TRAJECTORY_IMM_MODE,
            config=replace(state_space_config, mode=_TRAJECTORY_IMM_MODE),
        )
        models[_SORTED_SPIKE_TRAJECTORY_IMM] = trajectory_model
        return {name: models[name] for name in dict.fromkeys(names)}

    recovery.build_scoring_models = build_scoring_models_with_trajectory_imm
    recovery._trajectory_imm_recovery_patch_applied = True
