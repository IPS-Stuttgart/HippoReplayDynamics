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
_TRAJECTORY_IMM_ALIASES = frozenset({_SORTED_SPIKE_TRAJECTORY_IMM, _TRAJECTORY_IMM_MODE})


def apply_trajectory_imm_recovery_patch() -> None:
    """Make the implemented trajectory-IMM model available to recovery runs."""

    import hipporeplayimm.simulation_recovery as recovery

    from . import ground_truth_cell_split_strategy as gt_split_strategy

    gt_split_strategy.apply_ground_truth_cell_split_strategy_patch()

    if getattr(recovery, "_trajectory_imm_recovery_patch_applied", False):
        return

    recovery._TRAJECTORY = set(getattr(recovery, "_TRAJECTORY", set())) | set(
        _TRAJECTORY_IMM_ALIASES
    )

    previous_build_scoring_models = recovery.build_scoring_models

    def build_scoring_models_with_trajectory_imm(config: Any) -> dict[str, object]:
        names = recovery.parse_model_list(config.scoring_models)
        requested_trajectory_names = tuple(name for name in names if name in _TRAJECTORY_IMM_ALIASES)
        if not requested_trajectory_names:
            return previous_build_scoring_models(config)

        non_trajectory_names = tuple(
            name for name in names if name not in _TRAJECTORY_IMM_ALIASES
        )
        models: dict[str, object] = {}
        if non_trajectory_names:
            base_config = replace(config, scoring_models=non_trajectory_names)
            models.update(previous_build_scoring_models(base_config))

        state_space_config = recovery._recovery_state_space_config(config)
        for requested_name in dict.fromkeys(requested_trajectory_names):
            models[requested_name] = SortedSpikeStateSpaceReplayModel(
                mode=_TRAJECTORY_IMM_MODE,
                config=replace(state_space_config, mode=_TRAJECTORY_IMM_MODE),
                name=requested_name,
            )
        return {name: models[name] for name in dict.fromkeys(names)}

    recovery.build_scoring_models = build_scoring_models_with_trajectory_imm
    recovery._trajectory_imm_recovery_patch_applied = True
