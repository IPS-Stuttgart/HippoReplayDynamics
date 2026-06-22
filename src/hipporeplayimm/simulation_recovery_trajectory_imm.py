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
_TRAJECTORY_MODEL_ALIASES = frozenset({"state-space-velocity-momentum"})


def apply_trajectory_imm_recovery_patch() -> None:
    """Make the implemented trajectory-IMM model available to recovery runs."""

    import hipporeplayimm.simulation_recovery as recovery

    from . import ground_truth_cell_split_strategy as gt_split_strategy

    gt_split_strategy.apply_ground_truth_cell_split_strategy_patch()
    _patch_trajectory_imm_recovery_scoring(recovery)

    if getattr(recovery, "_trajectory_imm_recovery_patch_applied", False):
        return

    recovery._TRAJECTORY = set(getattr(recovery, "_TRAJECTORY", set())) | set(
        _TRAJECTORY_IMM_ALIASES | _TRAJECTORY_MODEL_ALIASES
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


def _patch_trajectory_imm_recovery_scoring(recovery: Any) -> None:
    """Avoid trajectory-posterior materialization in evidence-only recovery scoring."""

    if getattr(recovery, "_trajectory_imm_recovery_evidence_only_patch_applied", False):
        return

    previous_score_recovery_model = recovery._score_recovery_model

    def score_recovery_model_evidence_only_trajectory_imm(
        model: object,
        emissions: object,
        encoding: object,
        *,
        candidate_indices: list[object] | None = None,
        score_with_occupancy: bool = True,
    ) -> object:
        if isinstance(model, SortedSpikeStateSpaceReplayModel) and model.mode == _TRAJECTORY_IMM_MODE:
            kwargs: dict[str, object] = {"return_trajectory": False}
            if candidate_indices is not None:
                kwargs["candidate_indices"] = candidate_indices
            if score_with_occupancy:
                kwargs["occupancy_s"] = getattr(encoding, "occupancy_s")
            return model.score(
                emissions,  # type: ignore[arg-type]
                getattr(encoding, "bin_centers"),
                **kwargs,
            )
        return previous_score_recovery_model(
            model,
            emissions,
            encoding,
            candidate_indices=candidate_indices,
            score_with_occupancy=score_with_occupancy,
        )

    recovery._score_recovery_model = score_recovery_model_evidence_only_trajectory_imm
    recovery._trajectory_imm_recovery_evidence_only_patch_applied = True
