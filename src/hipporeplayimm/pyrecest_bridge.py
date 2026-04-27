"""Optional adapters to PyRecEst replay filters."""

from __future__ import annotations

import numpy as np


def is_pyrecest_available() -> bool:
    try:
        import pyrecest  # noqa: F401
    except ImportError:
        return False
    return True


def build_goal_conditioned_replay_imm_filter(
    initial_position: np.ndarray,
    position_covariance: np.ndarray,
    candidate_goals: np.ndarray,
    **kwargs,
):
    """Instantiate PyRecEst's goal-conditioned replay IMM filter."""

    from pyrecest.filters import GoalConditionedReplayIMMFilter

    return GoalConditionedReplayIMMFilter(
        initial_state=(np.asarray(initial_position, dtype=float), np.asarray(position_covariance, dtype=float)),
        candidate_goals=np.asarray(candidate_goals, dtype=float),
        **kwargs,
    )


def build_interacting_multiple_model_filter(filter_bank, transition_matrix, mode_probabilities=None):
    """Instantiate PyRecEst's generic linear-Gaussian IMM filter."""

    from pyrecest.filters import InteractingMultipleModelFilter

    return InteractingMultipleModelFilter(
        filter_bank=filter_bank,
        transition_matrix=np.asarray(transition_matrix, dtype=float),
        mode_probabilities=mode_probabilities,
    )
