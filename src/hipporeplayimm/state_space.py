'''State-space replay decoder public import surface.'''

from __future__ import annotations

from .goal_state_space import GoalStateSpaceReplayModel
from .models import EventScore, LOG_ZERO, _normalize_log_weights, _posterior_diagnostics
from .momentum_prediction_decay_validation import (
    apply_momentum_prediction_decay_validation_patch as _apply_momentum_prediction_decay_validation_patch,
)
from .state_space_model import (
    StateSpaceDecoderConfig,
    StateSpaceReplayModel,
    _augment_candidates_with_momentum_predictions,
    _candidate_evidence_support_label,
    _candidate_selection_label,
    _candidate_support_config_diagnostics,
)
from .state_space_candidates import (
    _advance_imm_pair_log_alpha,
    _backward_imm_pair,
    _backward_imm_pair_for_mode,
    _init_imm_pair_log_alpha,
    _score_imm_candidates,
)
from .state_space_candidates_momentum import (
    _advance_momentum_pair,
    _backward_momentum_pair,
    _init_pair_log_alpha,
    _score_momentum_candidates,
)
from .state_space_displacement_momentum import (
    _displacement_lattice,
    _score_displacement_momentum_exact,
)
from .state_space_displacement_imm import (
    _score_displacement_imm_exact,
)
from .state_space_sparse_momentum import (
    _score_sparse_momentum_exact,
)
from .state_space_trajectory_imm import (
    _score_trajectory_imm_exact_sparse,
)
from .state_space_first_order import (
    _apply_transition,
    _apply_transition_backward,
    _forward_backward_first_order,
    _forward_backward_first_order_time_varying,
    _score_first_order_imm,
    _score_fragmented,
    _score_stationary,
)
from .state_space_utils import (
    _as_log_probs,
    _candidate_log_masses,
    _first_order_imm_content_diagnostics,
    _full_grid_normalized_pairwise_gaussian_log_prob,
    _gaussian_transition_matrix,
    _mass_retaining_candidate_indices,
    _mean_entropy,
    _mode_transition_matrix,
    _pairwise_gaussian_log_prob,
    _per_bin_sigma,
    _restrict_candidates_to_valid_bins,
    _scaled_emissions,
    _top_candidate_indices,
    _valid_bin_count,
    _valid_bin_mask_from_occupancy,
    _validate_candidate_indices,
)

# The StateSpaceReplayModel.score implementation in state_space_model is already
# duration- and occupancy-aware. Mark it before legacy runtime patch modules
# inspect the public import surface so they do not replace it with older scorers.
StateSpaceReplayModel.score._native_duration_occupancy_aware = True
_apply_momentum_prediction_decay_validation_patch()

__all__ = [
    "EventScore",
    "GoalStateSpaceReplayModel",
    "LOG_ZERO",
    "StateSpaceDecoderConfig",
    "StateSpaceReplayModel",
    "_advance_imm_pair_log_alpha",
    "_advance_momentum_pair",
    "_apply_transition",
    "_apply_transition_backward",
    "_as_log_probs",
    "_augment_candidates_with_momentum_predictions",
    "_backward_imm_pair",
    "_backward_imm_pair_for_mode",
    "_backward_momentum_pair",
    "_candidate_log_masses",
    "_candidate_evidence_support_label",
    "_candidate_selection_label",
    "_candidate_support_config_diagnostics",
    "_first_order_imm_content_diagnostics",
    "_forward_backward_first_order",
    "_forward_backward_first_order_time_varying",
    "_full_grid_normalized_pairwise_gaussian_log_prob",
    "_displacement_lattice",
    "_gaussian_transition_matrix",
    "_init_imm_pair_log_alpha",
    "_init_pair_log_alpha",
    "_mass_retaining_candidate_indices",
    "_mean_entropy",
    "_mode_transition_matrix",
    "_normalize_log_weights",
    "_pairwise_gaussian_log_prob",
    "_per_bin_sigma",
    "_posterior_diagnostics",
    "_restrict_candidates_to_valid_bins",
    "_scaled_emissions",
    "_score_first_order_imm",
    "_score_fragmented",
    "_score_displacement_momentum_exact",
    "_score_displacement_imm_exact",
    "_score_imm_candidates",
    "_score_momentum_candidates",
    "_score_sparse_momentum_exact",
    "_score_trajectory_imm_exact_sparse",
    "_score_stationary",
    "_top_candidate_indices",
    "_valid_bin_count",
    "_valid_bin_mask_from_occupancy",
    "_validate_candidate_indices",
]
