"""State-space and IMM replay benchmarks for hippocampal open-field data."""
# ruff: noqa: E402

from . import accuracy_model_probability_status_patch as _accuracy_model_probability_status_patch
from . import accuracy_replay_gain_gamma_patch as _accuracy_replay_gain_gamma_patch
from . import advanced_result_empty_threshold_patch as _advanced_result_empty_threshold_patch
from . import advanced_result_quantile_array_patch as _advanced_result_quantile_array_patch
from . import advanced_result_status_patch as _advanced_result_status_patch
from . import advanced_result_threshold_validation as _advanced_result_threshold_validation
from . import advanced_result_wrong_map_bootstrap_patch as _advanced_result_wrong_map_bootstrap_patch
from . import axona_data_end_footer as _axona_data_end_footer
from . import benchmark_cell_split_metadata as _benchmark_cell_split_metadata
from . import benchmark_mark_cell_id_validation as _benchmark_mark_cell_id_validation
from . import benchmark_metadata_scope_patch as _benchmark_metadata_scope_patch
from . import benchmark_relative_grouping as _benchmark_relative_grouping
from . import bidirectional_infinite_evidence_patch as _bidirectional_infinite_evidence_patch
from . import bma_options_patch as _bma_options_patch
from . import candidate_log_mass_validation as _candidate_log_mass_validation
from . import candidate_support_normalization_validation as _candidate_support_normalization_validation
from . import candidate_support_quality_patch as _candidate_support_quality_patch
from . import cell_split_hashable_grouping as _cell_split_hashable_grouping
from . import cli_float_values_validation as _cli_float_values_validation
from . import clusterless_config_validation as _clusterless_config_validation
from . import clusterless_ground_truth as _clusterless_ground_truth
from . import clusterless_mark_group_validation as _clusterless_mark_group_validation
from . import data_cell_id_validation as _data_cell_id_validation
from . import displacement_imm_decay_validation as _displacement_imm_decay_validation
from . import duration_candidate_metadata_patch as _duration_candidate_metadata_patch
from . import duration_occupancy_metadata_guard as _duration_occupancy_metadata_guard
from . import duration_occupancy_mode_transition_validation as _duration_occupancy_mode_transition_validation
from . import emission_cell_id_validation as _emission_cell_id_validation
from . import encoding_grid_extra_columns as _encoding_grid_extra_columns
from . import encoding_select_cells_validation as _encoding_select_cells_validation
from . import evidence_status_coercion as _evidence_status_coercion
from . import first_order_imm_diagnostics_validation as _first_order_imm_diagnostics_validation
from . import goal_state_space_integration as _goal_state_space_integration
from . import ground_truth as _ground_truth
from . import ground_truth_cell_id_metadata as _ground_truth_cell_id_metadata
from . import ground_truth_cell_split_strategy as _ground_truth_cell_split_strategy
from . import ground_truth_float_metadata as _ground_truth_float_metadata
from . import ground_truth_integer_metadata as _ground_truth_integer_metadata
from . import ground_truth_sensitivity_metrics as _ground_truth_sensitivity_metrics
from . import ground_truth_window_scope as _ground_truth_window_scope
from . import improved_model_evidence_registry_patch as _improved_model_evidence_registry_patch
from . import kd_encoding_config_validation as _kd_encoding_config_validation
from . import kd_impossible_emission_patch as _kd_impossible_emission_patch
from . import kd_single_bin_momentum_patch2 as _kd_single_bin_momentum_patch2
from . import latent_path_validation as _latent_path_validation
from . import mark_complex_validation as _mark_complex_validation
from . import model_averaged_endpoint_scoping as _model_averaged_endpoint_scoping
from . import model_parameter_validation as _model_parameter_validation
from . import observation_sweep_config_validation as _observation_sweep_config_validation
from . import occupancy_candidate_support as _occupancy_candidate_support
from . import olafsdottir_header_float_patch as _olafsdottir_header_float_patch
from . import olafsdottir_zip_safety as _olafsdottir_zip_safety
from . import position_decoding_config_validation as _position_decoding_config_validation
from . import posterior_calibration_summary_patch as _posterior_calibration_summary_patch
from . import pyrecest_bin_center_validation as _pyrecest_bin_center_validation
from . import pyrecest_numeric_metadata_guard as _pyrecest_numeric_metadata_guard
from . import pyrecest_score_metadata as _pyrecest_score_metadata
from . import recovery_diagnostics_bool_patch as _recovery_diagnostics_bool_patch
from . import reverse_time_terminal_guard as _reverse_time_terminal_guard
from . import score_metadata as _score_metadata
from . import score_metadata_bool_validation as _score_metadata_bool_validation
from . import shuffle_spike_time_order as _shuffle_spike_time_order
from . import simulation_best_row_flags as _simulation_best_row_flags
from . import simulation_recovery as _simulation_recovery
from . import simulation_recovery_count_validation as _simulation_recovery_count_validation
from . import simulation_recovery_event_count as _simulation_recovery_event_count
from . import simulation_recovery_runtime_limits as _simulation_recovery_runtime_limits
from . import sparse_momentum_duration_validation as _sparse_momentum_duration_validation
from . import sparse_momentum_single_bin_diagnostics as _sparse_momentum_single_bin_diagnostics
from . import spike_rate_metadata as _spike_rate_metadata
from . import state_space_bin_center_validation as _state_space_bin_center_validation
from . import state_space_bin_count_validation as _state_space_bin_count_validation
from . import time_order_patch as _time_order_patch
from . import well_label_shuffle_patch as _well_label_shuffle_patch
from . import wrapper_return_trajectory as _wrapper_return_trajectory

# Keep score-table metadata and post-hoc decoding consistent before public
# symbols are imported from the patched modules.
_state_space_bin_count_validation.apply_state_space_bin_count_validation_patch()
_score_metadata.apply_model_hyperparam_patch()
_score_metadata_bool_validation.apply_score_metadata_bool_validation_patch()
_model_parameter_validation.apply_model_parameter_validation_patch()
_displacement_imm_decay_validation.apply_displacement_imm_decay_validation_patch()
_candidate_support_quality_patch.apply_candidate_support_quality_patch()
_benchmark_cell_split_metadata.apply_benchmark_cell_split_metadata_patch()
_benchmark_metadata_scope_patch.apply_benchmark_metadata_scope_patch()
_cell_split_hashable_grouping.apply_cell_split_hashable_grouping_patch()
_benchmark_relative_grouping.apply_benchmark_relative_grouping_patch()
_benchmark_mark_cell_id_validation.apply_benchmark_mark_cell_id_validation_patch()
_clusterless_ground_truth.apply_clusterless_ground_truth_patch()
_bma_options_patch.apply_bma_options_patch()
_pyrecest_numeric_metadata_guard.apply_pyrecest_numeric_metadata_guard_patch()
_pyrecest_score_metadata.apply_pyrecest_score_metadata_patch()
_pyrecest_bin_center_validation.apply_pyrecest_bin_center_validation_patch()
_goal_state_space_integration.apply_goal_state_space_patch()
_spike_rate_metadata.apply_spike_rate_metadata_patch()
_time_order_patch.apply_reverse_emission_time_patch()
_clusterless_config_validation.apply_clusterless_encoding_config_validation_patch()
_data_cell_id_validation.apply_data_cell_id_validation_patch()
_mark_complex_validation.apply_mark_complex_validation_patch()
_encoding_grid_extra_columns.apply_encoding_grid_extra_columns_patch()
_position_decoding_config_validation.apply_position_decoding_config_validation_patch()
_duration_candidate_metadata_patch.apply_duration_candidate_metadata_patch()
_ground_truth_window_scope.apply_ground_truth_window_scope_patch()
_ground_truth_integer_metadata.apply_ground_truth_integer_metadata_patch()
_ground_truth_float_metadata.apply_ground_truth_float_metadata_patch()
_ground_truth_cell_id_metadata.apply_ground_truth_cell_id_metadata_patch()
_ground_truth_cell_split_strategy.apply_ground_truth_cell_split_strategy_patch()
_simulation_recovery_runtime_limits.apply_simulation_recovery_runtime_limit_validation_patch()
_simulation_recovery_count_validation.apply_simulation_recovery_count_validation_patch()
_reverse_time_terminal_guard.apply_reverse_time_terminal_guard_patch()
_wrapper_return_trajectory.apply_wrapper_return_trajectory_patch()
_shuffle_spike_time_order.apply_shuffle_spike_time_order_patch()
_well_label_shuffle_patch.apply_well_label_shuffle_patch()
_olafsdottir_header_float_patch.apply_olafsdottir_header_float_patch()

from .benchmarks import BenchmarkConfig, BenchmarkResult, run_open_field_benchmark
from .clusterless import (
    ClusterlessMarkConfig,
    ClusterlessMarkEncoding,
    ClusterlessStateSpaceReplayModel,
    build_clusterless_mark_emissions,
    fit_clusterless_mark_encoding,
)
from .data import ReplaySession, load_open_field_sessions
from .duration_dynamics import apply_duration_dynamics_patch as _apply_duration_dynamics_patch
from .duration_occupancy import apply_duration_occupancy_patch as _apply_duration_occupancy_patch
from .encoding import EncodingConfig, EncodingModel, build_emissions, fit_place_field_encoding
from .evidence_reporting import patch_simulation_recovery_module as _patch_simulation_recovery_module
from .goal_state_space import GoalStateSpaceReplayModel
from .ground_truth import (
    GroundTruthConfig,
    compare_scores_to_ground_truth,
    generate_behavioral_ground_truth,
    infer_well_locations,
    label_session_behavioral_ground_truth,
)
from .ground_truth_candidate_support import (
    apply_ground_truth_candidate_support_patch as _apply_ground_truth_candidate_support_patch,
)
from .models import (
    CandidateKinematicModel,
    DiffusionModel,
    EventScore,
    RandomModel,
    StationaryModel,
    score_model,
)
from .pyrecest_models import PyRecEstGoalParticleModel
from .simulation_recovery_trajectory_imm import (
    apply_trajectory_imm_recovery_patch as _apply_trajectory_imm_recovery_patch,
)
from .state_space_imm_duration import apply_state_space_imm_duration_patch as _apply_state_space_imm_duration_patch
from .sweeps import (
    PyRecEstSweepConfig,
    PyRecEstSweepResult,
    run_pyrecest_parameter_sweep,
    write_pyrecest_sweep_outputs,
)

# Keep the public ground-truth module backward-compatible with legacy
# model-evidence score tables that used short metadata column names.
_ground_truth._encoding_config_for_scores = _score_metadata.encoding_config_for_scores
_ground_truth._emission_config_for_scores = _score_metadata.emission_config_for_scores
_ground_truth_sensitivity_metrics.apply_ground_truth_sensitivity_metric_patch(_ground_truth)
_apply_ground_truth_candidate_support_patch()


def _synchronize_duration_patched_emission_builders() -> None:
    """Update modules that imported emission builders before duration patching.

    Package imports intentionally load benchmark and ground-truth entry points
    before the duration dynamics patch is installed. Those modules import
    ``build_emissions`` by value, so their aliases would otherwise continue to
    point at the unwrapped builder and silently drop transition-duration
    metadata for partial final bins.
    """

    import sys

    from . import encoding as _encoding_module
    from . import kd_reference as _kd_reference_module

    for module in list(sys.modules.values()):
        module_name = getattr(module, "__name__", "")
        if not module_name.startswith("hipporeplayimm"):
            continue
        if hasattr(module, "build_emissions"):
            module.build_emissions = _encoding_module.build_emissions
        if hasattr(module, "build_kd_emissions"):
            module.build_kd_emissions = _kd_reference_module.build_kd_emissions


def apply_runtime_patches() -> None:
    """Install runtime compatibility patches in the package-defined order.

    The package still applies these patches during import for backward
    compatibility.  Exposing the operation as an idempotent public hook makes
    the patch order testable and gives downstream scripts a supported way to
    refresh module-level aliases after direct lower-level imports.
    """

    _state_space_bin_count_validation.apply_state_space_bin_count_validation_patch()
    _score_metadata.apply_model_hyperparam_patch()
    _score_metadata_bool_validation.apply_score_metadata_bool_validation_patch()
    _model_parameter_validation.apply_model_parameter_validation_patch()
    _displacement_imm_decay_validation.apply_displacement_imm_decay_validation_patch()
    _candidate_support_quality_patch.apply_candidate_support_quality_patch()
    _benchmark_cell_split_metadata.apply_benchmark_cell_split_metadata_patch()
    _benchmark_metadata_scope_patch.apply_benchmark_metadata_scope_patch()
    _cell_split_hashable_grouping.apply_cell_split_hashable_grouping_patch()
    _benchmark_relative_grouping.apply_benchmark_relative_grouping_patch()
    _benchmark_mark_cell_id_validation.apply_benchmark_mark_cell_id_validation_patch()
    _clusterless_ground_truth.apply_clusterless_ground_truth_patch()
    _bma_options_patch.apply_bma_options_patch()
    _pyrecest_numeric_metadata_guard.apply_pyrecest_numeric_metadata_guard_patch()
    _pyrecest_score_metadata.apply_pyrecest_score_metadata_patch()
    _pyrecest_bin_center_validation.apply_pyrecest_bin_center_validation_patch()
    _goal_state_space_integration.apply_goal_state_space_patch()
    _spike_rate_metadata.apply_spike_rate_metadata_patch()
    _clusterless_config_validation.apply_clusterless_encoding_config_validation_patch()
    _data_cell_id_validation.apply_data_cell_id_validation_patch()
    _mark_complex_validation.apply_mark_complex_validation_patch()
    _clusterless_mark_group_validation.apply_clusterless_mark_group_validation_patch()
    _encoding_select_cells_validation.apply_encoding_select_cells_validation_patch()
    _encoding_grid_extra_columns.apply_encoding_grid_extra_columns_patch()
    _position_decoding_config_validation.apply_position_decoding_config_validation_patch()
    _duration_candidate_metadata_patch.apply_duration_candidate_metadata_patch()
    _ground_truth._encoding_config_for_scores = _score_metadata.encoding_config_for_scores
    _ground_truth._emission_config_for_scores = _score_metadata.emission_config_for_scores
    _ground_truth_integer_metadata.apply_ground_truth_integer_metadata_patch()
    _ground_truth_float_metadata.apply_ground_truth_float_metadata_patch()
    _ground_truth_cell_id_metadata.apply_ground_truth_cell_id_metadata_patch()
    _ground_truth_cell_split_strategy.apply_ground_truth_cell_split_strategy_patch()
    _ground_truth_sensitivity_metrics.apply_ground_truth_sensitivity_metric_patch(_ground_truth)
    _apply_ground_truth_candidate_support_patch()
    _occupancy_candidate_support.apply_occupancy_candidate_support_patch()
    _apply_duration_dynamics_patch()
    _emission_cell_id_validation.apply_emission_cell_id_validation_patch()
    _kd_encoding_config_validation.apply_kd_encoding_config_validation_patch()
    _kd_impossible_emission_patch.apply_kd_impossible_emission_patch()
    _kd_single_bin_momentum_patch2.apply_kd_single_bin_momentum_patch2()
    _apply_state_space_imm_duration_patch()
    _apply_duration_occupancy_patch()
    _duration_occupancy_mode_transition_validation.apply_duration_occupancy_mode_transition_validation_patch()
    _first_order_imm_diagnostics_validation.apply_first_order_imm_diagnostics_validation_patch()
    _candidate_log_mass_validation.apply_candidate_log_mass_validation_patch()
    _candidate_support_normalization_validation.apply_candidate_support_normalization_validation_patch()
    _state_space_bin_center_validation.apply_state_space_bin_center_validation_patch()
    _duration_occupancy_metadata_guard.apply_duration_occupancy_metadata_guard_patch()
    _sparse_momentum_duration_validation.apply_sparse_momentum_duration_validation_patch()
    _sparse_momentum_single_bin_diagnostics.apply_sparse_momentum_single_bin_diagnostics_patch()
    _time_order_patch.apply_reverse_emission_time_patch()
    _reverse_time_terminal_guard.apply_reverse_time_terminal_guard_patch()
    _wrapper_return_trajectory.apply_wrapper_return_trajectory_patch()
    _shuffle_spike_time_order.apply_shuffle_spike_time_order_patch()
    _well_label_shuffle_patch.apply_well_label_shuffle_patch()
    _axona_data_end_footer.apply_axona_data_end_footer_patch()
    _olafsdottir_header_float_patch.apply_olafsdottir_header_float_patch()
    _olafsdottir_zip_safety.apply_olafsdottir_zip_safety_patch()
    _synchronize_duration_patched_emission_builders()
    _patch_simulation_recovery_module(_simulation_recovery)
    _simulation_best_row_flags.apply_simulation_best_row_flags_patch()
    _evidence_status_coercion.apply_evidence_status_coercion_patch()
    _accuracy_model_probability_status_patch.apply_model_probability_status_patch()
    _accuracy_replay_gain_gamma_patch.apply_accuracy_replay_gain_gamma_patch()
    _advanced_result_status_patch.apply_advanced_result_status_patch()
    _advanced_result_empty_threshold_patch.apply_advanced_result_empty_threshold_patch()
    _advanced_result_threshold_validation.apply_advanced_result_threshold_validation_patch()
    _advanced_result_wrong_map_bootstrap_patch.apply_wrong_map_rat_bootstrap_patch()
    _advanced_result_quantile_array_patch.apply_advanced_result_quantile_array_patch()
    _posterior_calibration_summary_patch.apply_posterior_calibration_summary_patch()
    _latent_path_validation.apply_latent_path_validation_patch()
    _simulation_recovery_runtime_limits.apply_simulation_recovery_runtime_limit_validation_patch()
    _simulation_recovery_count_validation.apply_simulation_recovery_count_validation_patch()
    _apply_trajectory_imm_recovery_patch()
    _simulation_recovery_event_count.apply_simulation_recovery_event_count_patch()
    _recovery_diagnostics_bool_patch.apply_recovery_diagnostics_bool_patch()
    _model_averaged_endpoint_scoping.apply_model_averaged_endpoint_scoping_patch()
    _ground_truth_window_scope.apply_ground_truth_window_scope_patch()
    _improved_model_evidence_registry_patch.apply_improved_model_evidence_registry_patch()
    _observation_sweep_config_validation.apply_observation_sweep_config_validation_patch()
    _cli_float_values_validation.apply_cli_float_values_validation_patch()
    _bidirectional_infinite_evidence_patch.apply_bidirectional_infinite_evidence_patch()


# Ensure replay dynamics use center-to-center transition durations when replay
# emissions include a partial final bin.
apply_runtime_patches()
from .encoding import build_emissions as build_emissions  # noqa: E402,F401,F811
from .ground_truth import compare_scores_to_ground_truth as compare_scores_to_ground_truth  # noqa: E402,F401,F811

# Keep synthetic recovery summaries from mixing exact evidences with truncated
# candidate lower bounds.
SimulationRecoveryConfig = _simulation_recovery.SimulationRecoveryConfig
SimulationRecoveryResult = _simulation_recovery.SimulationRecoveryResult
run_session_simulation_recovery = _simulation_recovery.run_session_simulation_recovery

__all__ = [
    'BenchmarkConfig',
    'BenchmarkResult',
    'CandidateKinematicModel',
    'ClusterlessMarkConfig',
    'ClusterlessMarkEncoding',
    'ClusterlessStateSpaceReplayModel',
    'DiffusionModel',
    'EncodingConfig',
    'EncodingModel',
    'EventScore',
    'GoalStateSpaceReplayModel',
    'GroundTruthConfig',
    'PyRecEstGoalParticleModel',
    'PyRecEstSweepConfig',
    'PyRecEstSweepResult',
    'RandomModel',
    'ReplaySession',
    'SimulationRecoveryConfig',
    'SimulationRecoveryResult',
    'StationaryModel',
    'apply_runtime_patches',
    'build_clusterless_mark_emissions',
    'build_emissions',
    'compare_scores_to_ground_truth',
    'fit_clusterless_mark_encoding',
    'fit_place_field_encoding',
    'generate_behavioral_ground_truth',
    'infer_well_locations',
    'label_session_behavioral_ground_truth',
    'load_open_field_sessions',
    'run_open_field_benchmark',
    'run_pyrecest_parameter_sweep',
    'run_session_simulation_recovery',
    'score_model',
    'write_pyrecest_sweep_outputs',
]
