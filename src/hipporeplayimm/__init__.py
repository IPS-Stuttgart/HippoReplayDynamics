'''State-space and IMM replay benchmarks for hippocampal open-field data.'''
# ruff: noqa: E402

from . import benchmark_cell_split_metadata as _benchmark_cell_split_metadata
from . import candidate_support_quality_patch as _candidate_support_quality_patch
from . import clusterless_config_validation as _clusterless_config_validation
from . import clusterless_ground_truth as _clusterless_ground_truth
from . import goal_state_space_integration as _goal_state_space_integration
from . import ground_truth as _ground_truth
from . import ground_truth_bma_options as _ground_truth_bma_options
from . import ground_truth_sensitivity_metrics as _ground_truth_sensitivity_metrics
from . import improved_model_evidence_registry_patch as _improved_model_evidence_registry_patch
from . import model_averaged_endpoint_scoping as _model_averaged_endpoint_scoping
from . import occupancy_candidate_support as _occupancy_candidate_support
from . import pyrecest_score_metadata as _pyrecest_score_metadata
from . import score_metadata as _score_metadata
from . import simulation_recovery as _simulation_recovery
from . import spike_rate_metadata as _spike_rate_metadata
from . import time_order_patch as _time_order_patch

# Keep score-table metadata and post-hoc decoding consistent before public
# symbols are imported from the patched modules.
_score_metadata.apply_model_hyperparam_patch()
_candidate_support_quality_patch.apply_candidate_support_quality_patch()
_benchmark_cell_split_metadata.apply_benchmark_cell_split_metadata_patch()
_clusterless_ground_truth.apply_clusterless_ground_truth_patch()
_ground_truth_bma_options.apply_ground_truth_bma_options_patch()
_pyrecest_score_metadata.apply_pyrecest_score_metadata_patch()
_goal_state_space_integration.apply_goal_state_space_patch()
_spike_rate_metadata.apply_spike_rate_metadata_patch()
_time_order_patch.apply_reverse_emission_time_patch()
_clusterless_config_validation.apply_clusterless_encoding_config_validation_patch()

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
from .ground_truth_candidate_support import (
    apply_ground_truth_candidate_support_patch as _apply_ground_truth_candidate_support_patch,
)
from .ground_truth import (
    GroundTruthConfig,
    compare_scores_to_ground_truth,
    generate_behavioral_ground_truth,
    infer_well_locations,
    label_session_behavioral_ground_truth,
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

    _score_metadata.apply_model_hyperparam_patch()
    _candidate_support_quality_patch.apply_candidate_support_quality_patch()
    _benchmark_cell_split_metadata.apply_benchmark_cell_split_metadata_patch()
    _clusterless_ground_truth.apply_clusterless_ground_truth_patch()
    _ground_truth_bma_options.apply_ground_truth_bma_options_patch()
    _clusterless_config_validation.apply_clusterless_encoding_config_validation_patch()
    _ground_truth._encoding_config_for_scores = _score_metadata.encoding_config_for_scores
    _ground_truth._emission_config_for_scores = _score_metadata.emission_config_for_scores
    _ground_truth_sensitivity_metrics.apply_ground_truth_sensitivity_metric_patch(_ground_truth)
    _apply_ground_truth_candidate_support_patch()
    _occupancy_candidate_support.apply_occupancy_candidate_support_patch()
    _apply_duration_dynamics_patch()
    _apply_state_space_imm_duration_patch()
    _apply_duration_occupancy_patch()
    _time_order_patch.apply_reverse_emission_time_patch()
    _synchronize_duration_patched_emission_builders()
    _patch_simulation_recovery_module(_simulation_recovery)
    _apply_trajectory_imm_recovery_patch()
    _model_averaged_endpoint_scoping.apply_model_averaged_endpoint_scoping_patch()
    _improved_model_evidence_registry_patch.apply_improved_model_evidence_registry_patch()


# Ensure replay dynamics use center-to-center transition durations when replay
# emissions include a partial final bin.
apply_runtime_patches()
from .encoding import build_emissions as build_emissions  # noqa: E402,F401,F811

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
