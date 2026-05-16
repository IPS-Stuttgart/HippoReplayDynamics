"""State-space and IMM replay benchmarks for hippocampal open-field data."""
# ruff: noqa: E402

from . import ground_truth as _ground_truth
from . import score_metadata as _score_metadata
from . import simulation_recovery as _simulation_recovery

# Keep score-table metadata and post-hoc decoding consistent before public
# symbols are imported from the patched modules.
_score_metadata.apply_model_hyperparam_patch()

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
from .encoding import EncodingConfig, EncodingModel, build_emissions, fit_place_field_encoding
from .evidence_reporting import patch_simulation_recovery_module as _patch_simulation_recovery_module
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

# Ensure replay dynamics use center-to-center transition durations when replay
# emissions include a partial final bin.
_apply_duration_dynamics_patch()
_apply_state_space_imm_duration_patch()
from .encoding import build_emissions as build_emissions  # noqa: E402,F401,F811

# Keep synthetic recovery summaries from mixing exact evidences with truncated
# candidate lower bounds.
_patch_simulation_recovery_module(_simulation_recovery)
SimulationRecoveryConfig = _simulation_recovery.SimulationRecoveryConfig
SimulationRecoveryResult = _simulation_recovery.SimulationRecoveryResult
run_session_simulation_recovery = _simulation_recovery.run_session_simulation_recovery

__all__ = [
    "BenchmarkConfig",
    "BenchmarkResult",
    "CandidateKinematicModel",
    "ClusterlessMarkConfig",
    "ClusterlessMarkEncoding",
    "ClusterlessStateSpaceReplayModel",
    "DiffusionModel",
    "EncodingConfig",
    "EncodingModel",
    "EventScore",
    "GroundTruthConfig",
    "PyRecEstGoalParticleModel",
    "PyRecEstSweepConfig",
    "PyRecEstSweepResult",
    "RandomModel",
    "ReplaySession",
    "SimulationRecoveryConfig",
    "SimulationRecoveryResult",
    "StationaryModel",
    "build_emissions",
    "build_clusterless_mark_emissions",
    "compare_scores_to_ground_truth",
    "fit_place_field_encoding",
    "fit_clusterless_mark_encoding",
    "generate_behavioral_ground_truth",
    "infer_well_locations",
    "label_session_behavioral_ground_truth",
    "load_open_field_sessions",
    "run_open_field_benchmark",
    "run_pyrecest_parameter_sweep",
    "run_session_simulation_recovery",
    "score_model",
    "write_pyrecest_sweep_outputs",
]
