"""State-space and IMM replay benchmarks for hippocampal open-field data."""

from . import ground_truth as _ground_truth
from . import score_metadata as _score_metadata
from .benchmarks import BenchmarkConfig, BenchmarkResult, run_open_field_benchmark
from .data import ReplaySession, load_open_field_sessions
from .encoding import EncodingConfig, EncodingModel, build_emissions, fit_place_field_encoding
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
from .simulation_recovery import (
    SimulationRecoveryConfig,
    SimulationRecoveryResult,
    run_session_simulation_recovery,
)
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

__all__ = [
    "BenchmarkConfig",
    "BenchmarkResult",
    "CandidateKinematicModel",
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
    "compare_scores_to_ground_truth",
    "fit_place_field_encoding",
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
