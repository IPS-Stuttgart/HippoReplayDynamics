import argparse
import importlib.util
from pathlib import Path

import numpy as np

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_model_evidence.py"
_SPEC = importlib.util.spec_from_file_location("benchmark_model_evidence", _SCRIPT)
assert _SPEC is not None
benchmark_model_evidence = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(benchmark_model_evidence)

_events = benchmark_model_evidence._events
_family = benchmark_model_evidence._family
_models = benchmark_model_evidence._models


class _SessionStub:
    ripple_count = 10

    @staticmethod
    def ripple_indices_in_run():
        return np.array([2, 4, 6, 8], dtype=int)


def test_model_evidence_accepts_sorted_spike_state_space_models():
    args = argparse.Namespace(
        models="sorted-spike-state-space-diffusion sorted-spike-state-space-momentum sorted-spike-state-space-imm",
        candidate_top_k=64,
        stationary_sigma_cm=2.0,
        diffusion_sigma_cm=12.0,
        momentum_sigma_cm=12.0,
        velocity_decay=0.95,
        mode_stickiness=0.94,
        state_space_stationary_sigma_cm=1.5,
        state_space_diffusion_sigma_cm_sqrt_s=42.0,
        state_space_max_step_sigma=3.0,
        state_space_imm_mode_stickiness=0.91,
        state_space_momentum_sigma_cm_sqrt_s=43.0,
        state_space_momentum_initial_sigma_cm_sqrt_s=44.0,
        state_space_momentum_velocity_decay=0.8,
        state_space_momentum_candidate_top_k=17,
    )

    models = _models(args)

    assert list(models) == [
        "sorted-spike-state-space-diffusion",
        "sorted-spike-state-space-momentum",
        "sorted-spike-state-space-imm",
    ]
    assert models["sorted-spike-state-space-diffusion"].name == "sorted-spike-state-space-diffusion"
    assert models["sorted-spike-state-space-momentum"].name == "sorted-spike-state-space-momentum"
    assert models["sorted-spike-state-space-imm"].name == "sorted-spike-state-space-imm"
    assert models["sorted-spike-state-space-diffusion"].config.diffusion_sigma_cm_sqrt_s == 42.0
    assert models["sorted-spike-state-space-momentum"].config.momentum_sigma_cm_sqrt_s == 43.0
    assert models["sorted-spike-state-space-momentum"].config.momentum_initial_sigma_cm_sqrt_s == 44.0
    assert models["sorted-spike-state-space-momentum"].config.momentum_velocity_decay == 0.8
    assert models["sorted-spike-state-space-momentum"].config.momentum_candidate_top_k == 17
    assert models["sorted-spike-state-space-imm"].config.imm_mode_stickiness == 0.91


def test_model_evidence_classifies_state_space_families():
    assert _family("sorted-spike-state-space-stationary") == "nontrajectory"
    assert _family("sorted-spike-state-space-diffusion") == "trajectory"
    assert _family("sorted-spike-state-space-fragmented") == "trajectory"
    assert _family("sorted-spike-state-space-momentum") == "trajectory"
    assert _family("sorted-spike-state-space-imm") == "trajectory"


def test_model_evidence_run_event_selection_uses_session_event_ids():
    assert _events("run", _SessionStub()) == [2, 4, 6, 8]
    assert _events("run:1-2", _SessionStub()) == [4, 6]
