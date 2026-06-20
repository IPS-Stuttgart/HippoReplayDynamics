from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import textwrap


ROOT = Path(__file__).resolve().parents[1]


def test_improved_model_evidence_registry_shim_accepts_extra_models() -> None:
    code = r'''
import argparse
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys

root = Path.cwd()
sys.path.insert(0, str(root / "src"))
sys.path.insert(0, str(root / "scripts"))
script = root / "scripts" / "benchmark_model_evidence_improved.py"
spec = importlib.util.spec_from_file_location("benchmark_model_evidence_improved_registry", script)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)

args = argparse.Namespace(
    models=(
        "sorted-spike-state-space-trajectory-imm-anchored-exact-sparse "
        "sorted-spike-state-space-trajectory-imm-low-leak-exact-sparse "
        "sorted-spike-state-space-trajectory-imm-persistent-exact-sparse "
        "sorted-spike-state-space-displacement-momentum "
        "clusterless-state-space-displacement-momentum"
    ),
    include_clusterless_defaults=False,
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
    state_space_imm_switch_tau_s=0.0,
    time_bin_s=0.003,
    state_space_momentum_sigma_cm_sqrt_s=43.0,
    state_space_momentum_initial_sigma_cm_sqrt_s=44.0,
    state_space_momentum_velocity_decay=0.8,
    state_space_momentum_velocity_decay_tau_s=0.0,
    state_space_momentum_candidate_top_k=17,
    state_space_momentum_candidate_mass_threshold=None,
    state_space_momentum_candidate_min_k=1,
    state_space_momentum_candidate_max_k=0,
    state_space_momentum_predicted_candidate_top_k=5,
    state_space_momentum_candidate_source="emission",
    state_space_valid_occupancy_threshold_s=0.0,
    clusterless_mark_likelihood="local-kde",
    goal_state_space_transition_sigma_cm_sqrt_s=85.0,
    goal_state_space_drift_speed_cm_s=400.0,
    goal_state_space_max_step_sigma=4.0,
    valid_state_min_occupancy_s=0.02,
    valid_state_top_occupancy_fraction=None,
    valid_state_sigma_cm=5.0,
    valid_state_max_step_sigma=4.0,
    valid_state_grid_diagonal_neighbors=True,
    valid_state_grid_stay_probability=0.0,
)
models = module._models(args, SimpleNamespace(session_id="Rat1/Open1"), encoding=None)
assert list(models) == [
    "sorted-spike-state-space-trajectory-imm-anchored-exact-sparse",
    "sorted-spike-state-space-trajectory-imm-low-leak-exact-sparse",
    "sorted-spike-state-space-trajectory-imm-persistent-exact-sparse",
    "sorted-spike-state-space-displacement-momentum",
    "clusterless-state-space-displacement-momentum",
]
assert models["sorted-spike-state-space-trajectory-imm-anchored-exact-sparse"].config.trajectory_imm_momentum_initial_probability == 0.05
assert models["sorted-spike-state-space-trajectory-imm-low-leak-exact-sparse"].config.trajectory_imm_momentum_switch_probability == 0.001
assert models["sorted-spike-state-space-trajectory-imm-persistent-exact-sparse"].config.trajectory_imm_mode_stickiness == 0.985
assert models["sorted-spike-state-space-displacement-momentum"].mode == "displacement-momentum"
assert models["clusterless-state-space-displacement-momentum"].mode == "displacement-momentum"
'''
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT / "src"), str(ROOT / "scripts"), env.get("PYTHONPATH", "")]
    )
    subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
