from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "marginalize_state_space_sweep.py"
sys.path.insert(0, str(_SCRIPT.parent))
_SPEC = importlib.util.spec_from_file_location("marginalize_state_space_sweep", _SCRIPT)
assert _SPEC is not None
marginalize_state_space_sweep = importlib.util.module_from_spec(_SPEC)
sys.modules["marginalize_state_space_sweep"] = marginalize_state_space_sweep
assert _SPEC.loader is not None
_SPEC.loader.exec_module(marginalize_state_space_sweep)


def _base_row(event_index: int, model: str, log_evidence: float) -> dict[str, object]:
    return {
        "status": "success",
        "session": "Rat1/Open1",
        "event_index": event_index,
        "model": model,
        "requested_model": model,
        "model_family": "trajectory",
        "log_evidence": log_evidence,
        "n_time": 3,
        "n_spikes": 7,
        "runtime_s": 0.1,
        "bin_size_cm": 6.0,
        "smoothing_sigma_bins": 2.0,
        "min_speed_cm_s": 5.0,
        "time_bin_s": 0.003,
    }


def test_marginalize_state_space_sweep_writes_marginalized_tables(tmp_path: Path):
    rows = []
    for event_index, values in {0: {50.0: 0.0, 60.0: -2.0}, 1: {50.0: -3.0, 60.0: -1.0}}.items():
        for diffusion_sigma, log_evidence in values.items():
            row = _base_row(event_index, "sorted-spike-state-space-diffusion", log_evidence)
            row["state_space_diffusion_sigma_cm_sqrt_s"] = diffusion_sigma
            rows.append(row)

    momentum_values = {
        0: {(100.0, 0.90): -1.0, (100.0, 0.95): -2.0, (110.0, 0.90): 1.0, (110.0, 0.95): 0.0},
        1: {(100.0, 0.90): -4.0, (100.0, 0.95): -3.0, (110.0, 0.90): -2.0, (110.0, 0.95): -1.0},
    }
    for event_index, values in momentum_values.items():
        for (momentum_sigma, decay), log_evidence in values.items():
            row = _base_row(event_index, "sorted-spike-state-space-momentum", log_evidence)
            row["state_space_momentum_sigma_cm_sqrt_s"] = momentum_sigma
            row["state_space_momentum_initial_sigma_cm_sqrt_s"] = 85.0
            row["state_space_momentum_velocity_decay"] = decay
            row["state_space_momentum_candidate_top_k"] = 128
            rows.append(row)

    input_csv = tmp_path / "state_space_evidence_sweep_event_scores.csv"
    pd.DataFrame(rows).to_csv(input_csv, index=False)

    out_dir = tmp_path / "marginalized"
    tables = marginalize_state_space_sweep.marginalize_sweep(input_csv, out_dir, prior="uniform")

    event_model_evidence = tables["event_model_evidence"]
    assert set(event_model_evidence["model"]) == {
        "sorted-spike-state-space-diffusion-marginalized",
        "sorted-spike-state-space-momentum-marginalized",
    }
    assert len(event_model_evidence) == 4

    diffusion_event0 = event_model_evidence[
        (event_model_evidence["event_index"] == 0)
        & (event_model_evidence["model"] == "sorted-spike-state-space-diffusion-marginalized")
    ].iloc[0]
    assert np.isclose(diffusion_event0["log_evidence"], logsumexp([0.0, -2.0]) - np.log(2.0))

    best_params = pd.read_csv(out_dir / "state_space_marginalized_gridsearch_best_params.csv")
    momentum_event0 = best_params[
        (best_params["event_index"] == 0)
        & (best_params["marginalized_model"] == "sorted-spike-state-space-momentum-marginalized")
    ].iloc[0]
    assert momentum_event0["best_state_space_momentum_sigma_cm_sqrt_s"] == 110.0
    assert momentum_event0["best_state_space_momentum_velocity_decay"] == 0.9

    priors = pd.read_csv(out_dir / "state_space_marginalized_prior_weights.csv")
    assert np.allclose(priors.groupby("marginalized_model")["prior_weight"].sum().to_numpy(), 1.0)
    assert (out_dir / "state_space_marginalized_model_evidence_summary.csv").exists()
    assert (out_dir / "state_space_marginalized_best_model_counts.csv").exists()
