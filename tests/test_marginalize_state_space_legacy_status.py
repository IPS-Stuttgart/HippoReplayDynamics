from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "marginalize_state_space_sweep.py"
sys.path.insert(0, str(_SCRIPT.parent))
_SPEC = importlib.util.spec_from_file_location("marginalize_state_space_sweep", _SCRIPT)
assert _SPEC is not None
marginalize_state_space_sweep = importlib.util.module_from_spec(_SPEC)
sys.modules["marginalize_state_space_sweep"] = marginalize_state_space_sweep
assert _SPEC.loader is not None
_SPEC.loader.exec_module(marginalize_state_space_sweep)


def _diffusion_row(event_index: int, log_evidence: float, status: object) -> dict[str, object]:
    return {
        "status": status,
        "session": "Rat1/Open1",
        "event_index": event_index,
        "model": "sorted-spike-state-space-diffusion",
        "requested_model": "sorted-spike-state-space-diffusion",
        "model_family": "trajectory",
        "log_evidence": log_evidence,
        "n_time": 3,
        "n_spikes": 7,
        "runtime_s": 0.1,
        "bin_size_cm": 6.0,
        "smoothing_sigma_bins": 2.0,
        "min_speed_cm_s": 5.0,
        "time_bin_s": 0.003,
        "spike_rate_scale": 1.0,
        "state_space_diffusion_sigma_cm_sqrt_s": 50.0,
    }


def test_marginalize_state_space_sweep_keeps_legacy_missing_status_rows(tmp_path: Path):
    input_csv = tmp_path / "state_space_evidence_sweep_event_scores.csv"
    pd.DataFrame(
        [
            _diffusion_row(0, -1.0, ""),
            _diffusion_row(1, -2.0, pd.NA),
            _diffusion_row(0, 100.0, "failed"),
        ]
    ).to_csv(input_csv, index=False)

    tables = marginalize_state_space_sweep.marginalize_sweep(
        input_csv,
        tmp_path / "marginalized",
        models=("diffusion",),
        prior="uniform",
    )

    event_model_evidence = tables["event_model_evidence"].sort_values("event_index").reset_index(drop=True)
    assert event_model_evidence["event_index"].tolist() == [0, 1]
    assert np.allclose(event_model_evidence["log_evidence"].to_numpy(float), [-1.0, -2.0])
