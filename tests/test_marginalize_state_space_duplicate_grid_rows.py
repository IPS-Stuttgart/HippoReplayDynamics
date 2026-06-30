from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "marginalize_state_space_sweep.py"
sys.path.insert(0, str(_SCRIPT.parent))
_SPEC = importlib.util.spec_from_file_location("marginalize_state_space_sweep_duplicates", _SCRIPT)
assert _SPEC is not None
marginalize_state_space_sweep = importlib.util.module_from_spec(_SPEC)
sys.modules["marginalize_state_space_sweep_duplicates"] = marginalize_state_space_sweep
assert _SPEC.loader is not None
_SPEC.loader.exec_module(marginalize_state_space_sweep)


def test_marginalize_state_space_sweep_rejects_duplicate_grid_rows(tmp_path: Path):
    row = {
        "status": "success",
        "session": "Rat1/Open1",
        "event_index": 0,
        "model": "sorted-spike-state-space-diffusion",
        "log_evidence": -1.0,
        "state_space_diffusion_sigma_cm_sqrt_s": 50.0,
    }
    input_csv = tmp_path / "state_space_evidence_sweep_event_scores.csv"
    pd.DataFrame([row, row.copy()]).to_csv(input_csv, index=False)

    with pytest.raises(ValueError, match="duplicate grid rows"):
        marginalize_state_space_sweep.marginalize_sweep(
            input_csv,
            tmp_path / "marginalized",
            models=("diffusion",),
            prior="uniform",
            observation_parameters="none",
        )
