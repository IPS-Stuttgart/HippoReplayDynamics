from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pandas as pd


def _load_compare_module():
    repo_root = Path(__file__).resolve().parents[1]
    src_path = repo_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    module_path = repo_root / "scripts" / "compare_olafsdottir_1d_2d_trajectory_family.py"
    spec = importlib.util.spec_from_file_location("compare_olafsdottir_1d_2d_trajectory_family_legacy_status", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_compare_event_metrics_keep_legacy_missing_status_rows() -> None:
    module = _load_compare_module()
    rows = []
    for index, model in enumerate(module.EXACT_CORE_MODELS):
        rows.append(
            {
                "session": "R2142/ZTrack20140806",
                "event_index": 0,
                "model": model,
                "status": ["", pd.NA, "<NA>", "na", "success"][index],
                "log_evidence": float(index),
                "n_spikes": 12,
                "n_time": 6,
                "diagnostic_evidence_support": "exact_full_grid",
                "diagnostic_evidence_comparable": True,
                "diagnostic_evidence_comparison": "exact_model_evidence",
            }
        )

    metrics = module.event_level_metrics(pd.DataFrame(rows), margin_threshold=1.0)

    assert metrics.shape[0] == 1
    assert bool(metrics.loc[0, "complete_exact_core"])
