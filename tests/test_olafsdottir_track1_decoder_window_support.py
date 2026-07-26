from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    src_path = repo_root / "src"
    scripts_path = repo_root / "scripts"
    for path in (src_path, scripts_path):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    module_path = scripts_path / "summarize_olafsdottir_track1_decoder_qc.py"
    spec = importlib.util.spec_from_file_location(
        "summarize_olafsdottir_track1_window_support",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _linearized_position() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time_s": np.array([0.00, 0.09, 0.18, 0.26], dtype=float),
            "linear_position_cm": np.array([0.0, 1.0, 2.0, 3.0], dtype=float),
            "valid_position": True,
        }
    )


def test_decode_windows_clip_final_bin_to_valid_position_support() -> None:
    module = _load_module()

    windows = module.decode_windows(_linearized_position(), 0.10)

    assert windows["end_time_s"].iloc[-1] == pytest.approx(0.26)
    assert np.all(windows["end_time_s"].to_numpy(dtype=float) <= 0.26)
    assert windows["true_position_cm"].iloc[-1] == pytest.approx(3.0)

    spikes = module.TrackSpikes(
        spike_times_s=np.array([0.25, 0.28], dtype=float),
        unit_ids=np.array([1, 1], dtype=int),
        units=(1,),
    )
    last = windows.iloc[-1]
    counts = module._impl.spike_counts_for_window(
        spikes,
        (1,),
        float(last["start_time_s"]),
        float(last["end_time_s"]),
    )

    np.testing.assert_array_equal(counts, np.array([1.0]))


def test_decode_windows_reject_nonpositive_window_size() -> None:
    module = _load_module()

    with pytest.raises(ValueError, match="decode_window_s"):
        module.decode_windows(_linearized_position(), 0.0)
