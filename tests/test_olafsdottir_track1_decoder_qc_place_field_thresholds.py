from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    src_path = repo_root / "src"
    scripts_path = repo_root / "scripts"
    for path in (src_path, scripts_path):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    module_path = scripts_path / "summarize_olafsdottir_track1_decoder_qc.py"
    spec = importlib.util.spec_from_file_location(
        "summarize_olafsdottir_track1_decoder_qc_thresholds", module_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_unit_qc_applies_place_information_and_peak_rate_thresholds() -> None:
    module = _load_module()
    times = np.arange(0.0, 1.0, 0.1)
    linearized = pd.DataFrame(
        {
            "time_s": times,
            "linear_position_cm": np.linspace(0.0, 9.0, times.size),
            "valid_position": True,
        }
    )
    spike_times = np.array([0.05, 0.25, 0.45, 0.65, 0.85], dtype=float)
    spikes = module.TrackSpikes(
        spike_times_s=spike_times,
        unit_ids=np.ones(spike_times.shape, dtype=int),
        units=(1,),
    )
    common = {
        "animal": "RTEST",
        "date": "2026-01-01",
        "track_session": "track1",
        "linearized": linearized,
        "spikes": spikes,
        "position_bin_size_cm": 2.0,
        "min_unit_spikes": 1,
        "min_unit_mean_rate_hz": 0.0,
        "smoothing_bins": 1,
    }

    passing, _ = module.unit_qc_table(
        **common,
        min_place_information_bits=0.0,
        min_place_peak_rate_hz=0.0,
    )
    assert bool(passing.loc[0, "unit_qc_passed"])
    assert passing.attrs["n_place_like_units"] == 1

    below_information, _ = module.unit_qc_table(
        **common,
        min_place_information_bits=1.0e6,
        min_place_peak_rate_hz=0.0,
    )
    assert not bool(below_information.loc[0, "unit_qc_passed"])
    assert below_information.attrs["n_place_like_units"] == 0

    below_peak_rate, _ = module.unit_qc_table(
        **common,
        min_place_information_bits=0.0,
        min_place_peak_rate_hz=1.0e6,
    )
    assert not bool(below_peak_rate.loc[0, "unit_qc_passed"])
    assert below_peak_rate.attrs["n_place_like_units"] == 0
