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
        "summarize_olafsdottir_track1_decoder_crossval_isolation",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_crossval_place_fields_use_training_spikes_only(monkeypatch) -> None:
    module = _load_module()
    sample_times = np.array([0.05, 0.15, 0.25, 0.34], dtype=float)
    linearized = pd.DataFrame(
        {
            "time_s": sample_times,
            "linear_position_cm": np.arange(sample_times.size, dtype=float),
            "valid_position": True,
        }
    )
    spike_times = np.array([0.10, 0.20, 0.30], dtype=float)
    spikes = module.TrackSpikes(
        spike_times_s=spike_times,
        unit_ids=np.ones(spike_times.shape, dtype=int),
        units=(1,),
    )
    fold_spike_times: list[np.ndarray] = []

    def record_training_spikes(
        *,
        linear,
        times,
        valid,
        spikes,
        unit_ids,
        edges,
        smoothing_bins,
        occupancy,
    ):
        del linear, times, valid, smoothing_bins, occupancy
        fold_spike_times.append(spikes.spike_times_s.copy())
        return np.ones((len(unit_ids), len(edges) - 1), dtype=float)

    monkeypatch.setattr(
        module,
        "_fit_place_fields_with_occupancy",
        record_training_spikes,
    )

    result = module.crossval_decode(
        linearized=linearized,
        spikes=spikes,
        unit_ids=(1,),
        crossval_folds=2,
        position_bin_size_cm=1.0,
        decode_window_s=0.1,
        smoothing_bins=1,
    )

    assert result["crossval_n_folds"] == 2
    assert len(fold_spike_times) == 2
    np.testing.assert_allclose(fold_spike_times[0], np.array([0.30]))
    np.testing.assert_allclose(fold_spike_times[1], np.array([0.10, 0.20]))
