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
        "summarize_olafsdottir_track1_decoder_occupancy_isolation",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_training_occupancy_is_clipped_at_fold_boundary() -> None:
    module = _load_module()
    times = np.array([0.0, 0.06, 0.12], dtype=float)
    linear = np.zeros(times.shape, dtype=float)
    intervals = [(0.0, 0.1)]
    valid = module._impl.sample_mask_in_intervals(times, intervals)
    edges = np.array([0.0, 1.0], dtype=float)

    unclipped = module._impl.occupancy_seconds(linear, times, valid, edges)
    clipped = module._occupancy_seconds_in_intervals(
        linear,
        times,
        valid,
        edges,
        intervals,
    )

    np.testing.assert_allclose(unclipped, np.array([0.12]))
    np.testing.assert_allclose(clipped, np.array([0.10]))


def test_crossval_uses_interval_clipped_training_occupancy(monkeypatch) -> None:
    module = _load_module()
    times = np.array([0.0, 0.06, 0.12, 0.18, 0.24], dtype=float)
    linearized = pd.DataFrame(
        {
            "time_s": times,
            "linear_position_cm": np.array(
                [0.25, 0.25, 1.25, 1.25, 2.25],
                dtype=float,
            ),
            "valid_position": True,
        }
    )
    spikes = module.TrackSpikes(
        spike_times_s=np.array([0.03, 0.15, 0.22], dtype=float),
        unit_ids=np.ones(3, dtype=int),
        units=(1,),
    )
    occupancy_totals: list[float] = []
    original = module._occupancy_seconds_in_intervals

    def record_occupancy(
        linear,
        sample_times,
        valid,
        edges,
        intervals,
    ):
        occupancy = original(
            linear,
            sample_times,
            valid,
            edges,
            intervals,
        )
        occupancy_totals.append(float(np.sum(occupancy)))
        return occupancy

    monkeypatch.setattr(
        module,
        "_occupancy_seconds_in_intervals",
        record_occupancy,
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
    np.testing.assert_allclose(occupancy_totals, np.array([0.06, 0.20]))
