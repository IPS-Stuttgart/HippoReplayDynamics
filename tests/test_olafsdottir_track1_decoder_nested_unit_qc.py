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
        "summarize_olafsdottir_track1_nested_unit_qc",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_crossval_selects_units_from_training_spikes_only(monkeypatch) -> None:
    module = _load_module()
    sample_times = np.array([0.05, 0.15, 0.25, 0.34], dtype=float)
    linearized = pd.DataFrame(
        {
            "time_s": sample_times,
            "linear_position_cm": np.arange(sample_times.size, dtype=float),
            "valid_position": True,
        }
    )
    spikes = module.TrackSpikes(
        spike_times_s=np.array([0.10, 0.20, 0.30], dtype=float),
        unit_ids=np.array([1, 1, 2], dtype=int),
        units=(1, 2),
    )
    selected_by_test_window: list[tuple[int, ...]] = []

    def constant_place_fields(
        *,
        linear,
        times,
        valid,
        spikes,
        unit_ids,
        edges,
        smoothing_bins,
    ):
        del linear, times, valid, spikes, smoothing_bins
        return np.ones((len(unit_ids), len(edges) - 1), dtype=float)

    original_spike_counts = module._impl.spike_counts_for_window

    def record_selected_units(spikes, unit_ids, start_s, end_s):
        selected_by_test_window.append(tuple(int(unit_id) for unit_id in unit_ids))
        return original_spike_counts(spikes, unit_ids, start_s, end_s)

    monkeypatch.setattr(module._impl, "fit_place_fields", constant_place_fields)
    monkeypatch.setattr(
        module._impl,
        "spike_counts_for_window",
        record_selected_units,
    )

    result = module.crossval_decode(
        linearized=linearized,
        spikes=spikes,
        unit_ids=(1, 2),
        crossval_folds=2,
        position_bin_size_cm=1.0,
        decode_window_s=0.1,
        smoothing_bins=1,
        min_unit_spikes=1,
        min_unit_mean_rate_hz=0.0,
        min_place_information_bits=0.0,
        min_place_peak_rate_hz=0.0,
    )

    assert result["crossval_n_folds"] == 2
    assert selected_by_test_window == [(2,), (2,), (1,)]


def test_session_context_uses_all_units_as_fold_candidates(monkeypatch) -> None:
    module = _load_module()
    sample_times = np.array([0.05, 0.15, 0.25, 0.34], dtype=float)
    linearized = pd.DataFrame(
        {
            "time_s": sample_times,
            "linear_position_cm": np.arange(sample_times.size, dtype=float),
            "valid_position": True,
        }
    )
    spikes = module.TrackSpikes(
        spike_times_s=np.array([0.10, 0.20, 0.30], dtype=float),
        unit_ids=np.array([1, 1, 2], dtype=int),
        units=(1, 2),
    )
    selected_by_test_window: list[tuple[int, ...]] = []

    def constant_place_fields(
        *,
        linear,
        times,
        valid,
        spikes,
        unit_ids,
        edges,
        smoothing_bins,
    ):
        del linear, times, valid, spikes, smoothing_bins
        return np.ones((len(unit_ids), len(edges) - 1), dtype=float)

    def record_selected_units(spikes, unit_ids, start_s, end_s):
        del spikes, start_s, end_s
        selected_by_test_window.append(tuple(int(unit_id) for unit_id in unit_ids))
        return np.zeros(len(unit_ids), dtype=float)

    monkeypatch.setattr(module._impl, "fit_place_fields", constant_place_fields)
    monkeypatch.setattr(
        module._impl,
        "spike_counts_for_window",
        record_selected_units,
    )

    token = module._crossval_unit_qc_context.set(
        {
            "min_unit_spikes": 1,
            "min_unit_mean_rate_hz": 0.0,
            "min_place_information_bits": 0.0,
            "min_place_peak_rate_hz": 0.0,
        }
    )
    try:
        module.crossval_decode(
            linearized=linearized,
            spikes=spikes,
            unit_ids=(1,),
            crossval_folds=2,
            position_bin_size_cm=1.0,
            decode_window_s=0.1,
            smoothing_bins=1,
        )
    finally:
        module._crossval_unit_qc_context.reset(token)

    assert selected_by_test_window == [(2,), (2,), (1,)]
