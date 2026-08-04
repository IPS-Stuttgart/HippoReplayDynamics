from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


def _load_adapter_module():
    repo_root = Path(__file__).resolve().parents[1]
    src_path = repo_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    module_path = repo_root / "scripts" / "prepare_olafsdottir_ztrack_sessions.py"
    spec = importlib.util.spec_from_file_location(
        "prepare_olafsdottir_ztrack_sessions_trace_boundary",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_expanded_ripple_window_stops_at_trace_end() -> None:
    """An expanded event must not admit spikes beyond observed LFP support."""

    module = _load_adapter_module()
    traces = np.zeros((1, 10), dtype=float)
    traces[0, -1] = 4.0
    config = module.RippleDetectionConfig(
        channel_numbers=(1,),
        band_low_hz=150.0,
        band_high_hz=250.0,
        high_threshold_z=3.0,
        low_threshold_z=1.0,
        min_duration_s=0.01,
        max_duration_s=0.5,
        expand_to_s=0.4,
        exclude_sleep_onset_s=0.0,
        detector_mode="mean-envelope",
        consensus_min_channels=1,
    )

    events, event_table = module.detect_ripple_events_from_traces(
        traces,
        10.0,
        config,
    )

    assert events.shape == (1, 6)
    np.testing.assert_allclose(events[0, :3], np.array([0.7, 1.0, 0.9]))
    assert np.isclose(float(event_table.loc[0, "end_s"]), 1.0)

    quality = module.event_spike_quality(
        events,
        np.array([[0.95, 1.0], [1.05, 2.0]], dtype=float),
    )
    assert int(quality.loc[0, "n_spikes"]) == 1
