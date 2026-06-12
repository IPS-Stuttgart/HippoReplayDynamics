from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from hipporeplayimm.duration_dynamics import attach_duration_metadata, transition_durations_s
from hipporeplayimm.encoding import LogEmissionTensor


def _load_track_event_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "track_event.py"
    spec = importlib.util.spec_from_file_location(
        "track_event_prefix_metadata_under_test",
        path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prefix_emissions_preserves_bin_and_transition_durations() -> None:
    track_event = _load_track_event_module()
    emissions = LogEmissionTensor(
        log_likelihood=np.zeros((4, 2), dtype=float),
        spike_counts=np.zeros((4, 1), dtype=int),
        times=np.array([0.05, 0.15, 0.30, 0.50], dtype=float),
        dt=0.1,
        cell_ids=np.array([1], dtype=int),
        n_spikes=0,
        bin_durations=np.array([0.1, 0.1, 0.2, 0.2], dtype=float),
        transition_durations=np.array([0.1, 0.15, 0.20], dtype=float),
    )
    attach_duration_metadata(emissions)

    prefix = track_event._prefix_emissions(emissions, 3)

    np.testing.assert_allclose(prefix.bin_durations, np.array([0.1, 0.1, 0.2]))
    np.testing.assert_allclose(transition_durations_s(prefix), np.array([0.1, 0.15]))
    np.testing.assert_allclose(
        np.asarray(prefix.dt.transition_durations, dtype=float),
        np.array([0.1, 0.15]),
    )
    assert prefix.n_time == 3
    assert prefix.n_spikes == 0
