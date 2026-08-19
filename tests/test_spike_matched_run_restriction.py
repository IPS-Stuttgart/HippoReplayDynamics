from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

from hipporeplayimm.data import ReplaySession

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from spike_matched_event_window_null import spike_matched_null_windows  # noqa: E402


def _session_without_run_times(tmp_path: Path) -> ReplaySession:
    position_times = np.arange(0.0, 2.01, 0.05)
    position = np.column_stack(
        [
            position_times,
            10.0 * position_times,
            np.zeros_like(position_times),
        ]
    )
    return ReplaySession(
        rat="Rat1",
        name="Open1",
        path=tmp_path,
        position=position,
        spikes=np.array(
            [
                [0.20, 1],
                [0.25, 2],
                [1.02, 1],
                [1.06, 2],
                [1.50, 1],
            ],
            dtype=float,
        ),
        tetrode_cell_ids=np.empty((0, 2), dtype=int),
        excitatory_neurons=np.array([1, 2], dtype=int),
        inhibitory_neurons=np.array([], dtype=int),
        ripple_events=np.array([[1.0, 1.1, 1.05, 1.0, 1.0, 1.0]], dtype=float),
        run_times=np.empty((0, 2), dtype=float),
        sleep_box_immobile_times=np.empty((0, 2), dtype=float),
        sleep_times=np.array([[0.0, 2.0]], dtype=float),
        rem_times=np.empty((0, 2), dtype=float),
        well_sequence=None,
        metadata={},
    )


def test_run_restricted_nulls_do_not_fall_back_to_whole_recording(tmp_path: Path) -> None:
    session = _session_without_run_times(tmp_path)

    restricted = spike_matched_null_windows(
        session,
        0,
        nulls_per_event=1,
        random_seed=1,
        candidate_step_s=0.1,
        restrict_to_run_times=True,
    )
    unrestricted = spike_matched_null_windows(
        session,
        0,
        nulls_per_event=1,
        random_seed=1,
        candidate_step_s=0.1,
        restrict_to_run_times=False,
    )

    assert restricted.empty
    assert len(unrestricted) == 1
    assert not bool(unrestricted.iloc[0]["restrict_to_run_times"])
