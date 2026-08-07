from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hipporeplayimm import apply_runtime_patches
from hipporeplayimm.data import ReplaySession, SpikeMarkData
import hipporeplayimm.result_improvements as result_improvements


def test_circular_shift_uses_session_time_support_and_keeps_marks_aligned() -> None:
    session = _session(
        spike_times=np.array([2.0, 4.0, 8.0]),
        position_times=np.array([0.0, 10.0]),
        with_marks=True,
    )

    shifted = result_improvements.circular_shift_spikes_session(session, shift_s=3.0)

    np.testing.assert_allclose(shifted.spikes[:, 0], np.array([1.0, 5.0, 7.0]))
    np.testing.assert_array_equal(shifted.spikes[:, 1].astype(int), np.array([30, 10, 20]))
    assert shifted.spike_marks is not None
    np.testing.assert_allclose(shifted.spike_marks.times, shifted.spikes[:, 0])
    np.testing.assert_array_equal(shifted.spike_marks.cell_ids, shifted.spikes[:, 1].astype(int))
    np.testing.assert_allclose(shifted.spike_marks.marks[:, 0], shifted.spikes[:, 1])


def test_circular_shift_does_not_alias_first_and_last_spike_without_session_support() -> None:
    session = _session(
        spike_times=np.array([0.0, 1.0, 4.0]),
        position_times=np.array([], dtype=float),
        with_marks=False,
    )

    shifted = result_improvements.circular_shift_spikes_session(session, shift_s=1.5)

    assert np.unique(shifted.spikes[:, 0]).size == 3


@pytest.mark.parametrize("shift_s", [True, np.inf, -np.inf, np.nan, 1.0 + 2.0j])
def test_circular_shift_rejects_invalid_explicit_shifts(shift_s: object) -> None:
    session = _session(
        spike_times=np.array([0.0, 1.0, 4.0]),
        position_times=np.array([0.0, 5.0]),
        with_marks=False,
    )

    with pytest.raises(ValueError, match="shift_s must be a finite real scalar"):
        result_improvements.circular_shift_spikes_session(session, shift_s=shift_s)


def test_runtime_patch_refreshes_replaced_circular_shift_helper(monkeypatch) -> None:
    def stale_circular_shift(session, shift_s=None, random_seed: int = 1):
        return session

    monkeypatch.setattr(
        result_improvements,
        "_circular_shift_spike_time_patch_applied",
        True,
        raising=False,
    )
    monkeypatch.setattr(
        result_improvements,
        "circular_shift_spikes_session",
        stale_circular_shift,
    )

    apply_runtime_patches()

    assert result_improvements.circular_shift_spikes_session is not stale_circular_shift


def _session(
    *,
    spike_times: np.ndarray,
    position_times: np.ndarray,
    with_marks: bool,
) -> ReplaySession:
    cell_ids = 10 * np.arange(1, spike_times.size + 1, dtype=int)
    spikes = np.column_stack((spike_times, cell_ids.astype(float)))
    if position_times.size:
        position = np.column_stack(
            (
                position_times,
                np.zeros(position_times.size, dtype=float),
                np.zeros(position_times.size, dtype=float),
            )
        )
    else:
        position = np.empty((0, 3), dtype=float)

    spike_marks = None
    if with_marks:
        spike_marks = SpikeMarkData(
            times=spike_times.copy(),
            marks=np.column_stack((cell_ids.astype(float), np.arange(spike_times.size, dtype=float))),
            source_file="synthetic",
            source_variable="marks",
            feature_names=("cell_id_proxy", "feature"),
            cell_ids=cell_ids.copy(),
            group_ids=np.arange(1, spike_times.size + 1, dtype=int),
        )

    return ReplaySession(
        rat="RatX",
        name="OpenX",
        path=Path("."),
        position=position,
        spikes=spikes,
        tetrode_cell_ids=np.empty((0, 2), dtype=int),
        excitatory_neurons=np.array([], dtype=int),
        inhibitory_neurons=np.array([], dtype=int),
        ripple_events=np.empty((0, 6), dtype=float),
        run_times=np.empty((0, 2), dtype=float),
        sleep_box_immobile_times=np.empty((0, 2), dtype=float),
        sleep_times=np.empty((0, 2), dtype=float),
        rem_times=np.empty((0, 2), dtype=float),
        well_sequence=None,
        metadata={},
        spike_marks=spike_marks,
    )
