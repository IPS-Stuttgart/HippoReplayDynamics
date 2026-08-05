from pathlib import Path

import numpy as np

import hipporeplayimm.clusterless_position_validation as validation
from hipporeplayimm.clusterless import ClusterlessMarkConfig
from hipporeplayimm.data import ReplaySession
from hipporeplayimm.encoding import EncodingConfig


def test_clusterless_position_validation_fits_each_fold_without_held_out_windows(monkeypatch):
    times = np.linspace(0.0, 4.0, 41)
    position = np.column_stack([times, 10.0 * times, np.zeros_like(times), np.zeros_like(times)])
    session = ReplaySession(
        rat="RatX",
        name="OpenX",
        path=Path("."),
        position=position,
        spikes=np.empty((0, 2), dtype=float),
        tetrode_cell_ids=np.empty((0, 2), dtype=float),
        excitatory_neurons=np.empty(0, dtype=int),
        inhibitory_neurons=np.empty(0, dtype=int),
        ripple_events=np.empty((0, 6), dtype=float),
        run_times=np.array([[0.0, 4.0]], dtype=float),
        sleep_box_immobile_times=np.empty((0, 2), dtype=float),
        sleep_times=np.empty((0, 2), dtype=float),
        rem_times=np.empty((0, 2), dtype=float),
        well_sequence=None,
        metadata={},
    )
    fitted_run_times: dict[int, np.ndarray] = {}

    def fake_fit(training_session, config):
        fold_index = len(fitted_run_times)
        fitted_run_times[fold_index] = training_session.run_times.copy()
        return object()

    def fake_decode(session, encoding, window, window_index, *, fold_index):
        return {
            "session": session.session_id,
            "fold": fold_index,
            "window_index": window_index,
            "start_time": window["start_time"],
            "end_time": window["end_time"],
        }

    monkeypatch.setattr(validation, "fit_clusterless_mark_encoding", fake_fit)
    monkeypatch.setattr(validation, "_decode_clusterless_window", fake_decode)

    samples = validation.validate_session_clusterless_position(
        session,
        validation.ClusterlessPositionValidationConfig(
            clusterless=ClusterlessMarkConfig(
                encoding=EncodingConfig(min_speed_cm_s=0.0),
            ),
            decode_bin_s=1.0,
            n_folds=2,
            random_seed=7,
        ),
    )

    assert sorted(samples["window_index"].tolist()) == [0, 1, 2, 3]
    assert set(samples["fold"]) == {0, 1}
    assert set(fitted_run_times) == {0, 1}
    for fold_index, training_intervals in fitted_run_times.items():
        held_out = samples.loc[samples["fold"] == fold_index, ["start_time", "end_time"]].to_numpy(dtype=float)
        for training_start, training_end in training_intervals:
            for held_out_start, held_out_end in held_out:
                assert training_end < held_out_start or training_start >= held_out_end


def test_clusterless_training_interval_subtraction_preserves_half_open_endpoints():
    result = validation._subtract_half_open_intervals(
        np.array([[0.0, 4.0]], dtype=float),
        np.array([[1.0, 2.0], [2.0, 3.0]], dtype=float),
    )

    assert result.shape == (2, 2)
    assert result[0, 0] == 0.0
    assert result[0, 1] < 1.0
    assert result[1].tolist() == [3.0, 4.0]
