from pathlib import Path

import numpy as np

import hipporeplayimm.clusterless as clusterless
import hipporeplayimm.clusterless_cv_exclusion as exclusion
import hipporeplayimm.clusterless_position_validation as validation
from hipporeplayimm.clusterless import ClusterlessMarkConfig
from hipporeplayimm.data import ReplaySession
from hipporeplayimm.encoding import EncodingConfig


def _session(position: np.ndarray, run_times: np.ndarray) -> ReplaySession:
    return ReplaySession(
        rat="RatX",
        name="OpenX",
        path=Path("."),
        position=position,
        spikes=np.empty((0, 2), dtype=float),
        tetrode_cell_ids=np.empty((0, 2), dtype=float),
        excitatory_neurons=np.empty(0, dtype=int),
        inhibitory_neurons=np.empty(0, dtype=int),
        ripple_events=np.empty((0, 6), dtype=float),
        run_times=run_times,
        sleep_box_immobile_times=np.empty((0, 2), dtype=float),
        sleep_times=np.empty((0, 2), dtype=float),
        rem_times=np.empty((0, 2), dtype=float),
        well_sequence=None,
        metadata={},
    )


def test_clusterless_position_validation_fits_each_fold_with_exact_exclusions(monkeypatch):
    times = np.linspace(0.0, 4.0, 41)
    position = np.column_stack([times, 10.0 * times, np.zeros_like(times), np.zeros_like(times)])
    session = _session(position, np.array([[0.0, 4.0]], dtype=float))
    fitted: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def fake_fit(training_session, config, excluded_intervals):
        fold_index = len(fitted)
        fitted[fold_index] = (
            training_session.run_times.copy(),
            np.asarray(excluded_intervals, dtype=float).copy(),
        )
        return object()

    def fake_decode(session, encoding, window, window_index, *, fold_index):
        return {
            "session": session.session_id,
            "fold": fold_index,
            "window_index": window_index,
            "start_time": window["start_time"],
            "end_time": window["end_time"],
        }

    monkeypatch.setattr(
        validation,
        "fit_clusterless_mark_encoding_excluding_intervals",
        fake_fit,
    )
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
    assert set(fitted) == {0, 1}
    for fold_index, (training_run_times, excluded_intervals) in fitted.items():
        np.testing.assert_array_equal(training_run_times, session.run_times)
        held_out = samples.loc[
            samples["fold"] == fold_index,
            ["start_time", "end_time"],
        ].to_numpy(dtype=float)
        np.testing.assert_array_equal(
            excluded_intervals[np.argsort(excluded_intervals[:, 0])],
            held_out[np.argsort(held_out[:, 0])],
        )


def test_clusterless_cv_excludes_partial_frame_exposure_and_half_open_marks(monkeypatch):
    times = np.array([0.0, 1.0, 2.0], dtype=float)
    position = np.column_stack([times, times, np.zeros_like(times), np.zeros_like(times)])
    session = _session(position, np.array([[0.0, 2.0]], dtype=float))

    monkeypatch.setattr(
        clusterless,
        "_frame_durations",
        lambda values: np.ones(np.asarray(values).shape, dtype=float),
    )
    monkeypatch.setattr(
        clusterless,
        "_times_in_intervals",
        lambda values, intervals: np.ones(np.asarray(values).shape, dtype=bool),
    )

    def fake_fit(training_session, config):
        return {
            "durations": clusterless._frame_durations(times),
            "position_membership": clusterless._times_in_intervals(
                times,
                training_session.run_times,
            ),
            "mark_membership": clusterless._times_in_intervals(
                np.array([0.25, 0.75, 1.5], dtype=float),
                training_session.run_times,
            ),
        }

    monkeypatch.setattr(clusterless, "fit_clusterless_mark_encoding", fake_fit)

    result = exclusion.fit_clusterless_mark_encoding_excluding_intervals(
        session,
        ClusterlessMarkConfig(),
        np.array([[0.5, 1.5]], dtype=float),
    )

    np.testing.assert_allclose(result["durations"], np.array([0.5, 0.5, 1.0]))
    np.testing.assert_array_equal(
        result["position_membership"],
        np.array([True, True, True]),
    )
    np.testing.assert_array_equal(
        result["mark_membership"],
        np.array([True, False, True]),
    )


def test_clusterless_training_interval_subtraction_preserves_half_open_endpoints():
    result = validation._subtract_half_open_intervals(
        np.array([[0.0, 4.0]], dtype=float),
        np.array([[1.0, 2.0], [2.0, 3.0]], dtype=float),
    )

    assert result.shape == (2, 2)
    assert result[0, 0] == 0.0
    assert result[0, 1] < 1.0
    assert result[1].tolist() == [3.0, 4.0]
