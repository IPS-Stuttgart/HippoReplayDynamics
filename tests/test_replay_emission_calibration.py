from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hipporeplayimm.data import ReplaySession
from hipporeplayimm.encoding import EncodingConfig, EncodingModel
from hipporeplayimm.replay_emission_calibration import apply_replay_cell_gains, fit_replay_cell_gains


def _empty_spike_session() -> ReplaySession:
    times = np.linspace(0.0, 2.0, 21)
    position = np.column_stack(
        [
            times,
            np.linspace(0.0, 20.0, times.shape[0]),
            np.linspace(0.0, 5.0, times.shape[0]),
        ]
    )
    return ReplaySession(
        rat="RatX",
        name="Open1",
        path=Path("RatX/Open1"),
        position=position,
        spikes=np.empty((0, 2), dtype=float),
        tetrode_cell_ids=np.empty((0, 2), dtype=float),
        excitatory_neurons=np.array([], dtype=int),
        inhibitory_neurons=np.array([], dtype=int),
        ripple_events=np.empty((0, 6), dtype=float),
        run_times=np.array([[0.0, 2.0]], dtype=float),
        sleep_box_immobile_times=np.empty((0, 2), dtype=float),
        sleep_times=np.empty((0, 2), dtype=float),
        rem_times=np.empty((0, 2), dtype=float),
        well_sequence=None,
        metadata={},
    )


def _ripple_session() -> ReplaySession:
    session = _empty_spike_session()
    session.ripple_events = np.array(
        [
            [0.20, 0.30, 0.25, 0.0, 0.0, 0.0],
            [0.50, 0.62, 0.56, 0.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    return session


def _two_cell_encoding() -> EncodingModel:
    cell_ids = np.array([1, 2], dtype=int)
    return EncodingModel(
        x_edges=np.array([0.0, 1.0], dtype=float),
        y_edges=np.array([0.0, 1.0], dtype=float),
        bin_centers=np.array([[0.5, 0.5]], dtype=float),
        rates_hz=np.ones((cell_ids.shape[0], 1), dtype=float),
        occupancy_s=np.array([1.0], dtype=float),
        cell_ids=cell_ids,
        config=EncodingConfig(),
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"prior_count": float("nan")}, "prior_count"),
        ({"prior_count": float("inf")}, "prior_count"),
        ({"prior_gain": float("nan")}, "prior_gain"),
        ({"min_gain": float("nan")}, "min_gain"),
        ({"max_gain": float("inf")}, "max_gain"),
    ],
)
def test_replay_cell_gain_calibration_rejects_nonfinite_scalars(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        fit_replay_cell_gains(_empty_spike_session(), _two_cell_encoding(), [], **kwargs)


@pytest.mark.parametrize("event_index", [True, False, np.bool_(True), np.bool_(False)])
def test_replay_cell_gain_calibration_rejects_boolean_event_indices(event_index) -> None:
    with pytest.raises(TypeError, match="event index must be an integer, not boolean"):
        fit_replay_cell_gains(_ripple_session(), _two_cell_encoding(), [event_index])


def test_replay_cell_gain_calibration_uses_prior_gain_without_valid_events() -> None:
    encoding = _two_cell_encoding()
    calibration = fit_replay_cell_gains(
        _empty_spike_session(),
        encoding,
        [],
        prior_count=0.0,
        prior_gain=1.0,
        min_gain=0.05,
        max_gain=20.0,
    )

    assert calibration.event_count == 0
    np.testing.assert_allclose(calibration.gains, np.ones(encoding.n_cells))
    np.testing.assert_allclose(calibration.observed_spikes, np.zeros(encoding.n_cells))
    np.testing.assert_allclose(calibration.expected_spikes, np.zeros(encoding.n_cells))


def test_apply_replay_cell_gains_aligns_manual_mapping_by_cell_id() -> None:
    encoding = _two_cell_encoding()
    calibrated = apply_replay_cell_gains(encoding, {2: 3.0})

    np.testing.assert_allclose(calibrated.rates_hz, np.array([[1.0], [3.0]]))
    np.testing.assert_allclose(encoding.rates_hz, np.ones((2, 1)))


@pytest.mark.parametrize(
    ("gains", "error_type", "message"),
    [
        (np.array([1.0, np.nan]), ValueError, "finite"),
        (np.array([1.0, np.inf]), ValueError, "finite"),
        (np.array([1.0, 0.0]), ValueError, "positive"),
        (np.array([1.0, -0.5]), ValueError, "positive"),
        ({1: np.nan}, ValueError, "finite"),
        ({2: 0.0}, ValueError, "positive"),
        (np.array([1.0, True], dtype=object), TypeError, "not boolean"),
        (np.array([1.0, np.bool_(True)], dtype=object), TypeError, "not boolean"),
        (np.array(["1.0", "2.0"]), TypeError, "not text"),
        (np.array([1.0, b"2.0"], dtype=object), TypeError, "not text"),
        ({1: True}, TypeError, "not boolean"),
        ({2: "3.0"}, TypeError, "not text"),
    ],
)
def test_apply_replay_cell_gains_rejects_invalid_manual_gains(gains, error_type, message) -> None:
    with pytest.raises(error_type, match=message):
        apply_replay_cell_gains(_two_cell_encoding(), gains)
