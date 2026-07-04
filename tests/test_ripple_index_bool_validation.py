from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hipporeplayimm.data import ReplaySession, _coerce_ripple_event
from hipporeplayimm.encoding import EmissionConfig, EncodingConfig, EncodingModel, build_emissions


def _two_ripple_session() -> ReplaySession:
    return ReplaySession(
        rat="RatX",
        name="Open1",
        path=Path("RatX/Open1"),
        position=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
            dtype=float,
        ),
        spikes=np.empty((0, 2), dtype=float),
        tetrode_cell_ids=np.empty((0, 2), dtype=float),
        excitatory_neurons=np.array([], dtype=int),
        inhibitory_neurons=np.array([], dtype=int),
        ripple_events=np.array(
            [
                [0.10, 0.20, 0.15, 1.0, 0.0, 0.0],
                [0.30, 0.40, 0.35, 1.0, 0.0, 0.0],
            ],
            dtype=float,
        ),
        run_times=np.array([[0.0, 1.0]], dtype=float),
        sleep_box_immobile_times=np.empty((0, 2), dtype=float),
        sleep_times=np.empty((0, 2), dtype=float),
        rem_times=np.empty((0, 2), dtype=float),
        well_sequence=None,
        metadata={},
    )


def _single_bin_encoding() -> EncodingModel:
    return EncodingModel(
        x_edges=np.array([0.0, 1.0], dtype=float),
        y_edges=np.array([0.0, 1.0], dtype=float),
        bin_centers=np.array([[0.5, 0.5]], dtype=float),
        rates_hz=np.zeros((0, 1), dtype=float),
        occupancy_s=np.array([1.0], dtype=float),
        cell_ids=np.array([], dtype=int),
        config=EncodingConfig(),
    )


@pytest.mark.parametrize(
    "ripple_index",
    [
        True,
        np.bool_(False),
        np.array(True),
        np.asarray(False, dtype=object),
    ],
)
def test_boolean_ripple_indices_are_rejected_before_aliasing_to_events(ripple_index) -> None:
    session = _two_ripple_session()

    with pytest.raises(TypeError, match="ripple index"):
        _coerce_ripple_event(session, ripple_index)


@pytest.mark.parametrize(
    "ripple_index",
    [
        np.array(1),
        np.array(1.0),
        np.asarray(1, dtype=object),
        np.asarray(1.0, dtype=object),
        np.float64(1.0),
    ],
)
def test_integral_scalar_array_ripple_indices_select_events(ripple_index) -> None:
    session = _two_ripple_session()

    event = _coerce_ripple_event(session, ripple_index)

    assert event.start == pytest.approx(0.30)
    assert event.end == pytest.approx(0.40)


@pytest.mark.parametrize(
    "ripple_index",
    [
        np.array(0.5),
        np.array(np.nan),
        np.array(np.inf),
        np.asarray("1", dtype=object),
        np.array([1]),
    ],
)
def test_nonintegral_scalar_array_ripple_indices_are_rejected(ripple_index) -> None:
    session = _two_ripple_session()

    with pytest.raises(TypeError, match="ripple index"):
        _coerce_ripple_event(session, ripple_index)


@pytest.mark.parametrize("ripple_index", [True, np.array(True)])
def test_emission_builder_rejects_boolean_ripple_indices_through_imported_alias(ripple_index) -> None:
    session = _two_ripple_session()
    encoding = _single_bin_encoding()

    with pytest.raises(TypeError, match="ripple index"):
        build_emissions(
            session,
            encoding,
            ripple_index,
            EmissionConfig(time_bin_s=0.05),
        )


def test_emission_builder_accepts_integral_scalar_array_ripple_indices_through_imported_alias() -> None:
    session = _two_ripple_session()
    encoding = _single_bin_encoding()

    emissions = build_emissions(
        session,
        encoding,
        np.asarray(1.0, dtype=object),
        EmissionConfig(time_bin_s=0.05),
    )

    assert emissions.n_time == 2
    np.testing.assert_allclose(emissions.times, np.array([0.325, 0.375]))
