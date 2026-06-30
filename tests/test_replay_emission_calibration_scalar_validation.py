from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.replay_emission_calibration import _event_index_int, fit_replay_cell_gains


@pytest.mark.parametrize(
    ("parameter", "kwargs"),
    [
        ("prior_count", {"prior_count": np.array([5.0])}),
        ("prior_gain", {"prior_gain": np.array([1.0])}),
        ("min_gain", {"min_gain": np.array([0.05])}),
        ("max_gain", {"max_gain": np.array([20.0])}),
    ],
)
def test_fit_replay_cell_gains_rejects_array_shaped_scalar_parameters(parameter: str, kwargs: dict[str, object]) -> None:
    with pytest.raises(TypeError, match=parameter):
        fit_replay_cell_gains(object(), object(), [], **kwargs)


@pytest.mark.parametrize(
    ("parameter", "kwargs"),
    [
        ("prior_count", {"prior_count": True}),
        ("prior_gain", {"prior_gain": np.bool_(True)}),
        ("min_gain", {"min_gain": np.asarray(True, dtype=object)}),
        ("max_gain", {"max_gain": False}),
    ],
)
def test_fit_replay_cell_gains_rejects_boolean_scalar_parameters(parameter: str, kwargs: dict[str, object]) -> None:
    with pytest.raises(TypeError, match=parameter):
        fit_replay_cell_gains(object(), object(), [], **kwargs)


@pytest.mark.parametrize("event_index", [0, np.int64(1)])
def test_event_index_int_accepts_integer_scalars(event_index: object) -> None:
    assert _event_index_int(event_index) == int(event_index)


@pytest.mark.parametrize(
    "event_index",
    [
        True,
        np.bool_(False),
        1.2,
        np.array([0]),
        np.asarray(0.0),
    ],
)
def test_event_index_int_rejects_boolean_float_and_array_inputs(event_index: object) -> None:
    with pytest.raises(TypeError, match="event index"):
        _event_index_int(event_index)
