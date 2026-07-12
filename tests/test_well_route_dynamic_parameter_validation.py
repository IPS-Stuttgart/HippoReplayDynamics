from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.well_route_state_space import WellRouteStateSpaceReplayModel


def _emissions() -> LogEmissionTensor:
    return LogEmissionTensor(
        log_likelihood=np.zeros((2, 2), dtype=float),
        spike_counts=np.empty((2, 0), dtype=int),
        times=np.array([0.0, 0.02], dtype=float),
        dt=0.02,
        cell_ids=np.empty(0, dtype=int),
        n_spikes=0,
    )


def _score(model: WellRouteStateSpaceReplayModel):
    return model.score(
        _emissions(),
        np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float),
    )


@pytest.mark.parametrize(
    ("parameter", "bad_value"),
    [
        ("transition_sigma_cm_sqrt_s", True),
        ("transition_sigma_cm_sqrt_s", "85.0"),
        ("drift_speed_cm_s", False),
        ("drift_speed_cm_s", b"400.0"),
        ("max_step_sigma", np.bool_(True)),
        ("max_step_sigma", np.array("4.0")),
        ("max_step_sigma", np.array([4.0])),
        ("max_step_sigma", 4.0 + 0.0j),
    ],
)
def test_route_state_space_rejects_non_real_scalar_dynamic_parameters(
    parameter: str,
    bad_value: object,
) -> None:
    kwargs = {
        "candidate_routes": np.array([[[0.0, 0.0], [1.0, 0.0]]], dtype=float),
        parameter: bad_value,
    }
    model = WellRouteStateSpaceReplayModel(**kwargs)

    with pytest.raises(ValueError, match=parameter):
        _score(model)


def test_route_state_space_keeps_zero_drift_valid() -> None:
    model = WellRouteStateSpaceReplayModel(
        candidate_routes=np.array([[[0.0, 0.0], [1.0, 0.0]]], dtype=float),
        drift_speed_cm_s=0.0,
    )

    score = _score(model)

    assert np.isfinite(score.log_likelihood)
    assert score.diagnostics["route_state_space_drift_speed_cm_s"] == 0.0
