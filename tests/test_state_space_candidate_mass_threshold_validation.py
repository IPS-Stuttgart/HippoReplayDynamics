from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.state_space import StateSpaceDecoderConfig, StateSpaceReplayModel


def _uniform_emissions() -> LogEmissionTensor:
    return LogEmissionTensor(
        log_likelihood=np.zeros((2, 2), dtype=float),
        spike_counts=np.empty((2, 0), dtype=int),
        times=np.array([0.0, 1.0]),
        dt=1.0,
        cell_ids=np.empty(0, dtype=int),
        n_spikes=0,
    )


@pytest.mark.parametrize(
    "threshold",
    [
        True,
        np.bool_(False),
        np.array(True, dtype=object),
    ],
)
def test_candidate_construction_rejects_boolean_mass_thresholds(
    threshold: object,
) -> None:
    model = StateSpaceReplayModel(
        mode="momentum",
        config=StateSpaceDecoderConfig(
            mode="momentum",
            momentum_candidate_mass_threshold=threshold,
        ),
    )

    with pytest.raises(TypeError, match="mass_threshold.*boolean"):
        model.candidate_indices(_uniform_emissions())


@pytest.mark.parametrize(
    ("threshold", "error_type", "message"),
    [
        (True, TypeError, "momentum_candidate_mass_threshold.*boolean"),
        (1.5, ValueError, "momentum_candidate_mass_threshold.*exceed"),
    ],
)
def test_provided_candidate_support_still_validates_mass_threshold(
    threshold: object,
    error_type: type[Exception],
    message: str,
) -> None:
    model = StateSpaceReplayModel(
        mode="momentum",
        config=StateSpaceDecoderConfig(
            mode="momentum",
            momentum_candidate_mass_threshold=threshold,
        ),
    )
    candidates = [
        np.array([0, 1], dtype=int),
        np.array([0, 1], dtype=int),
    ]

    with pytest.raises(error_type, match=message):
        model.score(
            _uniform_emissions(),
            np.array([[0.0], [1.0]], dtype=float),
            candidate_indices=candidates,
        )


def test_unit_mass_threshold_remains_valid() -> None:
    model = StateSpaceReplayModel(
        mode="momentum",
        config=StateSpaceDecoderConfig(
            mode="momentum",
            momentum_candidate_mass_threshold=1.0,
        ),
    )

    candidates = model.candidate_indices(_uniform_emissions())

    assert [candidate.tolist() for candidate in candidates] == [[0, 1], [0, 1]]
