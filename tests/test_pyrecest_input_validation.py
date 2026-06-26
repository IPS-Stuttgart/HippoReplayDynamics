from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.pyrecest_models import PyRecEstGoalParticleModel, _coerce_candidate_goals


def _single_bin_emissions(n_bins: int = 2) -> LogEmissionTensor:
    return LogEmissionTensor(
        log_likelihood=np.zeros((1, n_bins), dtype=float),
        spike_counts=np.zeros((1, 1), dtype=int),
        times=np.array([0.0]),
        dt=0.02,
        cell_ids=np.array([1], dtype=int),
        n_spikes=0,
    )


def test_pyrecest_score_rejects_bin_center_count_mismatch_before_optional_import() -> None:
    emissions = _single_bin_emissions(n_bins=2)

    with pytest.raises(ValueError, match="emissions.n_bins must match bin_centers rows"):
        PyRecEstGoalParticleModel().score(
            emissions,
            np.zeros((1, 2), dtype=float),
        )


def test_pyrecest_score_rejects_nonfinite_bin_centers_before_optional_import() -> None:
    emissions = _single_bin_emissions(n_bins=2)
    bin_centers = np.array([[0.0, 0.0], [np.nan, 1.0]], dtype=float)

    with pytest.raises(ValueError, match="bin_centers must be finite"):
        PyRecEstGoalParticleModel().score(emissions, bin_centers)


def test_pyrecest_candidate_goals_reject_nonfinite_values() -> None:
    bin_centers = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)

    with pytest.raises(ValueError, match=r"candidate_goals must .* finite"):
        _coerce_candidate_goals(
            np.array([[0.0, np.nan]], dtype=float),
            bin_centers,
        )
