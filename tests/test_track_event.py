import numpy as np
import pytest

from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.models import EventScore
from scripts.track_event import _trajectory_rows_from_log_posteriors


def test_trajectory_rows_entropy_handles_zero_probability_bins():
    emissions = LogEmissionTensor(
        log_likelihood=np.array([[0.0, -np.inf], [0.0, 0.0]]),
        spike_counts=np.zeros((2, 1), dtype=int),
        times=np.array([0.0, 0.02]),
        dt=0.02,
        cell_ids=np.array([1]),
        n_spikes=0,
    )
    log_posteriors = np.array(
        [
            [0.0, -np.inf],
            [np.log(0.25), np.log(0.75)],
        ]
    )
    score = EventScore("synthetic", 0.0, emissions.n_time, emissions.n_spikes)

    rows = _trajectory_rows_from_log_posteriors(
        log_posteriors=log_posteriors,
        emissions=emissions,
        bin_centers=np.array([[0.0, 0.0], [1.0, 0.0]]),
        score=score,
        likelihood_column="event_log_likelihood",
    )

    assert np.isfinite(rows.loc[0, "posterior_entropy"])
    assert rows.loc[0, "posterior_entropy"] == pytest.approx(0.0)
    expected_entropy = -(0.25 * np.log(0.25) + 0.75 * np.log(0.75))
    assert rows.loc[1, "posterior_entropy"] == pytest.approx(expected_entropy)
