from __future__ import annotations

import numpy as np
import pandas as pd

from hipporeplayimm.accuracy_upgrades import (
    ValidStateGridReplayModel,
    bootstrap_model_win_probabilities,
)
from hipporeplayimm.encoding import LogEmissionTensor


def test_valid_state_grid_model_expands_posterior_to_full_grid() -> None:
    emissions = LogEmissionTensor(
        log_likelihood=np.asarray(
            [
                [0.0, -1.0, -10.0, -2.0],
                [-1.0, 0.0, -10.0, -2.0],
            ],
            dtype=float,
        ),
        spike_counts=np.zeros((2, 1), dtype=int),
        times=np.asarray([0.0, 0.02]),
        dt=0.02,
        cell_ids=np.asarray([1]),
        n_spikes=0,
    )
    bin_centers = np.asarray(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ],
        dtype=float,
    )
    valid_mask = np.asarray([True, True, False, True])

    score = ValidStateGridReplayModel(valid_mask, grid_shape=(2, 2)).score(emissions, bin_centers)

    assert score.trajectory_log_posterior is not None
    assert score.trajectory_log_posterior.shape == (2, 4)
    assert np.exp(score.trajectory_log_posterior[:, ~valid_mask]).max() == 0.0


def test_bootstrap_model_win_probabilities_accepts_window_groups() -> None:
    scores = pd.DataFrame(
        {
            "session": ["s1", "s1", "s1", "s1"],
            "event_index": [0, 0, 0, 0],
            "window_index": [0, 0, 1, 1],
            "model": ["a", "b", "a", "b"],
            "log_evidence": [2.0, 1.0, 0.0, 3.0],
        }
    )

    out = bootstrap_model_win_probabilities(
        scores,
        n_bootstrap=25,
        random_seed=0,
        group_columns=("session", "event_index", "window_index"),
    )

    assert set(out["model"]) == {"a", "b"}
    assert np.isclose(out["bootstrap_win_probability"].sum(), 1.0)
