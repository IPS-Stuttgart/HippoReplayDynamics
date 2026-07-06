from __future__ import annotations

import numpy as np
import pandas as pd

from hipporeplayimm.accuracy_upgrades import (
    ValidStateGridReplayModel,
    bootstrap_model_win_probabilities,
    model_probability_diagnostics,
    reverse_emissions,
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


def test_model_probability_diagnostics_excludes_string_false_comparable_rows() -> None:
    scores = pd.DataFrame(
        {
            "session": ["s1", "s1"],
            "event_index": [0, 0],
            "model": ["exact", "lower-bound"],
            "log_evidence": [0.0, 100.0],
            "status": ["success", "success"],
            "evidence_comparable": ["True", "False"],
        }
    )

    out = model_probability_diagnostics(scores)

    assert out.loc[0, "models"] == 1
    assert out.loc[0, "best_model"] == "exact"


def test_model_probability_diagnostics_accepts_legacy_success_status_values() -> None:
    scores = pd.DataFrame(
        {
            "session": ["s1", "s1", "s2", "s2"],
            "event_index": [0, 0, 0, 0],
            "model": ["legacy-success", "missing-empty", "missing-nan", "failed"],
            "log_evidence": [2.0, 1.0, 3.0, 100.0],
            "status": ["Success", "", np.nan, "failed"],
            "evidence_comparable": [True, True, True, True],
        }
    )

    out = model_probability_diagnostics(scores).set_index("session")

    assert out.loc["s1", "models"] == 2
    assert out.loc["s1", "best_model"] == "legacy-success"
    assert out.loc["s2", "models"] == 1
    assert out.loc["s2", "best_model"] == "missing-nan"


def test_accuracy_reverse_emissions_keeps_time_coordinates_increasing() -> None:
    emissions = LogEmissionTensor(
        log_likelihood=np.asarray(
            [
                [0.0, -1.0],
                [-2.0, -0.5],
                [-0.25, -3.0],
            ],
            dtype=float,
        ),
        spike_counts=np.asarray([[0], [2], [1]], dtype=int),
        times=np.asarray([0.005, 0.020, 0.055], dtype=float),
        dt=0.02,
        cell_ids=np.asarray([7], dtype=int),
        n_spikes=3,
        bin_durations=np.asarray([0.010, 0.020, 0.030], dtype=float),
        transition_durations=np.asarray([0.015, 0.035], dtype=float),
        metadata={"source": "unit-test"},
    )

    reversed_emissions = reverse_emissions(emissions)

    np.testing.assert_allclose(reversed_emissions.log_likelihood, emissions.log_likelihood[::-1])
    np.testing.assert_allclose(reversed_emissions.bin_durations, emissions.bin_durations[::-1])
    np.testing.assert_allclose(reversed_emissions.transition_durations, np.asarray([0.035, 0.015]))
    np.testing.assert_allclose(reversed_emissions.times, np.asarray([0.005, 0.040, 0.055]))
    assert np.all(np.diff(reversed_emissions.times) > 0.0)
    assert reversed_emissions.metadata == emissions.metadata
    assert reversed_emissions.metadata is not emissions.metadata
