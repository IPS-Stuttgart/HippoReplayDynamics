from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.extract_replay_commitment_composition_posteriors import (
    build_gate_summary,
    continuous_bout_ids,
    normalized_posterior,
    posterior_path_summary,
)


def test_normalized_posterior_and_path_summary() -> None:
    log_values = np.log(np.array([[0.8, 0.2], [0.25, 0.75]], dtype=float))
    centers = np.array([[0.0, 0.0], [10.0, 0.0]])

    probability = normalized_posterior(log_values)
    summary = posterior_path_summary(log_values, centers)

    assert np.allclose(probability.sum(axis=1), 1.0)
    assert np.allclose(summary["mean_xy"][:, 0], [2.0, 7.5])
    assert np.allclose(summary["map_xy"][:, 0], [0.0, 10.0])
    assert np.all(summary["entropy"] > 0.0)


def test_continuous_bouts_exclude_stationary_and_fragmented_phases() -> None:
    mode = np.array([0, 1, 1, 2, 1, 0, 2, 1, 1], dtype=int)

    bouts = continuous_bout_ids(mode)

    assert bouts.tolist() == [-1, 0, 0, -1, 1, -1, -1, 2, 2]


def test_posterior_gate_fails_nonvacuously_when_no_events_are_scored() -> None:
    frozen = pd.DataFrame([{"session": "Rat1/Open1", "event_index": 1}])

    gates = build_gate_summary(
        frozen,
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        rescore_tolerance=0.025,
        include_momentum_posterior=True,
    ).set_index("gate")

    assert not bool(gates.loc["all_events_scored", "passed"])
    assert not bool(gates.loc["posterior_bins_present", "passed"])
    assert not bool(gates.loc["overall", "passed"])


def test_posterior_gate_requires_rescore_agreement_and_all_modes() -> None:
    frozen = pd.DataFrame([{"session": "Rat1/Open1", "event_index": 1}])
    bins = pd.DataFrame(
        [
            {
                "imm_posterior_mean_x_cm": 0.0,
                "emission_only_mean_x_cm": 0.0,
                "momentum_posterior_mean_x_cm": 0.0,
                "map_mode_index": mode,
            }
            for mode in (0, 1, 2)
        ]
    )
    transitions = pd.DataFrame([{"switch_probability": 0.2}])
    events = pd.DataFrame(
        [
            {
                "imm_rescore_error": 0.001,
                "momentum_rescore_error": 0.002,
            }
        ]
    )

    gates = build_gate_summary(
        frozen,
        bins,
        transitions,
        events,
        rescore_tolerance=0.025,
        include_momentum_posterior=True,
    ).set_index("gate")

    assert bool(gates.loc["overall", "passed"])
