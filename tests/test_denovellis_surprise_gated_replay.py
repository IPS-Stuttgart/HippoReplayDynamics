from types import SimpleNamespace

import numpy as np
import pandas as pd

from scripts.analyze_denovellis_surprise_gated_replay import (
    _candidate_pump_times,
    assign_causal_binary_surprise,
    build_gate_summary,
    extract_outbound_trials_from_epoch,
    link_replay_events,
    partial_rank_effect,
)


def _synthetic_linpos_epoch():
    pairs = [(1, 2), (2, 1), (1, 3), (3, 1), (1, 3)]
    samples_per_segment = 5
    wells = np.repeat(np.asarray(pairs), samples_per_segment, axis=0)
    time = np.arange(len(wells), dtype=float)
    distance = np.full((len(wells), 3), 100.0)
    for segment, (_, destination) in enumerate(pairs):
        start = segment * samples_per_segment
        distance[start : start + samples_per_segment, destination - 1] = [30.0, 20.0, 12.0, 5.0, 0.0]
    return SimpleNamespace(
        trajwells=np.asarray([[1, 2], [1, 3]]),
        statematrix=SimpleNamespace(
            time=time,
            wellExitEnter=wells,
            linearDistanceToWells=distance,
        ),
    )


def test_extract_outbound_trials_uses_final_near_well_dwell_and_alternation():
    trials = extract_outbound_trials_from_epoch(
        _synthetic_linpos_epoch(),
        animal="bon",
        day=3,
        epoch=2,
        well_radius_cm=10.0,
        max_window_s=10.0,
    )

    assert trials["destination_well"].tolist() == [2, 3, 3]
    assert pd.isna(trials.iloc[0]["alternation_consistent"])
    assert bool(trials.iloc[1]["alternation_consistent"])
    assert not bool(trials.iloc[2]["alternation_consistent"])
    assert trials["arrival_time_s"].tolist() == [3.0, 13.0, 23.0]
    assert trials["post_arrival_dwell_s"].tolist() == [1.0, 1.0, 1.0]
    assert trials["choice_analysis_exposure_s"].tolist() == [1.0, 1.0, 1.0]


def test_causal_surprise_uses_only_prior_outcomes():
    frame = pd.DataFrame(
        {
            "session": ["a"] * 3,
            "trial_index": [1, 2, 3],
            "outcome": [True, True, False],
        }
    )
    result = assign_causal_binary_surprise(frame, outcome_col="outcome", decay=1.0, prefix="test")

    np.testing.assert_allclose(
        result["test_surprise_nats_decay_1p0"],
        [-np.log(0.5), -np.log(2.0 / 3.0), -np.log(1.0 / 4.0)],
    )
    modified = frame.copy()
    modified.loc[2, "outcome"] = True
    changed = assign_causal_binary_surprise(modified, outcome_col="outcome", decay=1.0, prefix="test")
    np.testing.assert_allclose(
        result.loc[:1, "test_surprise_nats_decay_1p0"],
        changed.loc[:1, "test_surprise_nats_decay_1p0"],
    )


def test_link_replay_events_respects_exposure_window():
    trials = pd.DataFrame(
        {
            "animal": ["bon"],
            "day": [3],
            "epoch": [2],
            "trial_id": ["bon-03-02-outbound-1"],
            "choice_analysis_start_s": [10.0],
            "choice_analysis_exposure_s": [2.0],
        }
    )
    events = pd.DataFrame(
        {
            "animal": ["bon", "bon", "bon"],
            "day": [3, 3, 3],
            "epoch": [2, 2, 2],
            "ripple_number": [1, 2, 3],
            "start_time_s": [9.99, 10.5, 12.01],
            "trajectory_component_present": [True, True, False],
        }
    )

    linked_trials, linked_events = link_replay_events(trials, events)

    assert linked_events["ripple_number"].tolist() == [2]
    assert linked_trials.iloc[0]["choice_n_events"] == 1
    assert linked_trials.iloc[0]["choice_n_trajectory_events"] == 1
    assert linked_trials.iloc[0]["choice_all_event_rate_hz"] == 0.5


def test_partial_rank_effect_recovers_positive_within_session_association():
    rng = np.random.default_rng(8)
    rows = []
    for animal in ["a", "b", "c", "d"]:
        for session_index in range(2):
            for trial in range(20):
                surprise = rng.normal()
                quality = rng.normal()
                rows.append(
                    {
                        "animal": animal,
                        "session": f"{animal}-{session_index}",
                        "correct": bool(trial % 2),
                        "surprise": surprise,
                        "quality": quality,
                        "endpoint": 1.4 * surprise + 0.4 * quality + rng.normal(scale=0.15),
                    }
                )
    frame = pd.DataFrame(rows)

    effect, n = partial_rank_effect(
        frame,
        x_col="surprise",
        y_col="endpoint",
        outcome_col="correct",
        control_cols=("quality",),
    )

    assert n == len(frame)
    assert effect > 0.9


def test_candidate_pump_times_rejects_irregular_sensor_pulses():
    stable = SimpleNamespace(pulsetimes=np.asarray([[1000, 4000], [8000, 11000], [15000, 18000]]))
    irregular = SimpleNamespace(pulsetimes=np.asarray([[1000, 2000], [3000, 9000], [10000, 10100]]))
    epoch = np.asarray([stable, irregular], dtype=object)

    candidates = _candidate_pump_times(epoch)

    assert list(candidates) == [1]
    np.testing.assert_allclose(candidates[1], [0.4, 1.1, 1.8])


def test_gate_summary_never_promotes_result_to_bayesian_smoothing():
    n_trials = 1000
    trials = pd.DataFrame(
        {
            "animal": np.resize(["a", "b", "c", "d", "e"], n_trials),
            "alternation_consistent": True,
            "choice_analysis_exposure_s": 1.0,
        }
    )
    events = pd.DataFrame({"event": np.arange(500)})
    validation = pd.DataFrame(
        [
            {"scope": "animal", "animal": animal, "agreement_fraction": 0.9, "n_trials": 100}
            for animal in ["a", "b", "c"]
        ]
        + [{"scope": "pooled", "animal": "all", "agreement_fraction": 0.9, "n_trials": 300}]
    )
    associations = pd.DataFrame(
        {
            "analysis_scope": ["choice_surprise", "choice_surprise"],
            "endpoint": ["trajectory_event_rate_hz", "replay_total_distance"],
            "partial_spearman": [0.3, 0.4],
            "positive_supported": [True, True],
        }
    )
    audit = pd.DataFrame({"status": ["pass"]})

    gates = build_gate_summary(trials, events, validation, associations, audit)

    assert gates.set_index("gate").at["technical_overall", "passed"]
    assert gates.set_index("gate").at["surprise_gated_retrospective_replay_supported", "passed"]
    assert not gates.set_index("gate").at["bayesian_smoothing_identified", "passed"]
