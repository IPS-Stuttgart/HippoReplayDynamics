from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


def load_module():
    root = Path(__file__).resolve().parents[1]
    scripts = root / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    path = scripts / "score_hc11_webshare_native_ripple_evidence.py"
    spec = importlib.util.spec_from_file_location("score_hc11_webshare_native_ripple_evidence", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


hc11 = load_module()


def test_periodic_transition_wraps_across_coordinate_seam() -> None:
    centers = np.array([0.5, 1.5, 2.5, 3.5])
    linear = hc11.topology_gaussian_transition(
        centers,
        0.75,
        4.0,
        topology="linear",
        track_length_cm=4.0,
    ).toarray()
    circular = hc11.topology_gaussian_transition(
        centers,
        0.75,
        4.0,
        topology="circular",
        track_length_cm=4.0,
    ).toarray()

    assert circular[3, 0] == pytest.approx(circular[1, 0])
    assert circular[3, 0] > linear[3, 0]
    np.testing.assert_allclose(circular.sum(axis=0), 1.0)


def test_topology_distance_uses_geodesic_distance_on_circle() -> None:
    linear = hc11.topology_distance(np.array([0.5]), np.array([9.5]), "linear", 10.0)
    circular = hc11.topology_distance(np.array([0.5]), np.array([9.5]), "circular", 10.0)

    assert linear.item() == pytest.approx(9.0)
    assert circular.item() == pytest.approx(1.0)


def test_exact_four_model_direction_mixture_scores_are_finite() -> None:
    centers = np.arange(0.5, 8.0, 1.0)
    edges = np.arange(0.0, 9.0, 1.0)
    rates_forward = np.array(
        [
            [20, 8, 1, 0.2, 0.1, 0.1, 0.1, 0.1],
            [0.1, 1, 8, 20, 8, 1, 0.1, 0.1],
            [0.1, 0.1, 0.1, 1, 8, 20, 8, 1],
        ],
        dtype=float,
    )
    rates_reverse = rates_forward[:, ::-1]
    occupancy = np.ones(8)
    prior = np.full(8, 1.0 / 8.0)
    maps = [
        hc11.EncodingMap("negative_direction", (1, 2, 3), edges, centers, occupancy, prior, rates_reverse),
        hc11.EncodingMap("positive_direction", (1, 2, 3), edges, centers, occupancy, prior, rates_forward),
    ]
    counts = np.array([[1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 1, 1], [0, 0, 1]], dtype=int)
    time_edges = np.arange(0.0, 0.06, 0.01)

    scores = hc11.score_encoding_variant(
        counts,
        time_edges,
        maps,
        topology="linear",
        track_length_cm=8.0,
        diffusion_sigma_cm_sqrt_s=8.0,
        stationary_sigma_cm=0.5,
        max_step_sigma=4.0,
        imm_mode_stickiness=0.8,
    )

    assert set(scores) == set(hc11.MODELS)
    assert all(np.isfinite(score["log_evidence"]) for score in scores.values())
    assert np.asarray(scores["first_order_imm"]["posterior"]).shape == (5, 8)
    assert np.asarray(scores["first_order_imm"]["mode_posterior"]).shape == (5, 3)


def test_event_decisions_keep_family_and_imm_axes_separate() -> None:
    rows = []
    values = {"stationary": 0.0, "diffusion": 2.0, "fragmented": 3.0, "first_order_imm": 10.0}
    for model, logz in values.items():
        rows.append(
            {
                "animal": "RatA",
                "session": "RatA_day1",
                "geometry": "linear",
                "maze_type": "Linear Maze",
                "event_id": 7,
                "selection_rank_within_session": 1,
                "encoding_variant": "direction_mixture",
                "model": model,
                "log_evidence": logz,
                "status": "success",
                "duration_ms": 50.0,
                "raw_ripple_duration_ms": 40.0,
                "n_spikes": 12,
                "raw_ripple_n_spikes": 10,
                "n_active_units": 5,
                "n_time_bins": 5,
                "mean_stationary_mode_probability": 0.2,
                "mean_nonstationary_mode_probability": 0.8,
                "fraction_time_map_nonstationary": 0.8,
                "posterior_expected_path_length_cm": 40.0,
                "posterior_net_displacement_cm": 30.0,
                "posterior_path_speed_cm_s": 800.0,
            }
        )

    decisions = hc11.event_decisions(pd.DataFrame(rows), margin_threshold=5.5)

    assert len(decisions) == 1
    assert decisions.iloc[0]["delta_trajectory_minus_stationary"] == pytest.approx(10.0)
    assert decisions.iloc[0]["delta_imm_minus_fragmented"] == pytest.approx(7.0)
    assert bool(decisions.iloc[0]["trajectory_confident_claim"])
    assert bool(decisions.iloc[0]["imm_confident_over_fragmented"])


def test_zero_event_gate_cannot_pass_vacuously() -> None:
    gates = hc11.gate_summary(
        0,
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        20,
    )
    passed = dict(zip(gates["gate"], gates["passed"], strict=True))

    assert not bool(passed["native_ripple_sessions_present"])
    assert not bool(passed["required_models_complete"])
    assert not bool(passed["overall_technical"])


def test_event_ranking_is_pre_evidence_and_deterministic() -> None:
    events = pd.DataFrame(
        {
            "event_id": [1, 2, 3, 4],
            "peak_ripple_power_z": [10.0, 40.0, 30.0, 20.0],
            "n_spikes": [10, 11, 12, 30],
            "n_active_units": [4, 5, 6, 9],
        }
    )
    support = hc11.rank_native_events(
        events,
        event_ranking="spike_support",
        max_events=2,
        selection_seed=7,
        session_name="RatA_day1",
    )
    random_a = hc11.rank_native_events(
        events,
        event_ranking="random",
        max_events=3,
        selection_seed=7,
        session_name="RatA_day1",
    )
    random_b = hc11.rank_native_events(
        events,
        event_ranking="random",
        max_events=3,
        selection_seed=7,
        session_name="RatA_day1",
    )

    assert support["event_id"].tolist() == [4, 3]
    assert support["selection_score_name"].eq("n_active_units_then_n_spikes").all()
    assert random_a["event_id"].tolist() == random_b["event_id"].tolist()
    assert not any("evidence" in column.lower() for column in support.columns)
