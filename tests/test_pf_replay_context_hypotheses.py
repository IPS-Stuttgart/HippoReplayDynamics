from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.test_pf_replay_context_hypotheses import (
    build_context_event_table,
    infer_home_wells,
    run_context_hypotheses,
)


def _fixtures() -> tuple[pd.DataFrame, ...]:
    event_rows = []
    frozen_rows = []
    bin_rows = []
    eligibility_rows = []
    route_rows = []
    point_rows = []
    event_index = 0
    for rat_index, rat in enumerate(("Rat1", "Rat2", "Rat3", "Rat4"), start=1):
        session = f"{rat}/Open1"
        home = 10
        routes = [(1, home, 11), (2, 11, home), (3, home, 12), (4, 12, home)]
        for route_index, origin, destination in routes:
            route_id = f"{session}:route_{route_index:03d}"
            start = 100.0 * route_index
            route_rows.append(
                {
                    "session": session,
                    "rat": rat,
                    "route_id": route_id,
                    "route_index": route_index,
                    "cv_fold": route_index % 2,
                    "movement_start_time_s": start,
                    "movement_end_time_s": start + 10.0,
                    "origin_well_id": origin,
                    "destination_well_id": destination,
                }
            )
            for point_index in range(6):
                point_rows.append(
                    {
                        "session": session,
                        "rat": rat,
                        "route_id": route_id,
                        "route_index": route_index,
                        "cv_fold": route_index % 2,
                        "point_index": point_index,
                        "time_s": start + 2.0 * point_index,
                        "x_cm": 20.0 * point_index,
                        "y_cm": float(10 * route_index),
                    }
                )
        for rank in range(3):
            peak = 299.0 - (2 - rank)
            momentum_axis = float(rank * 3 + rat_index * 0.01)
            event_rows.append(
                {
                    "session": session,
                    "rat": rat,
                    "event_index": event_index,
                    "delta_momentum_minus_imm": momentum_axis,
                    "delta_imm_minus_fragmented": 8.0 - momentum_axis,
                    "trajectory_minus_stationary_log_evidence": 12.0,
                    "future_commitment_index": float(rank),
                    "future_commitment_index_cm": float(rank),
                    "emission_only_future_commitment_index_cm": float(rank * 5),
                    "event_duration_ms": 100.0 + rank * 10.0,
                    "n_spikes": 30 + rank,
                    "active_cell_count": 12,
                    "posterior_entropy": 2.0,
                    "posterior_path_length_cm": 100.0,
                    "run_decoder_error_cm": 10.0,
                    "route_frequency": 0 if rank == 0 else 1,
                    "previous_reward_arrival_time_s": 290.0,
                }
            )
            frozen_rows.append(
                {
                    "session": session,
                    "rat": rat,
                    "event_index": event_index,
                    "n_time": 25,
                    "logZ_stationary": 0.0,
                    "logZ_diffusion": 1.0,
                    "logZ_fragmented": 2.0,
                    "logZ_first_order_imm": 5.0,
                    "logZ_momentum_exact_sparse": 5.0 + momentum_axis,
                }
            )
            eligibility_rows.append(
                {
                    "session": session,
                    "rat": rat,
                    "event_index": event_index,
                    "event_peak_s": peak,
                    "enclosing_route_id": f"{session}:route_003",
                    "enclosing_route_index": 3,
                    "event_route_relation": "next_movement",
                    "excluded_cv_fold": 1,
                    "origin_well_id": home,
                    "destination_well_id": 12,
                    "route_movement_start_time_s": 300.0,
                    "route_movement_end_time_s": 310.0,
                }
            )
            for time_bin in range(25):
                bin_rows.append(
                    {
                        "session": session,
                        "rat": rat,
                        "event_index": event_index,
                        "time_bin": time_bin,
                        "emission_only_mean_x_cm": 4.0 * time_bin,
                        "emission_only_mean_y_cm": 30.0,
                        "map_mode_index": 1 if time_bin < 12 else 0,
                    }
                )
            event_index += 1
    return tuple(
        pd.DataFrame(rows)
        for rows in (
            event_rows,
            frozen_rows,
            bin_rows,
            route_rows,
            point_rows,
            eligibility_rows,
        )
    )


def test_context_table_recovers_home_pause_and_independent_paths() -> None:
    metrics, frozen, bins, routes, points, eligibility = _fixtures()
    home = infer_home_wells(routes)
    assert home["home_well"].tolist() == [10, 10, 10, 10]
    assert home["all_routes_involve_home"].all()

    events, pauses = build_context_event_table(
        metrics,
        frozen,
        bins,
        routes,
        points,
        eligibility,
    )

    assert len(events) == 12
    assert len(pauses) == 4
    assert set(events["goal_context"]) == {"away_bound"}
    assert events["emission_prospective_index_cm"].notna().all()
    assert events["future_route_novelty_cm"].notna().all()
    assert events.groupby("pause_id")["final_event_in_pause"].sum().eq(1).all()


def test_h1_detects_within_pause_commitment_progression() -> None:
    metrics, frozen, bins, routes, points, eligibility = _fixtures()
    events, _ = build_context_event_table(
        metrics,
        frozen,
        bins,
        routes,
        points,
        eligibility,
    )

    tests, by_rat, _, nulls, pause_effects = run_context_hypotheses(
        events,
        permutations=99,
        bootstraps=200,
        seed=7,
    )

    h1 = tests[tests["test"].eq("momentum_axis_increases_toward_departure")].iloc[0]
    assert h1["estimate"] > 0.0
    assert h1["permutation_p_value"] <= 0.05
    assert pause_effects["rank_slope"].gt(0.0).all()
    assert set(by_rat[by_rat["hypothesis"].eq("H1")]["rat"]) == {
        "Rat1",
        "Rat2",
        "Rat3",
        "Rat4",
    }
    assert len(nulls) > 0
    assert np.isfinite(h1["estimate"])
