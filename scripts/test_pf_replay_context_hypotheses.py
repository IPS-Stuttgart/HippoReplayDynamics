#!/usr/bin/env python3
"""Test frozen Pfeiffer/Foster replay context hypotheses H1-H4 and H10."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _provenance import build_script_provenance  # noqa: E402
from compute_replay_commitment_composition_metrics import (  # noqa: E402
    path_fit_distance_cm,
    path_length,
)
from test_replay_dynamics_behavior_hypotheses import (  # noqa: E402
    adjusted_coefficient,
    rat_cluster_bootstrap,
)


KEYS = ["session", "rat", "event_index"]
EVENT_OUTPUT = "pf_replay_context_hypothesis_events.csv"
PAUSE_OUTPUT = "pf_replay_context_pause_summary.csv"
TEST_OUTPUT = "pf_replay_context_hypothesis_tests.csv"
BY_RAT_OUTPUT = "pf_replay_context_hypothesis_by_rat.csv"
LOO_OUTPUT = "pf_replay_context_hypothesis_leave_one_rat_out.csv"
NULL_OUTPUT = "pf_replay_context_hypothesis_nulls.csv"
GATE_OUTPUT = "pf_replay_context_hypothesis_gate_summary.csv"
REPORT_OUTPUT = "pf_replay_context_hypothesis_report.md"
MANIFEST_OUTPUT = "pf_replay_context_hypothesis_manifest.json"

QUALITY_CONTROLS = (
    "log_n_spikes",
    "active_cell_count",
    "posterior_entropy",
    "trajectory_minus_stationary_log_evidence",
    "log_posterior_path_length_cm",
    "run_decoder_error_cm",
)


def _path_dictionary(points: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        str(route_id): group.sort_values("point_index").reset_index(drop=True)
        for route_id, group in points.groupby("route_id", sort=False)
    }


def infer_home_wells(routes: pd.DataFrame) -> pd.DataFrame:
    """Infer and validate the unique fixed Home well independently of replay."""

    rows: list[dict[str, object]] = []
    for session, group in routes.groupby("session", sort=True):
        endpoints = pd.concat(
            [group["origin_well_id"], group["destination_well_id"]],
            ignore_index=True,
        )
        counts = endpoints.value_counts()
        home = int(counts.index[0])
        involving = group["origin_well_id"].eq(home) | group["destination_well_id"].eq(home)
        runner_up = int(counts.iloc[1]) if len(counts) > 1 else 0
        rows.append(
            {
                "session": str(session),
                "home_well": home,
                "routes": int(len(group)),
                "home_endpoint_count": int(counts.iloc[0]),
                "runner_up_endpoint_count": runner_up,
                "all_routes_involve_home": bool(involving.all()),
                "home_unique": bool(int(counts.iloc[0]) > runner_up),
            }
        )
    return pd.DataFrame(rows)


def _valid_path(frame: pd.DataFrame) -> np.ndarray:
    if frame.empty:
        return np.empty((0, 2), dtype=float)
    path = frame[["x_cm", "y_cm"]].to_numpy(dtype=float)
    path = path[np.isfinite(path).all(axis=1)]
    return path if len(path) >= 2 and path_length(path) > 1e-9 else np.empty((0, 2), dtype=float)


def _path_json(path: np.ndarray) -> str:
    return json.dumps(np.asarray(path, dtype=float).tolist(), separators=(",", ":"))


def _path_from_json(value: object) -> np.ndarray:
    path = np.asarray(json.loads(str(value)), dtype=float)
    return path if path.ndim == 2 and path.shape[1:] == (2,) else np.empty((0, 2), dtype=float)


def _event_templates(
    row: pd.Series,
    *,
    routes_by_index: dict[tuple[str, int], pd.Series],
    route_paths: dict[str, pd.DataFrame],
) -> tuple[np.ndarray, np.ndarray]:
    session = str(row["session"])
    route_index = int(row["enclosing_route_index"])
    route_id = str(row["enclosing_route_id"])
    peak = float(row["event_peak_s"])
    relation = str(row["event_route_relation"])
    current = route_paths.get(route_id, pd.DataFrame())
    if relation == "next_movement":
        future = _valid_path(current)
        previous_row = routes_by_index.get((session, route_index - 1))
        previous = (
            route_paths.get(str(previous_row["route_id"]), pd.DataFrame())
            if previous_row is not None
            else pd.DataFrame()
        )
        past = _valid_path(previous)[::-1]
        return past, future
    if relation == "during_movement" and not current.empty and "time_s" in current:
        past = _valid_path(current[current["time_s"] <= peak])[::-1]
        future = _valid_path(current[current["time_s"] >= peak])
        return past, future
    return np.empty((0, 2), dtype=float), np.empty((0, 2), dtype=float)


def _candidate_suffix(path_frame: pd.DataFrame, start_xy: np.ndarray) -> tuple[np.ndarray, float]:
    xy = path_frame[["x_cm", "y_cm"]].to_numpy(dtype=float)
    if len(xy) < 2:
        return np.empty((0, 2), dtype=float), np.inf
    distances = np.linalg.norm(xy - np.asarray(start_xy, dtype=float), axis=1)
    index = int(np.argmin(distances))
    suffix = xy[index:]
    if len(suffix) < 2 or path_length(suffix) <= 1e-9:
        return np.empty((0, 2), dtype=float), float(distances[index])
    return suffix, float(distances[index])


def _future_route_novelty(
    row: pd.Series,
    future: np.ndarray,
    *,
    routes: pd.DataFrame,
    route_paths: dict[str, pd.DataFrame],
    maximum_start_distance_cm: float,
) -> tuple[float, int]:
    if len(future) < 2:
        return np.nan, 0
    candidates = routes[
        routes["session"].astype(str).eq(str(row["session"]))
        & ~routes["cv_fold"].eq(int(row["excluded_cv_fold"]))
    ]
    distances: list[float] = []
    for candidate_id in candidates["route_id"].astype(str):
        candidate_frame = route_paths.get(candidate_id, pd.DataFrame())
        candidate, start_distance = _candidate_suffix(candidate_frame, future[0])
        if start_distance > float(maximum_start_distance_cm) or len(candidate) < 2:
            continue
        try:
            distances.append(path_fit_distance_cm(future, candidate))
        except ValueError:
            continue
    return (float(min(distances)), len(distances)) if distances else (np.nan, 0)


def build_context_event_table(
    event_metrics: pd.DataFrame,
    frozen_events: pd.DataFrame,
    posterior_bins: pd.DataFrame,
    route_segments: pd.DataFrame,
    route_points: pd.DataFrame,
    eligibility: pd.DataFrame,
    *,
    maximum_start_distance_cm: float = 30.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Join frozen evidence to behavior-only context without testing outcomes."""

    frozen_columns = [
        *KEYS,
        "n_time",
        "logZ_stationary",
        "logZ_diffusion",
        "logZ_fragmented",
        "logZ_first_order_imm",
        "logZ_momentum_exact_sparse",
    ]
    frame = event_metrics.merge(
        frozen_events[frozen_columns],
        on=KEYS,
        how="inner",
        validate="one_to_one",
    ).merge(
        eligibility,
        on=KEYS,
        how="inner",
        validate="one_to_one",
        suffixes=("", "_eligibility"),
    )
    home = infer_home_wells(route_segments)
    if not home["all_routes_involve_home"].all() or not home["home_unique"].all():
        raise ValueError("behavior routes do not identify one unique Home well per session")
    frame = frame.merge(home[["session", "home_well"]], on="session", validate="many_to_one")
    frame["goal_context"] = np.where(
        frame["destination_well_id"].astype(int).eq(frame["home_well"].astype(int)),
        "home_bound",
        np.where(
            frame["origin_well_id"].astype(int).eq(frame["home_well"].astype(int)),
            "away_bound",
            "other",
        ),
    )

    route_paths = _path_dictionary(route_points)
    routes_by_index = {
        (str(row.session), int(row.route_index)): pd.Series(row._asdict())
        for row in route_segments.itertuples(index=False)
    }
    bin_groups = {
        (str(session), str(rat), int(event_index)): group.sort_values("time_bin")
        for (session, rat, event_index), group in posterior_bins.groupby(KEYS, sort=False)
    }
    derived_rows: list[dict[str, object]] = []
    for row in frame.itertuples(index=False):
        series = pd.Series(row._asdict())
        key = (str(row.session), str(row.rat), int(row.event_index))
        event_bins = bin_groups.get(key, pd.DataFrame())
        emission_path = (
            event_bins[["emission_only_mean_x_cm", "emission_only_mean_y_cm"]]
            .to_numpy(dtype=float)
            if not event_bins.empty
            else np.empty((0, 2), dtype=float)
        )
        past, future = _event_templates(
            series,
            routes_by_index=routes_by_index,
            route_paths=route_paths,
        )
        past_error = np.nan
        future_error = np.nan
        if len(emission_path) >= 2 and path_length(emission_path) > 1e-9:
            try:
                past_error = path_fit_distance_cm(emission_path, past)
            except ValueError:
                pass
            try:
                future_error = path_fit_distance_cm(emission_path, future)
            except ValueError:
                pass
        novelty, novelty_candidates = _future_route_novelty(
            series,
            future,
            routes=route_segments,
            route_paths=route_paths,
            maximum_start_distance_cm=float(maximum_start_distance_cm),
        )
        goal_error = (
            float(np.linalg.norm(emission_path[-1] - future[-1]))
            if len(emission_path) and len(future)
            else np.nan
        )
        nonfragmented_switches = 0
        if not event_bins.empty:
            modes = event_bins["map_mode_index"].to_numpy(dtype=int)
            nonfragmented_switches = int(
                np.sum((modes[1:] != modes[:-1]) & (modes[1:] != 2) & (modes[:-1] != 2))
            )
        duration_s = float(row.event_duration_ms) / 1000.0
        derived_rows.append(
            {
                **dict(zip(KEYS, key, strict=True)),
                "emission_past_route_error_cm": past_error,
                "emission_future_route_error_cm": future_error,
                "emission_path_xy_json": _path_json(emission_path),
                "past_template_xy_json": _path_json(past),
                "future_template_xy_json": _path_json(future),
                "emission_prospective_index_cm": (
                    past_error - future_error
                    if np.isfinite(past_error) and np.isfinite(future_error)
                    else np.nan
                ),
                "emission_next_goal_error_cm": goal_error,
                "future_route_novelty_cm": novelty,
                "future_route_novelty_candidates": int(novelty_candidates),
                "route_identity_unseen": bool(int(row.route_frequency) == 0),
                "nonfragmented_map_switch_count": nonfragmented_switches,
                "nonfragmented_map_switch_rate_hz": (
                    nonfragmented_switches / duration_s if duration_s > 0.0 else np.nan
                ),
            }
        )
    frame = frame.merge(pd.DataFrame(derived_rows), on=KEYS, validate="one_to_one")
    frame["log_n_spikes"] = np.log1p(pd.to_numeric(frame["n_spikes"], errors="coerce"))
    frame["log_posterior_path_length_cm"] = np.log1p(
        pd.to_numeric(frame["posterior_path_length_cm"], errors="coerce").clip(lower=0.0)
    )
    frame["log_event_duration_s"] = np.log(
        pd.to_numeric(frame["event_duration_ms"], errors="coerce") / 1000.0
    )
    frame["delta_momentum_minus_imm_per_bin"] = (
        frame["delta_momentum_minus_imm"] / frame["n_time"]
    )
    frame["spikes_per_bin"] = frame["n_spikes"] / frame["n_time"]

    pause_mask = frame["event_route_relation"].eq("next_movement")
    frame["pause_id"] = pd.NA
    frame.loc[pause_mask, "pause_id"] = (
        frame.loc[pause_mask, "session"].astype(str)
        + "::"
        + frame.loc[pause_mask, "enclosing_route_id"].astype(str)
    )
    frame["pause_event_count"] = 0
    frame["pause_event_rank"] = np.nan
    frame["normalized_pause_rank"] = np.nan
    frame["final_event_in_pause"] = False
    for pause_id, group in frame[pause_mask].groupby("pause_id", sort=False):
        ordered = group.sort_values(["event_peak_s", "event_index"])
        ranks = np.arange(len(ordered), dtype=int)
        frame.loc[ordered.index, "pause_event_count"] = len(ordered)
        frame.loc[ordered.index, "pause_event_rank"] = ranks
        frame.loc[ordered.index, "normalized_pause_rank"] = (
            ranks / (len(ordered) - 1) if len(ordered) > 1 else 0.0
        )
        frame.loc[ordered.index[-1], "final_event_in_pause"] = True
    pause_rows: list[dict[str, object]] = []
    for pause_id, group in frame[frame["pause_event_count"].ge(2)].groupby("pause_id", sort=True):
        ordered = group.sort_values("pause_event_rank")
        pause_rows.append(
            {
                "pause_id": pause_id,
                "session": str(ordered["session"].iloc[0]),
                "rat": str(ordered["rat"].iloc[0]),
                "events": int(len(ordered)),
                "first_event_index": int(ordered["event_index"].iloc[0]),
                "final_event_index": int(ordered["event_index"].iloc[-1]),
                "pause_duration_s": float(
                    ordered["route_movement_start_time_s"].iloc[0]
                    - ordered["previous_reward_arrival_time_s"].iloc[0]
                ),
            }
        )
    return frame, pd.DataFrame(pause_rows)


def _standardize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    scale = float(np.std(values))
    return (values - float(np.mean(values))) / scale if scale > 0.0 else np.zeros_like(values)


def _event_effect(
    frame: pd.DataFrame,
    *,
    outcome: str,
    predictor: str,
    controls: Sequence[str],
) -> float:
    return adjusted_coefficient(
        frame,
        outcome=outcome,
        predictor=predictor,
        numeric_controls=tuple(controls),
        categorical_controls=("session",),
    )[0]


def _residualize(frame: pd.DataFrame, outcome: str, controls: Sequence[str]) -> pd.Series:
    selected = frame[np.isfinite(pd.to_numeric(frame[outcome], errors="coerce"))].copy()
    if selected.empty:
        return pd.Series(dtype=float)
    columns = [np.ones(len(selected), dtype=float)]
    for control in controls:
        values = pd.to_numeric(selected[control], errors="coerce").to_numpy(dtype=float)
        median = float(np.nanmedian(values)) if np.isfinite(values).any() else 0.0
        values[~np.isfinite(values)] = median
        if np.std(values) > 0.0:
            columns.append(_standardize(values))
    session_dummies = pd.get_dummies(selected["session"].astype(str), drop_first=True, dtype=float)
    columns.extend(session_dummies[column].to_numpy(dtype=float) for column in session_dummies)
    design = np.column_stack(columns)
    y = _standardize(pd.to_numeric(selected[outcome], errors="raise").to_numpy(dtype=float))
    residual = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    return pd.Series(residual, index=selected.index, dtype=float)


def _pause_effects(frame: pd.DataFrame, outcome: str) -> pd.DataFrame:
    selected = frame[
        frame["pause_event_count"].ge(2)
        & np.isfinite(pd.to_numeric(frame[outcome], errors="coerce"))
    ].copy()
    residuals = _residualize(selected, outcome, QUALITY_CONTROLS)
    selected["residual_outcome"] = residuals
    rows: list[dict[str, object]] = []
    for pause_id, group in selected.groupby("pause_id", sort=True):
        if len(group) < 2:
            continue
        x = group["normalized_pause_rank"].to_numpy(dtype=float)
        y = group["residual_outcome"].to_numpy(dtype=float)
        denominator = float(np.sum((x - x.mean()) ** 2))
        slope = float(np.sum((x - x.mean()) * (y - y.mean())) / denominator)
        final = group[group["final_event_in_pause"]]["residual_outcome"]
        earlier = group[~group["final_event_in_pause"]]["residual_outcome"]
        rows.append(
            {
                "pause_id": pause_id,
                "session": str(group["session"].iloc[0]),
                "rat": str(group["rat"].iloc[0]),
                "events": int(len(group)),
                "rank_slope": slope,
                "final_minus_earlier": (
                    float(final.iloc[0] - earlier.mean())
                    if len(final) == 1 and len(earlier)
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def _rat_mean_bootstrap(
    effects: pd.DataFrame,
    column: str,
    *,
    replicates: int,
    seed: int,
) -> tuple[float, float, int]:
    groups = {
        rat: group[column].dropna().to_numpy(dtype=float)
        for rat, group in effects.groupby("rat", sort=True)
    }
    groups = {rat: values for rat, values in groups.items() if len(values)}
    if len(groups) < 2:
        return np.nan, np.nan, 0
    rats = sorted(groups)
    rng = np.random.default_rng(seed)
    draws: list[float] = []
    for _ in range(int(replicates)):
        sampled = rng.choice(rats, size=len(rats), replace=True)
        draws.append(float(np.mean([np.mean(groups[str(rat)]) for rat in sampled])))
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975)), len(draws)


def _permutation_p(observed: float, null: np.ndarray, expected_sign: int) -> float:
    null = np.asarray(null, dtype=float)
    null = null[np.isfinite(null)]
    if not np.isfinite(observed) or not len(null):
        return np.nan
    if expected_sign > 0:
        extreme = np.sum(null >= observed)
    elif expected_sign < 0:
        extreme = np.sum(null <= observed)
    else:
        extreme = np.sum(np.abs(null) >= abs(observed))
    return float((1 + extreme) / (1 + len(null)))


def _permuted_predictor(
    frame: pd.DataFrame,
    *,
    predictor: str,
    groups: Sequence[str],
    rng: np.random.Generator,
) -> pd.DataFrame:
    permuted = frame.copy()
    permuted[predictor] = permuted.groupby(
        list(groups), sort=False, dropna=False
    )[predictor].transform(lambda values: rng.permutation(values.to_numpy()))
    return permuted


def _h2_route_assignment_shift_null(
    frame: pd.DataFrame,
    *,
    controls: Sequence[str],
    permutations: int,
    seed: int,
) -> pd.DataFrame:
    """Reassign behavior templates circularly while keeping each decoded path fixed."""

    required = (
        "emission_path_xy_json",
        "past_template_xy_json",
        "future_template_xy_json",
    )
    selected = frame[
        np.isfinite(pd.to_numeric(frame["emission_prospective_index_cm"], errors="coerce"))
        & np.isfinite(pd.to_numeric(frame["delta_momentum_minus_imm"], errors="coerce"))
    ].copy()
    if selected.empty or any(column not in selected for column in required):
        return pd.DataFrame(columns=["hypothesis", "test", "replicate", "null_estimate"])
    decoded = {
        index: _path_from_json(value)
        for index, value in selected["emission_path_xy_json"].items()
    }
    past = {
        index: _path_from_json(value)
        for index, value in selected["past_template_xy_json"].items()
    }
    future = {
        index: _path_from_json(value)
        for index, value in selected["future_template_xy_json"].items()
    }
    ordered_groups = [
        group.sort_values(["event_peak_s", "event_index"]).index.to_numpy()
        for _, group in selected.groupby("session", sort=True)
        if len(group) >= 2
    ]
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for replicate in range(int(permutations)):
        shifted = selected.copy()
        shifted["emission_prospective_index_cm"] = np.nan
        for indices in ordered_groups:
            offset = int(rng.integers(1, len(indices)))
            template_indices = np.roll(indices, offset)
            for event_index, template_index in zip(indices, template_indices, strict=True):
                try:
                    past_error = path_fit_distance_cm(decoded[event_index], past[template_index])
                    future_error = path_fit_distance_cm(
                        decoded[event_index], future[template_index]
                    )
                except ValueError:
                    continue
                shifted.loc[event_index, "emission_prospective_index_cm"] = (
                    past_error - future_error
                )
        rows.append(
            {
                "hypothesis": "H2",
                "test": "clean_imm_is_more_prospective_than_momentum",
                "replicate": replicate,
                "null_estimate": _event_effect(
                    shifted,
                    outcome="emission_prospective_index_cm",
                    predictor="delta_momentum_minus_imm",
                    controls=controls,
                ),
                "null_control": "within_session_circular_behavior_template_shift",
            }
        )
    return pd.DataFrame(rows)


def _effect_status(
    estimate: float,
    low: float,
    high: float,
    p_value: float,
    expected_sign: int,
) -> str:
    if not all(np.isfinite(value) for value in (estimate, low, high, p_value)):
        return "insufficient"
    if expected_sign > 0 and low > 0.0 and p_value <= 0.05:
        return "directional_pass_unadjusted"
    if expected_sign < 0 and high < 0.0 and p_value <= 0.05:
        return "directional_pass_unadjusted"
    if expected_sign == 0 and (low > 0.0 or high < 0.0) and p_value <= 0.05:
        return "two_sided_pass_unadjusted"
    if expected_sign > 0 and high < 0.0:
        return "contradicted"
    if expected_sign < 0 and low > 0.0:
        return "contradicted"
    return "inconclusive"


def _event_hypothesis(
    frame: pd.DataFrame,
    *,
    hypothesis: str,
    test: str,
    outcome: str,
    predictor: str,
    controls: Sequence[str],
    expected_sign: int,
    permutations: int,
    bootstraps: int,
    seed: int,
    permutation_groups: Sequence[str] = ("session",),
    precomputed_null: pd.DataFrame | None = None,
    null_control: str = "within_session_predictor_permutation",
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected = frame[
        np.isfinite(pd.to_numeric(frame[outcome], errors="coerce"))
        & np.isfinite(pd.to_numeric(frame[predictor], errors="coerce"))
    ].copy()
    estimate = _event_effect(selected, outcome=outcome, predictor=predictor, controls=controls)

    def statistic(data: pd.DataFrame) -> float:
        return _event_effect(data, outcome=outcome, predictor=predictor, controls=controls)

    low, high, completed = rat_cluster_bootstrap(
        selected,
        statistic,
        replicates=int(bootstraps),
        seed=int(seed),
    )
    if precomputed_null is None:
        rng = np.random.default_rng(seed + 1000)
        null_rows: list[dict[str, object]] = []
        for replicate in range(int(permutations)):
            permuted = _permuted_predictor(
                selected,
                predictor=predictor,
                groups=permutation_groups,
                rng=rng,
            )
            null_rows.append(
                {
                    "hypothesis": hypothesis,
                    "test": test,
                    "replicate": replicate,
                    "null_estimate": statistic(permuted),
                    "null_control": null_control,
                }
            )
        null = pd.DataFrame(null_rows)
    else:
        null = precomputed_null.copy()
    p_value = _permutation_p(estimate, null["null_estimate"].to_numpy(), expected_sign)
    by_rat_rows: list[dict[str, object]] = []
    for rat, group in selected.groupby("rat", sort=True):
        raw = group[[predictor, outcome]].dropna()
        by_rat_rows.append(
            {
                "hypothesis": hypothesis,
                "test": test,
                "rat": rat,
                "events": int(len(group)),
                "effect": (
                    float(spearmanr(raw[predictor], raw[outcome]).statistic)
                    if len(raw) >= 3
                    else np.nan
                ),
                "expected_direction": bool(
                    expected_sign == 0
                    or (
                        len(raw) >= 3
                        and np.isfinite(spearmanr(raw[predictor], raw[outcome]).statistic)
                        and float(spearmanr(raw[predictor], raw[outcome]).statistic)
                        * expected_sign
                        > 0.0
                    )
                ),
            }
        )
    loo_rows: list[dict[str, object]] = []
    for omitted in sorted(selected["rat"].astype(str).unique()):
        retained = selected[~selected["rat"].astype(str).eq(omitted)]
        value = statistic(retained)
        loo_rows.append(
            {
                "hypothesis": hypothesis,
                "test": test,
                "omitted_rat": omitted,
                "events": int(len(retained)),
                "effect": value,
                "expected_direction": bool(
                    np.isfinite(value) and (expected_sign == 0 or value * expected_sign > 0.0)
                ),
            }
        )
    result = {
        "hypothesis": hypothesis,
        "test": test,
        "role": "primary",
        "outcome": outcome,
        "predictor": predictor,
        "expected_sign": expected_sign,
        "events": int(len(selected)),
        "rats": int(selected["rat"].nunique()),
        "sessions": int(selected["session"].nunique()),
        "estimate": estimate,
        "rat_bootstrap_ci_low": low,
        "rat_bootstrap_ci_high": high,
        "bootstrap_replicates_completed": completed,
        "permutation_p_value": p_value,
        "status_before_campaign_fdr": _effect_status(
            estimate, low, high, p_value, expected_sign
        ),
        "null_control": str(null["null_control"].iloc[0]) if len(null) else null_control,
    }
    return result, pd.DataFrame(by_rat_rows), pd.DataFrame(loo_rows), null


def run_context_hypotheses(
    events: pd.DataFrame,
    *,
    permutations: int,
    bootstraps: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tests: list[dict[str, object]] = []
    by_rat_parts: list[pd.DataFrame] = []
    loo_parts: list[pd.DataFrame] = []
    null_parts: list[pd.DataFrame] = []

    model_pause = _pause_effects(events, "delta_momentum_minus_imm")
    commitment_pause = _pause_effects(events, "emission_only_future_commitment_index_cm")
    model_estimate = float(model_pause["rank_slope"].mean()) if len(model_pause) else np.nan
    model_low, model_high, model_completed = _rat_mean_bootstrap(
        model_pause,
        "rank_slope",
        replicates=bootstraps,
        seed=seed,
    )
    commitment_estimate = (
        float(commitment_pause["final_minus_earlier"].mean())
        if len(commitment_pause)
        else np.nan
    )
    commitment_low, commitment_high, commitment_completed = _rat_mean_bootstrap(
        commitment_pause,
        "final_minus_earlier",
        replicates=bootstraps,
        seed=seed + 1,
    )
    rng = np.random.default_rng(seed + 2000)
    h1_null_rows: list[dict[str, object]] = []
    model_source = events[events["pause_event_count"].ge(2)].copy()
    commitment_source = model_source.copy()
    for replicate in range(int(permutations)):
        shuffled_model = model_source.copy()
        shuffled_commitment = commitment_source.copy()
        shuffled_model["delta_momentum_minus_imm"] = shuffled_model.groupby(
            "pause_id", sort=False
        )["delta_momentum_minus_imm"].transform(lambda values: rng.permutation(values.to_numpy()))
        shuffled_commitment["emission_only_future_commitment_index_cm"] = (
            shuffled_commitment.groupby("pause_id", sort=False)[
                "emission_only_future_commitment_index_cm"
            ].transform(lambda values: rng.permutation(values.to_numpy()))
        )
        shuffled_model_effects = _pause_effects(shuffled_model, "delta_momentum_minus_imm")
        shuffled_commitment_effects = _pause_effects(
            shuffled_commitment,
            "emission_only_future_commitment_index_cm",
        )
        h1_null_rows.extend(
            [
                {
                    "hypothesis": "H1",
                    "test": "momentum_axis_increases_toward_departure",
                    "replicate": replicate,
                    "null_estimate": float(shuffled_model_effects["rank_slope"].mean()),
                    "null_control": "within_pause_event_order_permutation",
                },
                {
                    "hypothesis": "H1",
                    "test": "final_event_best_predicts_future_route",
                    "replicate": replicate,
                    "null_estimate": float(
                        shuffled_commitment_effects["final_minus_earlier"].mean()
                    ),
                    "null_control": "within_pause_event_order_permutation",
                },
            ]
        )
    h1_null = pd.DataFrame(h1_null_rows)
    model_p = _permutation_p(
        model_estimate,
        h1_null[h1_null["test"].eq("momentum_axis_increases_toward_departure")][
            "null_estimate"
        ].to_numpy(),
        1,
    )
    commitment_p = _permutation_p(
        commitment_estimate,
        h1_null[h1_null["test"].eq("final_event_best_predicts_future_route")][
            "null_estimate"
        ].to_numpy(),
        1,
    )
    tests.extend(
        [
            {
                "hypothesis": "H1",
                "test": "momentum_axis_increases_toward_departure",
                "role": "primary",
                "outcome": "delta_momentum_minus_imm",
                "predictor": "normalized_pause_rank",
                "expected_sign": 1,
                "events": int(model_source.shape[0]),
                "rats": int(model_source["rat"].nunique()),
                "sessions": int(model_source["session"].nunique()),
                "estimate": model_estimate,
                "rat_bootstrap_ci_low": model_low,
                "rat_bootstrap_ci_high": model_high,
                "bootstrap_replicates_completed": model_completed,
                "permutation_p_value": model_p,
                "status_before_campaign_fdr": _effect_status(
                    model_estimate, model_low, model_high, model_p, 1
                ),
                "null_control": "within_pause_event_order_permutation",
            },
            {
                "hypothesis": "H1",
                "test": "final_event_best_predicts_future_route",
                "role": "companion_required",
                "outcome": "emission_only_future_commitment_index_cm",
                "predictor": "final_event_in_pause",
                "expected_sign": 1,
                "events": int(
                    model_source["emission_only_future_commitment_index_cm"].notna().sum()
                ),
                "rats": int(model_source["rat"].nunique()),
                "sessions": int(model_source["session"].nunique()),
                "estimate": commitment_estimate,
                "rat_bootstrap_ci_low": commitment_low,
                "rat_bootstrap_ci_high": commitment_high,
                "bootstrap_replicates_completed": commitment_completed,
                "permutation_p_value": commitment_p,
                "status_before_campaign_fdr": _effect_status(
                    commitment_estimate,
                    commitment_low,
                    commitment_high,
                    commitment_p,
                    1,
                ),
                "null_control": "within_pause_event_order_permutation",
            },
        ]
    )
    for test_name, effects, column in (
        ("momentum_axis_increases_toward_departure", model_pause, "rank_slope"),
        ("final_event_best_predicts_future_route", commitment_pause, "final_minus_earlier"),
    ):
        for rat, group in effects.groupby("rat", sort=True):
            value = float(group[column].mean())
            by_rat_parts.append(
                pd.DataFrame(
                    [
                        {
                            "hypothesis": "H1",
                            "test": test_name,
                            "rat": rat,
                            "events": int(group["events"].sum()),
                            "effect": value,
                            "expected_direction": bool(value > 0.0),
                        }
                    ]
                )
            )
    null_parts.append(h1_null)

    specifications = (
        (
            "H2",
            "clean_imm_is_more_prospective_than_momentum",
            "emission_prospective_index_cm",
            "delta_momentum_minus_imm",
            QUALITY_CONTROLS,
            -1,
        ),
        (
            "H3",
            "imm_favored_for_novel_future_routes",
            "delta_momentum_minus_imm",
            "future_route_novelty_cm",
            (*QUALITY_CONTROLS, "event_duration_ms"),
            -1,
        ),
        (
            "H4",
            "home_vs_away_dynamics_difference",
            "delta_momentum_minus_imm",
            "home_bound",
            (*QUALITY_CONTROLS, "event_duration_ms"),
            0,
        ),
        (
            "H10",
            "long_events_shift_from_momentum_after_per_bin_normalization",
            "delta_momentum_minus_imm_per_bin",
            "log_event_duration_s",
            (
                "spikes_per_bin",
                "active_cell_count",
                "posterior_entropy",
                "trajectory_minus_stationary_log_evidence",
                "run_decoder_error_cm",
            ),
            -1,
        ),
    )
    analysis = events.copy()
    analysis["home_bound"] = analysis["goal_context"].eq("home_bound").astype(float)
    analysis["spike_count_quartile"] = analysis.groupby("session", sort=False)[
        "n_spikes"
    ].transform(
        lambda values: pd.qcut(
            values.rank(method="first"),
            q=min(4, len(values)),
            labels=False,
            duplicates="drop",
        )
    )
    for index, specification in enumerate(specifications):
        precomputed_null = None
        permutation_groups: tuple[str, ...] = ("session",)
        null_control = "within_session_predictor_permutation"
        if specification[0] == "H2":
            precomputed_null = _h2_route_assignment_shift_null(
                analysis,
                controls=specification[4],
                permutations=permutations,
                seed=seed + 101,
            )
            null_control = "within_session_circular_behavior_template_shift"
        elif specification[0] == "H10":
            permutation_groups = ("session", "spike_count_quartile")
            null_control = "within_session_spike_quartile_duration_permutation"
        result, by_rat, loo, null = _event_hypothesis(
            analysis,
            hypothesis=specification[0],
            test=specification[1],
            outcome=specification[2],
            predictor=specification[3],
            controls=specification[4],
            expected_sign=specification[5],
            permutations=permutations,
            bootstraps=bootstraps,
            seed=seed + 10 + index,
            permutation_groups=permutation_groups,
            precomputed_null=precomputed_null,
            null_control=null_control,
        )
        tests.append(result)
        by_rat_parts.append(by_rat)
        loo_parts.append(loo)
        null_parts.append(null)

    return (
        pd.DataFrame(tests),
        pd.concat(by_rat_parts, ignore_index=True),
        pd.concat(loo_parts, ignore_index=True) if loo_parts else pd.DataFrame(),
        pd.concat(null_parts, ignore_index=True),
        model_pause.merge(
            commitment_pause[["pause_id", "final_minus_earlier"]],
            on="pause_id",
            how="outer",
            suffixes=("_model", "_commitment"),
        ),
    )


def build_gates(events: pd.DataFrame, pauses: pd.DataFrame, tests: pd.DataFrame) -> pd.DataFrame:
    primary = tests[tests["role"].eq("primary")]
    rows = [
        ("all_frozen_events_present", len(events) == 160, len(events), 160),
        ("all_four_rats_present", events["rat"].nunique() == 4, events["rat"].nunique(), 4),
        ("all_eight_sessions_present", events["session"].nunique() == 8, events["session"].nunique(), 8),
        ("unique_event_keys", not events.duplicated(KEYS).any(), int(events.duplicated(KEYS).sum()), 0),
        ("home_away_context_complete", events["goal_context"].isin(["home_bound", "away_bound"]).all(), int(events["goal_context"].isin(["home_bound", "away_bound"]).sum()), len(events)),
        ("multi_event_pause_cohort", len(pauses) >= 20, len(pauses), ">=20 pauses"),
        ("prospective_templates_available", events["emission_prospective_index_cm"].notna().sum() >= 100, int(events["emission_prospective_index_cm"].notna().sum()), ">=100 events"),
        ("route_novelty_available", events["future_route_novelty_cm"].notna().sum() >= 100, int(events["future_route_novelty_cm"].notna().sum()), ">=100 events"),
        ("primary_tests_computed", len(primary) == 5 and primary["estimate"].notna().all(), int(primary["estimate"].notna().sum()), 5),
        ("permutation_nulls_complete", primary["permutation_p_value"].notna().all(), int(primary["permutation_p_value"].notna().sum()), 5),
    ]
    gates = pd.DataFrame(
        {"gate": gate, "passed": bool(passed), "value": value, "required": required}
        for gate, passed, value, required in rows
    )
    gates.loc[len(gates)] = {
        "gate": "overall_technical",
        "passed": bool(gates["passed"].all()),
        "value": int(gates["passed"].sum()),
        "required": len(gates),
    }
    return gates


def run_analysis(
    *,
    event_metrics_csv: str | Path,
    frozen_events_csv: str | Path,
    posterior_bins_csv: str | Path,
    route_segments_csv: str | Path,
    route_points_csv: str | Path,
    eligibility_csv: str | Path,
    output_dir: str | Path,
    maximum_start_distance_cm: float = 30.0,
    permutations: int = 2000,
    bootstraps: int = 2000,
    seed: int = 20260804,
) -> dict[str, Path]:
    inputs = {
        "event_metrics_csv": Path(event_metrics_csv),
        "frozen_events_csv": Path(frozen_events_csv),
        "posterior_bins_csv": Path(posterior_bins_csv),
        "route_segments_csv": Path(route_segments_csv),
        "route_points_csv": Path(route_points_csv),
        "eligibility_csv": Path(eligibility_csv),
    }
    tables = {name: pd.read_csv(path) for name, path in inputs.items()}
    events, pause_summary = build_context_event_table(
        tables["event_metrics_csv"],
        tables["frozen_events_csv"],
        tables["posterior_bins_csv"],
        tables["route_segments_csv"],
        tables["route_points_csv"],
        tables["eligibility_csv"],
        maximum_start_distance_cm=float(maximum_start_distance_cm),
    )
    tests, by_rat, loo, nulls, pause_effects = run_context_hypotheses(
        events,
        permutations=int(permutations),
        bootstraps=int(bootstraps),
        seed=int(seed),
    )
    pauses = pause_summary.merge(pause_effects, on=["pause_id", "session", "rat"], how="left")
    gates = build_gates(events, pauses, tests)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    outputs = {
        EVENT_OUTPUT: events,
        PAUSE_OUTPUT: pauses,
        TEST_OUTPUT: tests,
        BY_RAT_OUTPUT: by_rat,
        LOO_OUTPUT: loo,
        NULL_OUTPUT: nulls,
        GATE_OUTPUT: gates,
    }
    paths: dict[str, Path] = {}
    for name, table in outputs.items():
        path = output / name
        table.to_csv(path, index=False)
        paths[name] = path
    technical = bool(gates.loc[gates["gate"].eq("overall_technical"), "passed"].iloc[0])
    report_lines = [
        "# Pfeiffer/Foster replay context hypotheses",
        "",
        f"Technical status: **{'pass' if technical else 'fail'}**.",
        "Campaign-wide FDR is intentionally deferred until H1-H10 are assembled.",
        "",
    ]
    for row in tests.itertuples(index=False):
        report_lines.append(
            f"- {row.hypothesis} `{row.test}` ({row.role}): estimate {row.estimate:+.3f}, "
            f"rat-bootstrap CI [{row.rat_bootstrap_ci_low:+.3f}, "
            f"{row.rat_bootstrap_ci_high:+.3f}], permutation p={row.permutation_p_value:.4f}, "
            f"status `{row.status_before_campaign_fdr}`."
        )
    report_path = output / REPORT_OUTPUT
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    paths[REPORT_OUTPUT] = report_path
    manifest = {
        "analysis": "pf_replay_context_hypotheses_h1_h2_h3_h4_h10",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "primary_model_axis": "logZ_exact_sparse_momentum_minus_logZ_first_order_imm",
        "represented_path_estimator": "emission_only_posterior_mean",
        "maximum_start_distance_cm": float(maximum_start_distance_cm),
        "permutations": int(permutations),
        "bootstraps": int(bootstraps),
        "seed": int(seed),
        "campaign_fdr_applied": False,
        "outputs": {name: str(path) for name, path in paths.items()},
        "provenance": build_script_provenance(input_paths=inputs, cwd=ROOT),
    }
    manifest_path = output / MANIFEST_OUTPUT
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    paths[MANIFEST_OUTPUT] = manifest_path
    return paths


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-metrics", required=True)
    parser.add_argument("--frozen-events", required=True)
    parser.add_argument("--posterior-bins", required=True)
    parser.add_argument("--route-segments", required=True)
    parser.add_argument("--route-points", required=True)
    parser.add_argument("--eligibility", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--maximum-start-distance-cm", type=float, default=30.0)
    parser.add_argument("--permutations", type=int, default=2000)
    parser.add_argument("--bootstraps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260804)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_analysis(
        event_metrics_csv=args.event_metrics,
        frozen_events_csv=args.frozen_events,
        posterior_bins_csv=args.posterior_bins,
        route_segments_csv=args.route_segments,
        route_points_csv=args.route_points,
        eligibility_csv=args.eligibility,
        output_dir=args.output_dir,
        maximum_start_distance_cm=args.maximum_start_distance_cm,
        permutations=args.permutations,
        bootstraps=args.bootstraps,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
