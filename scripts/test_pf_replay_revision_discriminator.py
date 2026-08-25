#!/usr/bin/env python3
"""Audit Pfeiffer/Foster replay as retrospective geometry, planning, or PE.

This module deliberately stops short of calling retrospective path similarity
"smoothing". It provides the geometry and classifiability gates that a
separate filtering-to-smoothing analysis must pass before making that claim.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _provenance import build_script_provenance, file_sha256  # noqa: E402
from compute_replay_commitment_composition_metrics import (  # noqa: E402
    path_fit_distance_cm,
    path_length,
    resample_path,
)

KEYS = ["session", "rat", "event_index"]
PURE_LABELS = ("past_reversed", "future_plan", "pe_disordered", "null_mismatched")
FEATURES = (
    "retrospective_geometry_score",
    "minimum_template_error_over_noise",
    "transition_surprise_nats",
    "path_roughness",
)

EVENT_OUTPUT = "pf_replay_revision_event_scores.csv"
RAT_OUTPUT = "pf_replay_revision_by_rat.csv"
PRIMARY_OUTPUT = "pf_replay_revision_primary.csv"
PRIMARY_NULL_OUTPUT = "pf_replay_revision_circular_null.csv"
INJECTION_OUTPUT = "pf_replay_revision_injections.csv"
RECOVERY_OUTPUT = "pf_replay_revision_recovery_folds.csv"
CONFUSION_OUTPUT = "pf_replay_revision_recovery_confusion.csv"
REAL_LABEL_OUTPUT = "pf_replay_revision_real_labels.csv"
PE_EVENT_OUTPUT = "pf_replay_revision_pe_events.csv"
PE_NULL_OUTPUT = "pf_replay_revision_pe_time_order_null.csv"
PE_SUMMARY_OUTPUT = "pf_replay_revision_pe_summary.csv"
GATE_OUTPUT = "pf_replay_revision_gates.csv"
REPORT_OUTPUT = "pf_replay_revision_report.md"
MANIFEST_OUTPUT = "pf_replay_revision_manifest.json"


def path_from_json(value: object) -> np.ndarray:
    """Parse a two-dimensional path, returning an empty path on invalid input."""

    try:
        path = np.asarray(json.loads(str(value)), dtype=float)
    except (TypeError, ValueError, json.JSONDecodeError):
        return np.empty((0, 2), dtype=float)
    if path.ndim != 2 or path.shape[1:] != (2,):
        return np.empty((0, 2), dtype=float)
    path = path[np.isfinite(path).all(axis=1)]
    return path if len(path) >= 2 and path_length(path) > 1e-9 else np.empty((0, 2), dtype=float)


def retrospective_geometry_score(
    decoded: np.ndarray,
    past_reversed: np.ndarray,
    future_plan: np.ndarray,
) -> tuple[float, float, float]:
    """Return past error, future error, and a bounded paired geometry score."""

    try:
        past_error = path_fit_distance_cm(decoded, past_reversed)
        future_error = path_fit_distance_cm(decoded, future_plan)
    except ValueError:
        return np.nan, np.nan, np.nan
    denominator = past_error + future_error
    score = (future_error - past_error) / denominator if denominator > 1e-12 else 0.0
    return float(past_error), float(future_error), float(score)


def score_real_events(events: pd.DataFrame) -> pd.DataFrame:
    """Add paired retrospective/future geometry metrics and exclusions."""

    required = {
        *KEYS,
        "event_peak_s",
        "event_route_relation",
        "excluded_cv_fold",
        "emission_path_xy_json",
        "past_template_xy_json",
        "future_template_xy_json",
    }
    missing = sorted(required.difference(events.columns))
    if missing:
        raise ValueError(f"event table is missing required columns: {missing}")
    rows: list[dict[str, object]] = []
    for row in events.itertuples(index=False):
        values = row._asdict()
        decoded = path_from_json(values["emission_path_xy_json"])
        past = path_from_json(values["past_template_xy_json"])
        future = path_from_json(values["future_template_xy_json"])
        past_error, future_error, score = retrospective_geometry_score(decoded, past, future)
        missing_parts = [
            name
            for name, path in (("decoded", decoded), ("past_reversed", past), ("future", future))
            if len(path) < 2
        ]
        rows.append(
            {
                **{key: values[key] for key in KEYS},
                "retrospective_path_error_cm": past_error,
                "future_path_error_cm": future_error,
                "retrospective_geometry_score": score,
                "paired_geometry_eligible": bool(np.isfinite(score)),
                "paired_geometry_exclusion": "|".join(missing_parts),
                "decoded_path_points": int(len(decoded)),
                "past_template_points": int(len(past)),
                "future_template_points": int(len(future)),
            }
        )
    result = events.merge(pd.DataFrame(rows), on=KEYS, how="left", validate="one_to_one")
    strata = ["session", "event_route_relation"]
    eligible = result["paired_geometry_eligible"]
    counts = result.loc[eligible].groupby(strata, dropna=False)["event_index"].transform("size")
    result["circular_null_eligible"] = False
    result.loc[counts.index, "circular_null_eligible"] = counts.ge(2)
    result.loc[
        result["paired_geometry_eligible"] & ~result["circular_null_eligible"],
        "paired_geometry_exclusion",
    ] = "singleton_session_relation_stratum"
    return result


def equal_animal_mean(frame: pd.DataFrame, column: str) -> float:
    """Mean of within-animal means, giving every animal equal weight."""

    selected = frame[["rat", column]].copy()
    selected[column] = pd.to_numeric(selected[column], errors="coerce")
    means = selected.dropna().groupby("rat", sort=True)[column].mean()
    return float(means.mean()) if len(means) else np.nan


def hierarchical_equal_animal_bootstrap(
    frame: pd.DataFrame,
    column: str,
    *,
    replicates: int,
    seed: int,
) -> tuple[float, float, np.ndarray]:
    """Bootstrap rats, sessions within rats, and events within sessions."""

    selected = frame[["rat", "session", column]].dropna().copy()
    rats = sorted(selected["rat"].astype(str).unique())
    if len(rats) < 2:
        return np.nan, np.nan, np.empty(0, dtype=float)
    rng = np.random.default_rng(seed)
    draws = np.empty(int(replicates), dtype=float)
    for replicate in range(int(replicates)):
        rat_draw = rng.choice(rats, size=len(rats), replace=True)
        sampled_rat_means: list[float] = []
        for rat in rat_draw:
            animal = selected[selected["rat"].astype(str).eq(str(rat))]
            sessions = sorted(animal["session"].astype(str).unique())
            session_draw = rng.choice(sessions, size=len(sessions), replace=True)
            sampled_events: list[float] = []
            for session in session_draw:
                values = animal.loc[
                    animal["session"].astype(str).eq(str(session)), column
                ].to_numpy(dtype=float)
                sampled_events.extend(
                    rng.choice(values, size=len(values), replace=True).tolist()
                )
            sampled_rat_means.append(float(np.mean(sampled_events)))
        draws[replicate] = float(np.mean(sampled_rat_means))
    return (
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
        draws,
    )


def restricted_circular_template_null(
    events: pd.DataFrame,
    *,
    permutations: int,
    seed: int,
) -> pd.DataFrame:
    """Shift paired templates within session/relation using nonzero offsets."""

    cohort = events[events["circular_null_eligible"]].copy()
    groups = [
        group.sort_values(["event_peak_s", "event_index"]).index.to_numpy()
        for _, group in cohort.groupby(["session", "event_route_relation"], sort=True, dropna=False)
    ]
    if not groups or any(len(group) < 2 for group in groups):
        return pd.DataFrame(columns=["replicate", "equal_animal_mean", "offsets_json"])
    decoded = {index: path_from_json(cohort.loc[index, "emission_path_xy_json"]) for index in cohort.index}
    past = {index: path_from_json(cohort.loc[index, "past_template_xy_json"]) for index in cohort.index}
    future = {index: path_from_json(cohort.loc[index, "future_template_xy_json"]) for index in cohort.index}
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for replicate in range(int(permutations)):
        shifted = cohort.copy()
        shifted["retrospective_geometry_score"] = np.nan
        offsets: list[int] = []
        for indices in groups:
            offset = int(rng.integers(1, len(indices)))
            offsets.append(offset)
            template_indices = np.roll(indices, offset)
            for target, source in zip(indices, template_indices, strict=True):
                _, _, value = retrospective_geometry_score(
                    decoded[target],
                    past[source],
                    future[source],
                )
                shifted.loc[target, "retrospective_geometry_score"] = value
        rows.append(
            {
                "replicate": replicate,
                "equal_animal_mean": equal_animal_mean(
                    shifted, "retrospective_geometry_score"
                ),
                "offsets_json": json.dumps(offsets, separators=(",", ":")),
                "null_control": "nonzero_circular_paired_template_shift_within_session_and_relation",
            }
        )
    return pd.DataFrame(rows)


def primary_geometry_analysis(
    events: pd.DataFrame,
    *,
    permutations: int,
    bootstraps: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run equal-animal inference for the paired geometry score."""

    cohort = events[events["circular_null_eligible"]].copy()
    estimate = equal_animal_mean(cohort, "retrospective_geometry_score")
    low, high, draws = hierarchical_equal_animal_bootstrap(
        cohort,
        "retrospective_geometry_score",
        replicates=bootstraps,
        seed=seed,
    )
    null = restricted_circular_template_null(
        cohort,
        permutations=permutations,
        seed=seed + 1,
    )
    values = null["equal_animal_mean"].to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    p_retrospective = (
        float((1 + np.sum(values >= estimate)) / (1 + len(values)))
        if np.isfinite(estimate) and len(values)
        else np.nan
    )
    p_future = (
        float((1 + np.sum(values <= estimate)) / (1 + len(values)))
        if np.isfinite(estimate) and len(values)
        else np.nan
    )
    centered = estimate - float(np.mean(values)) if len(values) else np.nan
    p_two_sided = (
        float(
            (1 + np.sum(np.abs(values - np.mean(values)) >= abs(centered)))
            / (1 + len(values))
        )
        if np.isfinite(centered) and len(values)
        else np.nan
    )
    if np.isfinite(low) and low > 0.0 and p_retrospective <= 0.05:
        direction = "retrospective_geometry"
    elif np.isfinite(high) and high < 0.0 and p_future <= 0.05:
        direction = "future_geometry"
    else:
        direction = "inconclusive"
    primary = pd.DataFrame(
        [
            {
                "analysis": "paired_retrospective_vs_future_geometry",
                "claim_level": "retrospective_content_geometry_only",
                "events": int(len(cohort)),
                "rats": int(cohort["rat"].nunique()),
                "sessions": int(cohort["session"].nunique()),
                "equal_animal_mean": estimate,
                "hierarchical_bootstrap_ci_low": low,
                "hierarchical_bootstrap_ci_high": high,
                "bootstrap_replicates_completed": int(len(draws)),
                "circular_null_replicates_completed": int(len(values)),
                "retrospective_one_sided_p": p_retrospective,
                "future_one_sided_p": p_future,
                "two_sided_p": p_two_sided,
                "direction_before_recovery_gate": direction,
                "null_control": "nonzero_circular_paired_template_shift_within_session_and_relation",
            }
        ]
    )
    rat_rows: list[dict[str, object]] = []
    for rat, group in cohort.groupby("rat", sort=True):
        values_rat = group["retrospective_geometry_score"].to_numpy(dtype=float)
        rat_rows.append(
            {
                "rat": str(rat),
                "events": int(len(group)),
                "sessions": int(group["session"].nunique()),
                "mean_retrospective_geometry_score": float(np.mean(values_rat)),
                "median_retrospective_geometry_score": float(np.median(values_rat)),
                "fraction_positive": float(np.mean(values_rat > 0.0)),
            }
        )
    for omitted in sorted(cohort["rat"].astype(str).unique()):
        retained = cohort[~cohort["rat"].astype(str).eq(omitted)]
        rat_rows.append(
            {
                "rat": f"leave_out::{omitted}",
                "events": int(len(retained)),
                "sessions": int(retained["session"].nunique()),
                "mean_retrospective_geometry_score": equal_animal_mean(
                    retained, "retrospective_geometry_score"
                ),
                "median_retrospective_geometry_score": float(
                    retained["retrospective_geometry_score"].median()
                ),
                "fraction_positive": float(
                    np.mean(retained["retrospective_geometry_score"] > 0.0)
                ),
            }
        )
    return primary, pd.DataFrame(rat_rows), null


def _xy_state(path: np.ndarray, model: dict[str, Any]) -> list[tuple[int, int]]:
    origin = np.asarray(model["origin"], dtype=float)
    bins = np.floor((np.asarray(path, dtype=float) - origin) / float(model["bin_cm"])).astype(int)
    states = [tuple(map(int, row)) for row in bins]
    return [state for index, state in enumerate(states) if index == 0 or state != states[index - 1]]


def build_cross_validated_markov_models(
    route_segments: pd.DataFrame,
    route_points: pd.DataFrame,
    *,
    bin_cm: float,
    alpha: float,
) -> dict[tuple[str, int], dict[str, Any]]:
    """Fit finite behavior-only Markov models with each route fold held out."""

    required_routes = {"session", "route_id", "cv_fold"}
    required_points = {"session", "route_id", "point_index", "x_cm", "y_cm"}
    if missing := sorted(required_routes.difference(route_segments.columns)):
        raise ValueError(f"route segments are missing columns: {missing}")
    if missing := sorted(required_points.difference(route_points.columns)):
        raise ValueError(f"route points are missing columns: {missing}")
    if bin_cm <= 0.0 or alpha <= 0.0:
        raise ValueError("bin_cm and alpha must be positive")
    models: dict[tuple[str, int], dict[str, Any]] = {}
    for session, session_routes in route_segments.groupby("session", sort=True):
        session_points = route_points[route_points["session"].astype(str).eq(str(session))]
        origin = np.floor(
            session_points[["x_cm", "y_cm"]].min().to_numpy(dtype=float) / bin_cm
        ) * bin_cm
        folds = sorted(session_routes["cv_fold"].astype(int).unique())
        for held_fold in folds:
            train_ids = set(
                session_routes.loc[
                    ~session_routes["cv_fold"].astype(int).eq(held_fold), "route_id"
                ].astype(str)
            )
            support: set[tuple[int, int]] = set()
            counts: dict[tuple[tuple[int, int], tuple[int, int]], int] = {}
            outgoing: dict[tuple[int, int], int] = {}
            provisional = {"origin": origin, "bin_cm": float(bin_cm)}
            for route_id in sorted(train_ids):
                points = session_points[
                    session_points["route_id"].astype(str).eq(route_id)
                ].sort_values("point_index")
                states = _xy_state(
                    points[["x_cm", "y_cm"]].to_numpy(dtype=float),
                    provisional,
                )
                support.update(states)
                for source, target in zip(states[:-1], states[1:], strict=True):
                    key = (source, target)
                    counts[key] = counts.get(key, 0) + 1
                    outgoing[source] = outgoing.get(source, 0) + 1
            models[(str(session), int(held_fold))] = {
                "origin": origin,
                "bin_cm": float(bin_cm),
                "alpha": float(alpha),
                "support": support,
                "counts": counts,
                "outgoing": outgoing,
                "target_categories": int(len(support) + 1),
                "training_route_count": int(len(train_ids)),
            }
    return models


def finite_transition_surprise(
    path: np.ndarray,
    model: dict[str, Any],
) -> tuple[float, int]:
    """Mean finite surprise with an explicit out-of-support target class."""

    states = _xy_state(path, model)
    if len(states) < 2:
        # A valid path can remain in one 10-cm state after consecutive-state
        # collapse. It then contains zero evaluated transitions and therefore
        # contributes zero mean transition surprise instead of poisoning
        # classifier calibration with NaN.
        return 0.0, 0
    support = model["support"]
    categories = max(1, int(model["target_categories"]))
    alpha = float(model["alpha"])
    values: list[float] = []
    for source, target in zip(states[:-1], states[1:], strict=True):
        if source not in support:
            probability = 1.0 / categories
        else:
            denominator = float(model["outgoing"].get(source, 0)) + alpha * categories
            count = model["counts"].get((source, target), 0) if target in support else 0
            probability = (float(count) + alpha) / denominator
        values.append(float(-np.log(max(probability, np.finfo(float).tiny))))
    return float(np.mean(values)), int(len(values))


def nontrivial_time_permutation(path: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Permute time while preserving the decoded point set and first point."""

    points = np.asarray(path, dtype=float)
    if len(points) < 3:
        return points[::-1].copy()
    tail = np.arange(1, len(points))
    permuted = rng.permutation(tail)
    if np.array_equal(permuted, tail):
        permuted = np.roll(tail, 1)
    return points[np.concatenate([[0], permuted])]


def compute_pe_diagnostic(
    events: pd.DataFrame,
    models: dict[tuple[str, int], dict[str, Any]],
    *,
    permutations: int,
    bootstraps: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compare ordered replay surprise with a fixed-point-set time-order null."""

    rng = np.random.default_rng(seed)
    event_rows: list[dict[str, object]] = []
    null_rows: list[dict[str, object]] = []
    for row in events.itertuples(index=False):
        values = row._asdict()
        path = path_from_json(values["emission_path_xy_json"])
        model = models.get((str(values["session"]), int(values["excluded_cv_fold"])))
        if model is None or len(path) < 2:
            event_rows.append(
                {
                    **{key: values[key] for key in KEYS},
                    "pe_diagnostic_eligible": False,
                    "pe_exclusion": "missing_path_or_cv_model",
                }
            )
            continue
        ordered, pairs = finite_transition_surprise(path, model)
        reversed_value, _ = finite_transition_surprise(path[::-1], model)
        shuffled_values: list[float] = []
        for replicate in range(int(permutations)):
            value, shuffled_pairs = finite_transition_surprise(
                nontrivial_time_permutation(path, rng),
                model,
            )
            shuffled_values.append(value)
            null_rows.append(
                {
                    **{key: values[key] for key in KEYS},
                    "replicate": replicate,
                    "permuted_transition_surprise_nats": value,
                    "transition_pairs": shuffled_pairs,
                    "null_control": "within_event_nontrivial_time_permutation_fixed_point_set",
                }
            )
        null_mean = float(np.nanmean(shuffled_values))
        event_rows.append(
            {
                **{key: values[key] for key in KEYS},
                "pe_diagnostic_eligible": bool(np.isfinite(ordered) and np.isfinite(null_mean)),
                "pe_exclusion": "",
                "ordered_transition_surprise_nats": ordered,
                "reversed_transition_surprise_nats": reversed_value,
                "time_permuted_mean_surprise_nats": null_mean,
                "ordered_minus_time_permuted_surprise_nats": ordered - null_mean,
                "ordered_minus_reversed_surprise_nats": ordered - reversed_value,
                "transition_pairs": pairs,
                "no_transition_after_binning": pairs == 0,
                "legacy_transition_surprise_nats": values.get("transition_surprise_nats", np.nan),
                "legacy_transition_surprise_pairs": values.get("transition_surprise_pairs", 0),
            }
        )
    pe_events = pd.DataFrame(event_rows)
    pe_null = pd.DataFrame(null_rows)
    eligible = pe_events[pe_events["pe_diagnostic_eligible"]].copy()
    estimate = equal_animal_mean(eligible, "ordered_minus_time_permuted_surprise_nats")
    low, high, draws = hierarchical_equal_animal_bootstrap(
        eligible,
        "ordered_minus_time_permuted_surprise_nats",
        replicates=bootstraps,
        seed=seed + 1,
    )
    null_aggregate = (
        pe_null.groupby("replicate", sort=True)
        .apply(
            lambda group: equal_animal_mean(
                group.rename(columns={"permuted_transition_surprise_nats": "null_value"}),
                "null_value",
            ),
            include_groups=False,
        )
        .to_numpy(dtype=float)
        if len(pe_null)
        else np.empty(0, dtype=float)
    )
    null_aggregate = null_aggregate[np.isfinite(null_aggregate)]
    ordered_aggregate = equal_animal_mean(
        eligible.rename(columns={"ordered_transition_surprise_nats": "ordered"}),
        "ordered",
    )
    coherence_p = (
        float((1 + np.sum(null_aggregate <= ordered_aggregate)) / (1 + len(null_aggregate)))
        if len(null_aggregate) and np.isfinite(ordered_aggregate)
        else np.nan
    )
    high_surprise_p = (
        float((1 + np.sum(null_aggregate >= ordered_aggregate)) / (1 + len(null_aggregate)))
        if len(null_aggregate) and np.isfinite(ordered_aggregate)
        else np.nan
    )
    summary = pd.DataFrame(
        [
            {
                "analysis": "finite_cross_validated_transition_surprise",
                "claim_level": "prediction_error_time_order_diagnostic_only",
                "events": int(len(eligible)),
                "rats": int(eligible["rat"].nunique()),
                "sessions": int(eligible["session"].nunique()),
                "equal_animal_ordered_minus_permuted": estimate,
                "hierarchical_bootstrap_ci_low": low,
                "hierarchical_bootstrap_ci_high": high,
                "bootstrap_replicates_completed": int(len(draws)),
                "ordered_equal_animal_surprise": ordered_aggregate,
                "time_permutation_p_lower": coherence_p,
                "ordered_lower_than_permuted_coherence_p": coherence_p,
                "ordered_higher_than_permuted_pe_surrogate_p": high_surprise_p,
                "finite_markov_model": True,
                "zero_transition_events": int(
                    eligible.get("no_transition_after_binning", pd.Series(dtype=bool)).sum()
                ),
                "null_control": "within_event_nontrivial_time_permutation_fixed_point_set",
            }
        ]
    )
    return pe_events, pe_null, summary


def correlated_noisy_path(
    base_path: np.ndarray,
    *,
    n_points: int,
    radial_rms_cm: float,
    rho: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Inject anchored AR(1) decoding noise at a fixed radial RMS."""

    base = resample_path(base_path, n_points=max(3, int(n_points)))
    noise = np.empty_like(base)
    noise[0] = 0.0
    innovation_scale = float(np.sqrt(max(0.0, 1.0 - rho * rho)))
    for index in range(1, len(base)):
        noise[index] = rho * noise[index - 1] + innovation_scale * rng.normal(size=2)
    taper = np.minimum(1.0, np.arange(len(base), dtype=float) / 3.0)[:, None]
    noise *= taper
    rms = float(np.sqrt(np.mean(np.sum(noise * noise, axis=1))))
    if rms > 0.0:
        noise *= float(radial_rms_cm) / rms
    return base + noise


def disordered_path(base_path: np.ndarray, *, n_points: int, rng: np.random.Generator) -> np.ndarray:
    """Destroy global order while retaining actual path segments and points."""

    path = resample_path(base_path, n_points=max(5, int(n_points)))
    tail = path[1:]
    blocks = [block for block in np.array_split(tail, min(4, len(tail))) if len(block)]
    order = rng.permutation(len(blocks))
    if np.array_equal(order, np.arange(len(blocks))):
        order = np.roll(order, 1)
    permuted = [blocks[int(index)][::-1] if rng.random() < 0.5 else blocks[int(index)] for index in order]
    return np.vstack([path[:1], *permuted])


def path_roughness(path: np.ndarray) -> float:
    """Dimensionless second-difference roughness."""

    steps = np.diff(np.asarray(path, dtype=float), axis=0)
    if len(steps) < 2:
        return 0.0
    return float(
        np.linalg.norm(np.diff(steps, axis=0), axis=1).sum()
        / (np.linalg.norm(steps, axis=1).sum() + 1e-12)
    )


def path_features(
    path: np.ndarray,
    *,
    past: np.ndarray,
    future: np.ndarray,
    noise_cm: float,
    markov_model: dict[str, Any],
) -> dict[str, float]:
    """Features used by the leakage-safe candidate classifier."""

    past_error, future_error, score = retrospective_geometry_score(path, past, future)
    surprise, transition_pairs = finite_transition_surprise(path, markov_model)
    return {
        "retrospective_geometry_score": score,
        "minimum_template_error_over_noise": min(past_error, future_error) / max(float(noise_cm), 1e-12),
        "transition_surprise_nats": surprise,
        "transition_pairs": float(transition_pairs),
        "path_roughness": path_roughness(path),
    }


def generate_actual_geometry_injections(
    events: pd.DataFrame,
    models: dict[tuple[str, int], dict[str, Any]],
    *,
    injections_per_candidate: int,
    seed: int,
    rho: float = 0.75,
    minimum_noise_cm: float = 2.5,
    maximum_noise_cm: float = 50.0,
) -> pd.DataFrame:
    """Generate pure candidates and mixtures on actual event geometry."""

    cohort = events[events["circular_null_eligible"]].copy()
    decoder_error = pd.to_numeric(cohort.get("run_decoder_error_cm"), errors="coerce")
    fallback = float(decoder_error.median()) if decoder_error.notna().any() else 15.0
    mismatch: dict[int, int] = {}
    for _, group in cohort.groupby(["session", "event_route_relation"], sort=True, dropna=False):
        indices = group.sort_values(["event_peak_s", "event_index"]).index.to_numpy()
        for target, source in zip(indices, np.roll(indices, 1), strict=True):
            mismatch[int(target)] = int(source)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for index, row in cohort.iterrows():
        decoded = path_from_json(row["emission_path_xy_json"])
        past = path_from_json(row["past_template_xy_json"])
        future = path_from_json(row["future_template_xy_json"])
        other = cohort.loc[mismatch[int(index)]]
        other_past = path_from_json(other["past_template_xy_json"])
        other_future = path_from_json(other["future_template_xy_json"])
        model = models.get((str(row["session"]), int(row["excluded_cv_fold"])))
        if model is None:
            continue
        raw_noise = pd.to_numeric(pd.Series([row.get("run_decoder_error_cm", np.nan)]), errors="coerce").iloc[0]
        noise_cm = float(raw_noise) if np.isfinite(raw_noise) else fallback
        noise_cm = float(np.clip(noise_cm, minimum_noise_cm, maximum_noise_cm))
        n_points = int(np.clip(len(decoded), 8, 41))
        for replicate in range(int(injections_per_candidate)):
            pe_base = past if replicate % 2 == 0 else future
            null_base = other_past if replicate % 2 == 0 else other_future
            past_resampled = resample_path(past, n_points=n_points)
            future_resampled = resample_path(future, n_points=n_points)
            bases = {
                "past_reversed": past,
                "future_plan": future,
                "pe_disordered": disordered_path(pe_base, n_points=n_points, rng=rng),
                "null_mismatched": null_base,
                "mixture_50_50": 0.5 * past_resampled + 0.5 * future_resampled,
            }
            for label, base in bases.items():
                injected = correlated_noisy_path(
                    base,
                    n_points=n_points,
                    radial_rms_cm=noise_cm,
                    rho=float(rho),
                    rng=rng,
                )
                rows.append(
                    {
                        **{key: row[key] for key in KEYS},
                        "replicate": replicate,
                        "sample_kind": "mixture" if label == "mixture_50_50" else "pure",
                        "true_label": label,
                        "noise_cm": noise_cm,
                        "n_points": n_points,
                        "rho": float(rho),
                        **path_features(
                            injected,
                            past=past,
                            future=future,
                            noise_cm=noise_cm,
                            markov_model=model,
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _fit_centroid_model(train: pd.DataFrame) -> dict[str, Any]:
    values = train.loc[:, FEATURES].to_numpy(dtype=float)
    if values.ndim != 2 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("classifier training features must be nonempty and finite")
    labels = train["true_label"].astype(str).to_numpy()
    missing_labels = sorted(set(PURE_LABELS).difference(labels))
    if missing_labels:
        raise ValueError(f"classifier training data are missing pure classes: {missing_labels}")
    center = np.mean(values, axis=0)
    raw_scale = np.std(values, axis=0)
    if np.all(raw_scale <= 1e-12):
        raise ValueError("all classifier training features are degenerate")
    scale = raw_scale.copy()
    scale[scale <= 1e-12] = 1.0
    standardized = (values - center) / scale
    centroids = {
        label: np.mean(standardized[labels == label], axis=0)
        for label in PURE_LABELS
    }
    if any(not np.isfinite(centroid).all() for centroid in centroids.values()):
        raise ValueError("classifier centroids must be finite")
    return {"center": center, "scale": scale, "centroids": centroids}


def _centroid_predictions(
    frame: pd.DataFrame,
    model: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = frame.loc[:, FEATURES].to_numpy(dtype=float)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("classifier prediction features must be finite")
    if (
        not np.isfinite(model["center"]).all()
        or not np.isfinite(model["scale"]).all()
        or np.any(np.asarray(model["scale"]) <= 0.0)
    ):
        raise ValueError("classifier standardization parameters must be finite and positive")
    if any(
        label not in model["centroids"]
        or not np.isfinite(model["centroids"][label]).all()
        for label in PURE_LABELS
    ):
        raise ValueError("classifier model is missing finite pure-class centroids")
    standardized = (values - model["center"]) / model["scale"]
    distances = np.column_stack(
        [
            np.sum((standardized - model["centroids"][label]) ** 2, axis=1)
            for label in PURE_LABELS
        ]
    )
    order = np.argsort(distances, axis=1)
    labels = np.asarray(PURE_LABELS, dtype=object)[order[:, 0]]
    nearest = distances[np.arange(len(frame)), order[:, 0]]
    margin = distances[np.arange(len(frame)), order[:, 1]] - nearest
    return labels, nearest, margin


def _apply_thresholds(
    labels: np.ndarray,
    nearest: np.ndarray,
    margin: np.ndarray,
    *,
    maximum_distance: float,
    minimum_margin: float,
) -> np.ndarray:
    result = labels.astype(object).copy()
    abstain = (nearest > float(maximum_distance)) | (margin < float(minimum_margin))
    result[abstain] = "abstain"
    return result


def _balanced_accuracy(truth: Iterable[object], prediction: Iterable[object]) -> float:
    true = np.asarray(list(truth), dtype=object)
    pred = np.asarray(list(prediction), dtype=object)
    recalls = [
        float(np.mean(pred[true == label] == label))
        for label in PURE_LABELS
        if np.any(true == label)
    ]
    return float(np.mean(recalls)) if recalls else np.nan


def calibrate_abstention(
    train_pure: pd.DataFrame,
    train_mixture: pd.DataFrame,
    model: dict[str, Any],
    *,
    target_accuracy: float,
    minimum_coverage: float,
    minimum_mixture_abstention: float,
) -> tuple[float, float, dict[str, float]]:
    """Select thresholds using training groups only."""

    pure_labels, pure_distance, pure_margin = _centroid_predictions(train_pure, model)
    mix_labels, mix_distance, mix_margin = _centroid_predictions(train_mixture, model)
    distance_grid = np.unique(np.quantile(pure_distance, [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 1.0]))
    margin_grid = np.unique(np.quantile(pure_margin, [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]))
    feasible: list[tuple[float, float, float, float, float]] = []
    truth = train_pure["true_label"].astype(str).to_numpy()
    for maximum_distance in distance_grid:
        for minimum_margin_value in margin_grid:
            pure_prediction = _apply_thresholds(
                pure_labels,
                pure_distance,
                pure_margin,
                maximum_distance=float(maximum_distance),
                minimum_margin=float(minimum_margin_value),
            )
            mix_prediction = _apply_thresholds(
                mix_labels,
                mix_distance,
                mix_margin,
                maximum_distance=float(maximum_distance),
                minimum_margin=float(minimum_margin_value),
            )
            retained = pure_prediction != "abstain"
            coverage = float(np.mean(retained))
            retained_accuracy = float(np.mean(pure_prediction[retained] == truth[retained])) if np.any(retained) else 0.0
            mixture_abstention = float(np.mean(mix_prediction == "abstain"))
            if (
                retained_accuracy >= target_accuracy
                and coverage >= minimum_coverage
                and mixture_abstention >= minimum_mixture_abstention
            ):
                feasible.append(
                    (
                        coverage,
                        retained_accuracy,
                        mixture_abstention,
                        float(maximum_distance),
                        float(minimum_margin_value),
                    )
                )
    if not feasible:
        # These inverse bounds force every candidate to abstain. The opposite
        # infinities would accidentally retain every row after failed calibration.
        return np.inf, -np.inf, {
            "training_retained_accuracy": np.nan,
            "training_pure_coverage": 0.0,
            "training_mixture_abstention": 1.0,
            "calibration_feasible": 0.0,
        }
    coverage, accuracy, mixture_abstention, maximum_distance, minimum_margin_value = max(
        feasible, key=lambda row: (row[0], row[1], row[2])
    )
    return minimum_margin_value, maximum_distance, {
        "training_retained_accuracy": accuracy,
        "training_pure_coverage": coverage,
        "training_mixture_abstention": mixture_abstention,
        "calibration_feasible": 1.0,
    }


def cross_validated_recovery(
    injections: pd.DataFrame,
    *,
    target_accuracy: float,
    minimum_coverage: float,
    minimum_mixture_abstention: float,
    schemes: Sequence[str] = ("rat", "session"),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[tuple[str, str], dict[str, Any]]]:
    """LO-animal and LO-session recovery with train-only calibration."""

    pure = injections[injections["sample_kind"].eq("pure")].copy()
    mixture = injections[injections["sample_kind"].eq("mixture")].copy()
    fold_rows: list[dict[str, object]] = []
    confusion_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    fitted: dict[tuple[str, str], dict[str, Any]] = {}
    for scheme in schemes:
        for held_out in sorted(injections[scheme].astype(str).unique()):
            train_pure = pure[~pure[scheme].astype(str).eq(held_out)].copy()
            test_pure = pure[pure[scheme].astype(str).eq(held_out)].copy()
            train_mix = mixture[~mixture[scheme].astype(str).eq(held_out)].copy()
            test_mix = mixture[mixture[scheme].astype(str).eq(held_out)].copy()
            if train_pure.empty or test_pure.empty or train_mix.empty or test_mix.empty:
                continue
            model = _fit_centroid_model(train_pure)
            minimum_margin_value, maximum_distance, calibration = calibrate_abstention(
                train_pure,
                train_mix,
                model,
                target_accuracy=target_accuracy,
                minimum_coverage=minimum_coverage,
                minimum_mixture_abstention=minimum_mixture_abstention,
            )
            labels, distance, margin = _centroid_predictions(test_pure, model)
            prediction = _apply_thresholds(
                labels,
                distance,
                margin,
                maximum_distance=maximum_distance,
                minimum_margin=minimum_margin_value,
            )
            mix_labels, mix_distance, mix_margin = _centroid_predictions(test_mix, model)
            mix_prediction = _apply_thresholds(
                mix_labels,
                mix_distance,
                mix_margin,
                maximum_distance=maximum_distance,
                minimum_margin=minimum_margin_value,
            )
            truth = test_pure["true_label"].astype(str).to_numpy()
            retained = prediction != "abstain"
            scheme_name = "leave_one_animal_out" if scheme == "rat" else "leave_one_session_out"
            fold_rows.append(
                {
                    "scheme": scheme_name,
                    "group_column": scheme,
                    "held_out_group": held_out,
                    "training_groups_json": json.dumps(sorted(train_pure[scheme].astype(str).unique().tolist()), separators=(",", ":")),
                    "pure_samples": int(len(test_pure)),
                    "mixture_samples": int(len(test_mix)),
                    "balanced_accuracy_including_abstention": _balanced_accuracy(truth, prediction),
                    "retained_accuracy": float(np.mean(prediction[retained] == truth[retained])) if np.any(retained) else np.nan,
                    "pure_coverage": float(np.mean(retained)),
                    "mixture_abstention": float(np.mean(mix_prediction == "abstain")),
                    "minimum_margin": minimum_margin_value,
                    "maximum_distance": maximum_distance,
                    **calibration,
                }
            )
            for true_label in PURE_LABELS:
                for predicted_label in (*PURE_LABELS, "abstain"):
                    confusion_rows.append(
                        {
                            "scheme": scheme_name,
                            "held_out_group": held_out,
                            "true_label": true_label,
                            "predicted_label": predicted_label,
                            "count": int(np.sum((truth == true_label) & (prediction == predicted_label))),
                        }
                    )
            prediction_frame = test_pure.loc[:, [*KEYS, "replicate", "true_label"]].copy()
            prediction_frame["scheme"] = scheme_name
            prediction_frame["held_out_group"] = held_out
            prediction_frame["predicted_label"] = prediction
            prediction_rows.extend(prediction_frame.to_dict(orient="records"))
            fitted[(scheme, held_out)] = {
                "model": model,
                "minimum_margin": minimum_margin_value,
                "maximum_distance": maximum_distance,
                "training_groups": sorted(train_pure[scheme].astype(str).unique().tolist()),
            }
    return pd.DataFrame(fold_rows), pd.DataFrame(confusion_rows), pd.DataFrame(prediction_rows), fitted


def classify_real_events(
    events: pd.DataFrame,
    pe_events: pd.DataFrame,
    models: dict[tuple[str, int], dict[str, Any]],
    fitted: dict[tuple[str, str], dict[str, Any]],
) -> pd.DataFrame:
    """Apply only out-of-animal and out-of-session fitted models."""

    rows: list[dict[str, object]] = []
    decoder_error = pd.to_numeric(events.get("run_decoder_error_cm"), errors="coerce")
    fallback = float(decoder_error.median()) if decoder_error.notna().any() else 15.0
    pe_lookup = pe_events.set_index(KEYS)
    for _, row in events[events["circular_null_eligible"]].iterrows():
        path = path_from_json(row["emission_path_xy_json"])
        past = path_from_json(row["past_template_xy_json"])
        future = path_from_json(row["future_template_xy_json"])
        markov_model = models.get((str(row["session"]), int(row["excluded_cv_fold"])))
        if markov_model is None:
            continue
        raw_noise = pd.to_numeric(pd.Series([row.get("run_decoder_error_cm", np.nan)]), errors="coerce").iloc[0]
        noise_cm = float(raw_noise) if np.isfinite(raw_noise) else fallback
        features = path_features(
            path,
            past=past,
            future=future,
            noise_cm=float(np.clip(noise_cm, 2.5, 50.0)),
            markov_model=markov_model,
        )
        feature_frame = pd.DataFrame([features])
        result: dict[str, object] = {**{key: row[key] for key in KEYS}, **features}
        for scheme, group_value, output_name in (
            ("rat", str(row["rat"]), "loao_raw_candidate_label"),
            ("session", str(row["session"]), "loso_raw_candidate_label"),
        ):
            fold = fitted.get((scheme, group_value))
            if fold is None:
                result[output_name] = "abstain"
                continue
            labels, distance, margin = _centroid_predictions(feature_frame, fold["model"])
            result[output_name] = _apply_thresholds(
                labels,
                distance,
                margin,
                maximum_distance=fold["maximum_distance"],
                minimum_margin=fold["minimum_margin"],
            )[0]
        key = tuple(row[item] for item in KEYS)
        result["ordered_minus_time_permuted_surprise_nats"] = (
            pe_lookup.loc[key, "ordered_minus_time_permuted_surprise_nats"]
            if key in pe_lookup.index
            else np.nan
        )
        rows.append(result)
    return pd.DataFrame(rows)


def adjudication_interpretation(
    *,
    technical_passed: bool,
    candidate_classifiable: bool,
    historical_positive_control_exactly_reproducible: bool,
    direction: str,
) -> str:
    """Separate computational validity from scientific adjudication readiness."""

    if not technical_passed:
        return "technical_failure"
    if not (
        candidate_classifiable
        and historical_positive_control_exactly_reproducible
    ):
        return "technical_nonadjudicative"
    return str(direction)


def build_gates(
    events: pd.DataFrame,
    primary: pd.DataFrame,
    pe_events: pd.DataFrame,
    recovery: pd.DataFrame,
) -> pd.DataFrame:
    """Build technical and candidate-classifiability gates."""

    eligible = events[events["circular_null_eligible"]]
    loao = recovery[recovery["scheme"].eq("leave_one_animal_out")]
    loso = recovery[recovery["scheme"].eq("leave_one_session_out")]
    loao_ba = float(loao["balanced_accuracy_including_abstention"].mean()) if len(loao) else np.nan
    loso_ba = float(loso["balanced_accuracy_including_abstention"].mean()) if len(loso) else np.nan
    loao_min = float(loao["balanced_accuracy_including_abstention"].min()) if len(loao) else np.nan
    mixture = float(pd.concat([loao["mixture_abstention"], loso["mixture_abstention"]]).mean()) if len(loao) and len(loso) else np.nan
    rows = [
        ("paired_geometry_events", len(eligible) >= 100, len(eligible), ">=100"),
        ("paired_geometry_all_animals", eligible["rat"].nunique() == 4, int(eligible["rat"].nunique()), 4),
        ("paired_geometry_all_sessions", eligible["session"].nunique() == 8, int(eligible["session"].nunique()), 8),
        ("primary_finite", primary["equal_animal_mean"].notna().all(), int(primary["equal_animal_mean"].notna().sum()), 1),
        (
            "pe_diagnostic_finite",
            int(pe_events["pe_diagnostic_eligible"].sum()) >= 100
            and np.isfinite(
                pd.to_numeric(
                    pe_events.loc[pe_events["pe_diagnostic_eligible"], "ordered_transition_surprise_nats"],
                    errors="coerce",
                )
            ).all(),
            int(pe_events["pe_diagnostic_eligible"].sum()),
            ">=100 finite",
        ),
        ("loao_all_animals", len(loao) == 4, len(loao), 4),
        ("loso_all_sessions", len(loso) == 8, len(loso), 8),
        ("loao_equal_animal_balanced_accuracy", np.isfinite(loao_ba) and loao_ba >= 0.80, loao_ba, ">=0.80"),
        ("loso_equal_session_balanced_accuracy", np.isfinite(loso_ba) and loso_ba >= 0.80, loso_ba, ">=0.80"),
        ("loao_minimum_animal_balanced_accuracy", np.isfinite(loao_min) and loao_min >= 0.70, loao_min, ">=0.70"),
        ("mixture_abstention", np.isfinite(mixture) and mixture >= 0.50, mixture, ">=0.50"),
        (
            "historical_pf2013_positive_control_exactly_reproducible",
            False,
            False,
            True,
        ),
    ]
    gates = pd.DataFrame(
        {"gate": gate, "passed": bool(passed), "value": value, "required": required}
        for gate, passed, value, required in rows
    )
    technical_names = {
        "paired_geometry_events",
        "paired_geometry_all_animals",
        "paired_geometry_all_sessions",
        "primary_finite",
        "pe_diagnostic_finite",
        "loao_all_animals",
        "loso_all_sessions",
    }
    class_names = {
        "loao_equal_animal_balanced_accuracy",
        "loso_equal_session_balanced_accuracy",
        "loao_minimum_animal_balanced_accuracy",
        "mixture_abstention",
    }
    gates.loc[len(gates)] = {
        "gate": "overall_technical",
        "passed": bool(gates[gates["gate"].isin(technical_names)]["passed"].all()),
        "value": int(gates[gates["gate"].isin(technical_names)]["passed"].sum()),
        "required": len(technical_names),
    }
    gates.loc[len(gates)] = {
        "gate": "candidate_classifiability",
        "passed": bool(gates[gates["gate"].isin(class_names)]["passed"].all()),
        "value": int(gates[gates["gate"].isin(class_names)]["passed"].sum()),
        "required": len(class_names),
    }
    technical_passed = bool(
        gates.loc[gates["gate"].eq("overall_technical"), "passed"].iloc[0]
    )
    classifiable = bool(
        gates.loc[gates["gate"].eq("candidate_classifiability"), "passed"].iloc[0]
    )
    historical_available = bool(
        gates.loc[
            gates["gate"].eq("historical_pf2013_positive_control_exactly_reproducible"),
            "passed",
        ].iloc[0]
    )
    gates.loc[len(gates)] = {
        "gate": "adjudication_ready",
        "passed": technical_passed and classifiable and historical_available,
        "value": int(technical_passed) + int(classifiable) + int(historical_available),
        "required": 3,
    }
    return gates


def run_analysis(
    *,
    events_csv: str | Path,
    route_segments_csv: str | Path,
    route_points_csv: str | Path,
    output_dir: str | Path,
    permutations: int = 2000,
    bootstraps: int = 4000,
    pe_permutations: int = 500,
    injections_per_candidate: int = 200,
    seed: int = 20260825,
    markov_bin_cm: float = 10.0,
    markov_alpha: float = 0.5,
    target_recovery_accuracy: float = 0.80,
    minimum_recovery_coverage: float = 0.50,
    minimum_mixture_abstention: float = 0.50,
) -> dict[str, Path]:
    """Run and freeze the complete retrospective-geometry discriminator."""

    inputs = {
        "events_csv": Path(events_csv),
        "route_segments_csv": Path(route_segments_csv),
        "route_points_csv": Path(route_points_csv),
    }
    events_raw = pd.read_csv(inputs["events_csv"])
    routes = pd.read_csv(inputs["route_segments_csv"])
    points = pd.read_csv(inputs["route_points_csv"])
    events = score_real_events(events_raw)
    primary, by_rat, primary_null = primary_geometry_analysis(
        events,
        permutations=permutations,
        bootstraps=bootstraps,
        seed=seed,
    )
    markov_models = build_cross_validated_markov_models(
        routes,
        points,
        bin_cm=markov_bin_cm,
        alpha=markov_alpha,
    )
    pe_events, pe_null, pe_summary = compute_pe_diagnostic(
        events,
        markov_models,
        permutations=pe_permutations,
        bootstraps=bootstraps,
        seed=seed + 100,
    )
    injections = generate_actual_geometry_injections(
        events,
        markov_models,
        injections_per_candidate=injections_per_candidate,
        seed=seed + 200,
    )
    recovery, confusion, _, fitted = cross_validated_recovery(
        injections,
        target_accuracy=target_recovery_accuracy,
        minimum_coverage=minimum_recovery_coverage,
        minimum_mixture_abstention=minimum_mixture_abstention,
    )
    real_labels = classify_real_events(events, pe_events, markov_models, fitted)
    gates = build_gates(events, primary, pe_events, recovery)
    classifiable = bool(gates.loc[gates["gate"].eq("candidate_classifiability"), "passed"].iloc[0])
    technical = bool(
        gates.loc[gates["gate"].eq("overall_technical"), "passed"].iloc[0]
    )
    adjudication_ready = bool(
        gates.loc[gates["gate"].eq("adjudication_ready"), "passed"].iloc[0]
    )
    real_labels["candidate_label"] = np.where(
        adjudication_ready,
        np.where(
            real_labels["loao_raw_candidate_label"].eq(real_labels["loso_raw_candidate_label"]),
            real_labels["loao_raw_candidate_label"],
            "abstain_cv_disagreement",
        ),
        "abstain_nonidentified",
    )
    events = events.merge(
        real_labels[[*KEYS, "candidate_label"]],
        on=KEYS,
        how="left",
        validate="one_to_one",
    )
    events["candidate_label"] = events["candidate_label"].fillna("abstain_ineligible")
    primary["candidate_classifiability_gate"] = classifiable
    primary["historical_pf2013_positive_control_exactly_reproducible"] = False
    primary["adjudication_ready"] = adjudication_ready
    primary["interpretation"] = adjudication_interpretation(
        technical_passed=technical,
        candidate_classifiable=classifiable,
        historical_positive_control_exactly_reproducible=False,
        direction=str(primary["direction_before_recovery_gate"].iloc[0]),
    )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    tables = {
        EVENT_OUTPUT: events,
        RAT_OUTPUT: by_rat,
        PRIMARY_OUTPUT: primary,
        PRIMARY_NULL_OUTPUT: primary_null,
        INJECTION_OUTPUT: injections,
        RECOVERY_OUTPUT: recovery,
        CONFUSION_OUTPUT: confusion,
        REAL_LABEL_OUTPUT: real_labels,
        PE_EVENT_OUTPUT: pe_events,
        PE_NULL_OUTPUT: pe_null,
        PE_SUMMARY_OUTPUT: pe_summary,
        GATE_OUTPUT: gates,
    }
    paths: dict[str, Path] = {}
    for name, table in tables.items():
        path = output / name
        table.to_csv(path, index=False)
        paths[name] = path

    primary_row = primary.iloc[0]
    pe_row = pe_summary.iloc[0]
    report_lines = [
        "# Pfeiffer/Foster replay revision discriminator",
        "",
        f"Technical gate: **{'pass' if technical else 'fail'}**.",
        f"Candidate-classifiability gate: **{'pass' if classifiable else 'fail'}**.",
        f"Adjudication-ready gate: **{'pass' if adjudication_ready else 'fail'}**.",
        "",
        "## Paired retrospective-content geometry",
        "",
        (
            f"Equal-animal score {primary_row['equal_animal_mean']:+.4f}, "
            f"hierarchical 95% CI [{primary_row['hierarchical_bootstrap_ci_low']:+.4f}, "
            f"{primary_row['hierarchical_bootstrap_ci_high']:+.4f}], "
            f"restricted circular-null retrospective p={primary_row['retrospective_one_sided_p']:.4f}; "
            f"interpretation '{primary_row['interpretation']}'."
        ),
        "",
        "Positive values are closer to the reversed past route; negative values are closer to the future route.",
        "",
        "## Finite prediction-error diagnostic",
        "",
        (
            f"Ordered-minus-time-permuted transition surprise "
            f"{pe_row['equal_animal_ordered_minus_permuted']:+.4f} nats, "
            f"hierarchical 95% CI [{pe_row['hierarchical_bootstrap_ci_low']:+.4f}, "
            f"{pe_row['hierarchical_bootstrap_ci_high']:+.4f}]."
        ),
        (
            "Lower-tail coherence-surrogate p="
            f"{pe_row['ordered_lower_than_permuted_coherence_p']:.4f}; "
            "upper-tail high-surprise/PE-surrogate p="
            f"{pe_row['ordered_higher_than_permuted_pe_surrogate_p']:.4f}."
        ),
        "",
        "## Historical positive-control boundary",
        "",
        "The exact PF2013 away-reward-well future-path positive control is unavailable "
        "in this frozen top-20-per-session cohort. PF2013 selected hundreds of confirmed "
        "trajectory events using its own 20 ms decoder advanced by 5 ms, shuffle tests, "
        "and continuity truncation; the present 4 ms emission/IMM posteriors and selected "
        "event boundaries do not recover those event identities. Physical position and "
        "behavior permit a sensitivity analysis, but not the exact historical control. "
        "The geometry result is therefore technical but nonadjudicative.",
        "",
        "Post-hoc audit context only (not a preregistered reproduction gate): for the 43 "
        "paired next-movement/home-bound events (4 rats, 8 sessions), emission-mean "
        "Euclidean geometry was future-directed (-0.0866). A closer PF-style sensitivity "
        "used the physical animal position, radii 15 cm then +2 cm, first segment "
        "crossings, route paths truncated after both 10 s and 50 cm, and longest decoded "
        "runs with steps below 20 cm. Only 9 emission-MAP events (3 rats/4 sessions) and "
        "14 IMM-MAP events (3 rats/5 sessions) crossed comparable rings; their equal-rat "
        "past-minus-future angular indices were +43.95 and +21.98 degrees, respectively. "
        "After the >=10-bin and >=40-cm displacement proxy, coverage fell to 1 and 3 "
        "events. Directional values at this coverage cannot substitute for the unavailable "
        "historical positive control.",
        "",
        "## Claim boundary",
        "",
        "This is a retrospective-content geometry and candidate-recovery audit. "
        "Path similarity is not a measurement of Bayesian smoothing, posterior revision, "
        "prediction-error signaling, acetylcholine, or causal replay function. A separate "
        "pre-replay filtering to post-replay smoothing bridge is required.",
        "",
        "The `pe_disordered` injection is only a high-transition-surprise time-disorder "
        "surrogate. It is not a generator or measurement of neural prediction error.",
        "",
        "The primary estimand gives each rat equal weight and events equal weight within "
        "rat. Its hierarchical bootstrap resamples rat, then session, then event while "
        "preserving each sampled session's realized event count.",
    ]
    report_path = output / REPORT_OUTPUT
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    paths[REPORT_OUTPUT] = report_path

    manifest = {
        "analysis": "pf_replay_retrospective_geometry_discriminator",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "claim_level": "retrospective_content_geometry_only",
        "candidate_labels": list(PURE_LABELS),
        "calibration": {
            "schemes": ["leave_one_animal_out", "leave_one_session_out"],
            "target_retained_accuracy": float(target_recovery_accuracy),
            "minimum_pure_coverage": float(minimum_recovery_coverage),
            "minimum_mixture_abstention": float(minimum_mixture_abstention),
            "real_label_requires_loao_loso_agreement": True,
            "real_label_requires_adjudication_ready": True,
        },
        "historical_positive_control": {
            "analysis": "PF2013 away-reward-well future-path preference",
            "exactly_reproducible_from_frozen_inputs": False,
            "reason": (
                "The frozen top-20-per-session cohort uses 4 ms emission/IMM posteriors "
                "and does not identify the hundreds of PF2013 confirmed trajectory events "
                "selected by the original 20 ms decoder advanced by 5 ms, shuffle tests, "
                "event boundaries, and continuity truncation."
            ),
            "role": "required for adjudication; unavailable means technical_nonadjudicative",
            "exploratory_proxy_not_a_gate": {
                "cohort": "next_movement and home_bound",
                "events": 43,
                "rats": 4,
                "sessions": 8,
                "emission_mean_euclidean_retrospective_score": -0.08660214362772249,
                "physical_center_angular_proxy": {
                    "method": (
                        "radii 15 cm then +2 cm; first segment crossings; behavior until "
                        "both 10 s and 50 cm; longest decoded run with steps below 20 cm"
                    ),
                    "emission_map": {
                        "events": 9, "rats": 3, "sessions": 4,
                        "equal_rat_past_minus_future_degrees": 43.954877115886276,
                        "continuity_proxy_events": 1,
                    },
                    "imm_map": {
                        "events": 14, "rats": 3, "sessions": 5,
                        "equal_rat_past_minus_future_degrees": 21.984848834983424,
                        "continuity_proxy_events": 3,
                    },
                },
            },
        },
        "adjudication_ready": adjudication_ready,
        "geometry": {
            "positive_score": "closer_to_reversed_past_route",
            "negative_score": "closer_to_future_route",
            "null": "nonzero_circular_paired_template_shift_within_session_and_relation",
            "noise_source": "event_run_decoder_error_cm",
            "noise_model": "anchored_ar1_fixed_radial_rms",
        },
        "prediction_error_diagnostic": {
            "behavior_training": "run_routes_excluding_event_cv_fold",
            "finite_oos_category": True,
            "bin_cm": float(markov_bin_cm),
            "alpha": float(markov_alpha),
            "null": "within_event_nontrivial_time_permutation_fixed_point_set",
            "pe_disordered_candidate": (
                "high-transition-surprise time-disorder surrogate; not a neural "
                "prediction-error generator or measurement"
            ),
            "ordered_lower_than_permuted_sign": "temporal coherence surrogate",
            "ordered_higher_than_permuted_sign": "high-surprise/PE surrogate",
        },
        "estimand": {
            "primary": "equal-rat mean with equal event weight within rat",
            "bootstrap": (
                "rat-to-session-to-event resampling preserving sampled sessions' "
                "realized event counts"
            ),
        },
        "parameters": {
            "permutations": int(permutations),
            "bootstraps": int(bootstraps),
            "pe_permutations": int(pe_permutations),
            "injections_per_candidate": int(injections_per_candidate),
            "seed": int(seed),
        },
        "claim_boundary": (
            "Retrospective path similarity is a geometry gate, not evidence of Bayesian "
            "smoothing or acetylcholine. The filtering-to-smoothing bridge is separate."
        ),
        "outputs": {name: str(path) for name, path in paths.items()},
        "output_file_sha256": {name: file_sha256(path) for name, path in paths.items()},
        "provenance": build_script_provenance(input_paths=inputs, cwd=ROOT),
    }
    manifest_path = output / MANIFEST_OUTPUT
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    paths[MANIFEST_OUTPUT] = manifest_path
    return paths


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", required=True)
    parser.add_argument("--route-segments", required=True)
    parser.add_argument("--route-points", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--permutations", type=int, default=2000)
    parser.add_argument("--bootstraps", type=int, default=4000)
    parser.add_argument("--pe-permutations", type=int, default=500)
    parser.add_argument("--injections-per-candidate", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--markov-bin-cm", type=float, default=10.0)
    parser.add_argument("--markov-alpha", type=float, default=0.5)
    parser.add_argument("--target-recovery-accuracy", type=float, default=0.80)
    parser.add_argument("--minimum-recovery-coverage", type=float, default=0.50)
    parser.add_argument("--minimum-mixture-abstention", type=float, default=0.50)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_analysis(
        events_csv=args.events,
        route_segments_csv=args.route_segments,
        route_points_csv=args.route_points,
        output_dir=args.output_dir,
        permutations=args.permutations,
        bootstraps=args.bootstraps,
        pe_permutations=args.pe_permutations,
        injections_per_candidate=args.injections_per_candidate,
        seed=args.seed,
        markov_bin_cm=args.markov_bin_cm,
        markov_alpha=args.markov_alpha,
        target_recovery_accuracy=args.target_recovery_accuracy,
        minimum_recovery_coverage=args.minimum_recovery_coverage,
        minimum_mixture_abstention=args.minimum_mixture_abstention,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
