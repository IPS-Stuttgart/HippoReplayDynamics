#!/usr/bin/env python3
"""Audit matched hc-11 PRE/POST events with map, order, and held-out controls.

The input event table is frozen before model evidence is inspected. For every
selected event and firing-rate population, this script compares the real map
with cell-identity-permuted maps, the original order with whole-bin shuffles,
and model predictions for cells excluded from latent-state inference.

Held-out scores are posterior-marginal predictive scores: the latent posterior
is inferred from training-cell replay spikes only, then held-out emissions are
evaluated against that fixed posterior. Held-out spikes never update the path.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
from math import comb
from pathlib import Path
import sys
import time
import zlib

import numpy as np
import pandas as pd
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
SCRIPT_DIR = ROOT / "scripts"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _provenance import build_script_provenance  # noqa: E402
import score_hc11_pre_post_learning_evidence as learning  # noqa: E402
import score_hc11_webshare_native_ripple_evidence as hc11  # noqa: E402


DEFAULT_DATASET_ROOT = Path("/mnt/lexar4tb/datasets/hc11_grosmark_buzsaki/webshare_processed")
DEFAULT_OUTPUT_DIR = Path("results/hc11-pre-post-learning-controls")

CONTROL_EVIDENCE_OUTPUT = "hc11_pre_post_control_model_evidence.csv"
HELDOUT_OUTPUT = "hc11_pre_post_heldout_predictive_scores.csv"
EVENT_OUTPUT = "hc11_pre_post_control_event_summary.csv"
CONTRAST_OUTPUT = "hc11_pre_post_control_learning_contrasts.csv"
POPULATION_OUTPUT = "hc11_pre_post_control_by_population.csv"
SESSION_EFFECT_OUTPUT = "hc11_pre_post_control_learning_effects_by_session.csv"
ANIMAL_EFFECT_OUTPUT = "hc11_pre_post_control_learning_effects_by_animal.csv"
INFERENCE_OUTPUT = "hc11_pre_post_control_learning_inference.csv"
LEAVE_ONE_ANIMAL_OUT_OUTPUT = "hc11_pre_post_control_leave_one_animal_out.csv"
GATE_OUTPUT = "hc11_pre_post_control_gate_summary.csv"
MANIFEST_OUTPUT = "hc11_pre_post_control_manifest.json"
SUMMARY_OUTPUT = "hc11_pre_post_control_summary.md"

LEARNING_METRICS = {
    "post_minus_pre_validated_ordered_trajectory": "mean",
    "post_minus_pre_validated_clean_imm": "mean",
    "post_minus_pre_map_specific_excess_mean_nonstationary_mode_probability": "median",
    "post_minus_pre_map_specific_excess_posterior_net_displacement_cm": "median",
    "post_minus_pre_time_order_advantage_ordered_minus_nonordered": "median",
    "post_minus_pre_time_order_advantage_imm_minus_fragmented": "median",
    "post_minus_pre_median_heldout_ordered_minus_nonordered": "median",
    "post_minus_pre_median_heldout_imm_minus_fragmented": "median",
}


def stable_seed(base_seed: int, *parts: object) -> int:
    token = "|".join(str(part) for part in parts).encode("utf-8")
    return (int(base_seed) + int(zlib.crc32(token))) % (2**32)


def permute_encoding_maps(
    encodings: list[hc11.EncodingMap],
    permutation: np.ndarray,
) -> list[hc11.EncodingMap]:
    """Permute cell-to-field identity identically across direction maps."""

    order = np.asarray(permutation, dtype=int)
    if not encodings:
        raise ValueError("at least one encoding map is required")
    n_units = len(encodings[0].unit_ids)
    if order.shape != (n_units,) or not np.array_equal(np.sort(order), np.arange(n_units)):
        raise ValueError("permutation must contain every encoding-row index exactly once")
    output: list[hc11.EncodingMap] = []
    for encoding in encodings:
        if len(encoding.unit_ids) != n_units:
            raise ValueError("all direction maps must contain the same units")
        output.append(replace(encoding, rates_hz=np.asarray(encoding.rates_hz)[order]))
    return output


def shuffled_event(
    counts: np.ndarray,
    edges: np.ndarray,
    permutation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Shuffle whole population bins while preserving each bin's duration."""

    values = np.asarray(counts)
    durations = np.diff(np.asarray(edges, dtype=float))
    order = np.asarray(permutation, dtype=int)
    if order.shape != (len(values),) or not np.array_equal(np.sort(order), np.arange(len(values))):
        raise ValueError("time permutation must contain every bin exactly once")
    shuffled_edges = np.concatenate(
        [[float(edges[0])], float(edges[0]) + np.cumsum(durations[order])]
    )
    return values[order].copy(), shuffled_edges


def split_unit_ids(
    unit_ids: tuple[int, ...],
    test_fraction: float,
    seed: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if len(unit_ids) < 2:
        raise ValueError("held-out prediction requires at least two cells")
    fraction = float(test_fraction)
    if not 0.0 < fraction < 1.0:
        raise ValueError("test_fraction must lie strictly between zero and one")
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(len(unit_ids))
    n_test = min(max(int(np.rint(fraction * len(unit_ids))), 1), len(unit_ids) - 1)
    test_index = set(int(value) for value in order[:n_test])
    train = tuple(unit for index, unit in enumerate(unit_ids) if index not in test_index)
    test = tuple(unit for index, unit in enumerate(unit_ids) if index in test_index)
    return train, test


def _model_kwargs(args: argparse.Namespace, track: hc11.TrackSamples) -> dict[str, object]:
    return {
        "topology": track.topology,
        "track_length_cm": track.track_length_cm,
        "diffusion_sigma_cm_sqrt_s": args.diffusion_sigma_cm_sqrt_s,
        "stationary_sigma_cm": args.stationary_sigma_cm,
        "max_step_sigma": args.max_step_sigma,
        "imm_mode_stickiness": args.imm_mode_stickiness,
    }


def score_metrics(
    scores: dict[str, dict[str, object]],
    encoding: hc11.EncodingMap,
    track: hc11.TrackSamples,
    duration_s: float,
    margin_threshold: float,
) -> dict[str, object]:
    logz = {model: float(scores[model]["log_evidence"]) for model in hc11.MODELS}
    best_model = max(logz, key=logz.get)
    best_ordered = max(learning.ORDERED_MODELS, key=lambda model: logz[model])
    best_nonordered = max(learning.NONORDERED_MODELS, key=lambda model: logz[model])
    ordered_margin = logz[best_ordered] - logz[best_nonordered]
    imm_margin = logz["first_order_imm"] - logz["fragmented"]
    content = hc11.imm_content_diagnostics(
        np.asarray(scores["first_order_imm"]["posterior"], dtype=float),
        np.asarray(scores["first_order_imm"]["mode_posterior"], dtype=float),
        encoding.bin_centers_cm,
        track.topology,
        track.track_length_cm,
        duration_s,
    )
    content_positive = bool(
        content["mean_nonstationary_mode_probability"] >= 0.5
        and content["posterior_net_displacement_cm"] >= 10.0
    )
    return {
        **{f"logZ_{model}": value for model, value in logz.items()},
        "best_model": best_model,
        "best_ordered_model": best_ordered,
        "best_nonordered_model": best_nonordered,
        "ordered_minus_nonordered": ordered_margin,
        "imm_minus_fragmented": imm_margin,
        "ordered_confident": ordered_margin >= float(margin_threshold),
        "imm_confident_over_fragmented": imm_margin >= float(margin_threshold),
        "posterior_content_positive": content_positive,
        **content,
    }


def posterior_marginal_predictive_scores(
    train_counts: np.ndarray,
    test_counts: np.ndarray,
    edges: np.ndarray,
    train_encodings: list[hc11.EncodingMap],
    test_encodings: list[hc11.EncodingMap],
    *,
    model_kwargs: dict[str, object],
) -> tuple[dict[str, float], dict[str, object]]:
    """Score held-out cells against training-only smoothed state posteriors."""

    if len(train_encodings) != len(test_encodings) or not train_encodings:
        raise ValueError("train and test direction maps must be non-empty and aligned")
    durations = np.diff(np.asarray(edges, dtype=float))
    train_map_scores = [
        hc11.score_single_encoding(train_counts, edges, encoding, **model_kwargs)
        for encoding in train_encodings
    ]
    heldout_emissions = [
        hc11.poisson_log_likelihood(test_counts, encoding.rates_hz, durations)
        for encoding in test_encodings
    ]
    combined: dict[str, float] = {}
    imm_map_weights: np.ndarray | None = None
    for model in hc11.MODELS:
        train_logz = np.asarray(
            [float(score[model]["log_evidence"]) for score in train_map_scores],
            dtype=float,
        )
        log_weights = train_logz - logsumexp(train_logz)
        map_predictive: list[float] = []
        for score, test_log_likelihood in zip(
            train_map_scores,
            heldout_emissions,
            strict=True,
        ):
            train_log_posterior = np.asarray(score[model]["posterior"], dtype=float)
            per_bin = logsumexp(train_log_posterior + test_log_likelihood, axis=1)
            map_predictive.append(float(np.sum(per_bin)))
        combined[model] = float(logsumexp(log_weights + np.asarray(map_predictive)))
        if model == "first_order_imm":
            imm_map_weights = np.exp(log_weights)

    assert imm_map_weights is not None
    posterior = sum(
        float(weight) * np.exp(np.asarray(score["first_order_imm"]["posterior"], dtype=float))
        for weight, score in zip(imm_map_weights, train_map_scores, strict=True)
    )
    mode_posterior = sum(
        float(weight) * np.asarray(score["first_order_imm"]["mode_posterior"], dtype=float)
        for weight, score in zip(imm_map_weights, train_map_scores, strict=True)
    )
    diagnostics = hc11.imm_content_diagnostics(
        posterior,
        mode_posterior,
        train_encodings[0].bin_centers_cm,
        str(model_kwargs["topology"]),
        float(model_kwargs["track_length_cm"]),
        float(np.sum(durations)),
    )
    return combined, diagnostics


def _quantile(values: pd.Series, probability: float) -> float:
    array = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    return float(np.quantile(array, probability, method="higher")) if array.size else np.nan


def _empirical_upper_p(original: float, null_values: pd.Series) -> float:
    array = pd.to_numeric(null_values, errors="coerce").dropna().to_numpy(dtype=float)
    if not array.size or not np.isfinite(original):
        return np.nan
    return float((1 + np.count_nonzero(array >= original)) / (1 + len(array)))


def build_event_summary(
    controls: pd.DataFrame,
    heldout: pd.DataFrame,
    margin_threshold: float,
) -> pd.DataFrame:
    keys = ["animal", "session", "geometry", "phase", "match_pair_id", "event_id", "population"]
    rows: list[dict[str, object]] = []
    successful = controls[controls["status"].eq("success")]
    for key, group in successful.groupby(keys, sort=True):
        original_rows = group[group["control_type"].eq("original")]
        if len(original_rows) != 1:
            continue
        original = original_rows.iloc[0]
        map_null = group[group["control_type"].eq("map_permutation")]
        order_null = group[group["control_type"].eq("time_shuffle")]
        held = heldout
        for column, value in zip(keys, key, strict=True):
            held = held[held[column].eq(value)]

        row = dict(zip(keys, key, strict=True))
        for metric in (
            "ordered_minus_nonordered",
            "imm_minus_fragmented",
            "mean_nonstationary_mode_probability",
            "posterior_expected_path_length_cm",
            "posterior_net_displacement_cm",
        ):
            original_value = float(original[metric])
            map_median = float(map_null[metric].median()) if not map_null.empty else np.nan
            row[f"original_{metric}"] = original_value
            row[f"map_null_median_{metric}"] = map_median
            row[f"map_specific_excess_{metric}"] = original_value - map_median
            row[f"map_null_p95_{metric}"] = _quantile(map_null[metric], 0.95)
            row[f"map_specific_empirical_p_{metric}"] = _empirical_upper_p(
                original_value,
                map_null[metric],
            )
        for metric in ("ordered_minus_nonordered", "imm_minus_fragmented"):
            original_value = float(original[metric])
            order_median = float(order_null[metric].median()) if not order_null.empty else np.nan
            row[f"time_shuffle_median_{metric}"] = order_median
            row[f"time_order_advantage_{metric}"] = original_value - order_median
            row[f"time_shuffle_p95_{metric}"] = _quantile(order_null[metric], 0.95)
            row[f"time_order_empirical_p_{metric}"] = _empirical_upper_p(
                original_value,
                order_null[metric],
            )

        if not held.empty:
            held = held.copy()
            held["heldout_ordered_minus_nonordered"] = held[
                [f"heldout_{model}" for model in learning.ORDERED_MODELS]
            ].max(axis=1) - held[[f"heldout_{model}" for model in learning.NONORDERED_MODELS]].max(axis=1)
            held["heldout_imm_minus_fragmented"] = (
                held["heldout_first_order_imm"] - held["heldout_fragmented"]
            )
            for metric in ("heldout_ordered_minus_nonordered", "heldout_imm_minus_fragmented"):
                row[f"median_{metric}"] = float(held[metric].median())
                row[f"fraction_positive_{metric}"] = float((held[metric] > 0.0).mean())
            row["heldout_splits"] = int(held["split_index"].nunique())
            row["median_train_nonstationary_mode_probability"] = float(
                held["train_mean_nonstationary_mode_probability"].median()
            )
        else:
            row["median_heldout_ordered_minus_nonordered"] = np.nan
            row["fraction_positive_heldout_ordered_minus_nonordered"] = np.nan
            row["median_heldout_imm_minus_fragmented"] = np.nan
            row["fraction_positive_heldout_imm_minus_fragmented"] = np.nan
            row["heldout_splits"] = 0
            row["median_train_nonstationary_mode_probability"] = np.nan

        row["best_model"] = str(original["best_model"])
        row["original_ordered_confident"] = bool(
            float(original["ordered_minus_nonordered"]) >= float(margin_threshold)
        )
        row["original_imm_confident"] = bool(
            float(original["imm_minus_fragmented"]) >= float(margin_threshold)
        )
        row["original_content_positive"] = bool(original["posterior_content_positive"])
        row["map_specific_content"] = bool(
            float(original["mean_nonstationary_mode_probability"])
            > row["map_null_p95_mean_nonstationary_mode_probability"]
            and float(original["posterior_net_displacement_cm"])
            > row["map_null_p95_posterior_net_displacement_cm"]
        )
        row["time_order_sensitive_ordered"] = bool(
            float(original["ordered_minus_nonordered"])
            > row["time_shuffle_p95_ordered_minus_nonordered"]
        )
        row["time_order_sensitive_imm"] = bool(
            float(original["imm_minus_fragmented"])
            > row["time_shuffle_p95_imm_minus_fragmented"]
        )
        row["heldout_ordered_positive"] = bool(
            np.isfinite(row["median_heldout_ordered_minus_nonordered"])
            and row["median_heldout_ordered_minus_nonordered"] > 0.0
        )
        row["heldout_imm_positive"] = bool(
            np.isfinite(row["median_heldout_imm_minus_fragmented"])
            and row["median_heldout_imm_minus_fragmented"] > 0.0
        )
        row["validated_ordered_trajectory"] = bool(
            row["original_ordered_confident"]
            and row["original_content_positive"]
            and row["map_specific_content"]
            and row["time_order_sensitive_ordered"]
            and row["heldout_ordered_positive"]
        )
        row["validated_clean_imm"] = bool(
            row["best_model"] == "first_order_imm"
            and row["original_ordered_confident"]
            and row["original_imm_confident"]
            and row["original_content_positive"]
            and row["map_specific_content"]
            and row["time_order_sensitive_imm"]
            and row["heldout_imm_positive"]
        )
        rows.append(row)
    return pd.DataFrame(rows)


def learning_contrasts(event_summary: pd.DataFrame) -> pd.DataFrame:
    if event_summary.empty:
        return pd.DataFrame()
    metrics = (
        "map_specific_excess_ordered_minus_nonordered",
        "map_specific_excess_imm_minus_fragmented",
        "map_specific_excess_mean_nonstationary_mode_probability",
        "map_specific_excess_posterior_net_displacement_cm",
        "time_order_advantage_ordered_minus_nonordered",
        "time_order_advantage_imm_minus_fragmented",
        "median_heldout_ordered_minus_nonordered",
        "median_heldout_imm_minus_fragmented",
        "validated_ordered_trajectory",
        "validated_clean_imm",
    )
    index = ["animal", "session", "geometry", "population", "match_pair_id"]
    wide = event_summary.pivot_table(index=index, columns="phase", values=list(metrics), aggfunc="first")
    rows: list[dict[str, object]] = []
    for key, values in wide.iterrows():
        if not all((metric, phase) in wide.columns for metric in metrics for phase in learning.PHASES):
            continue
        row = dict(zip(index, key, strict=True))
        for metric in metrics:
            row[f"pre_{metric}"] = float(values[(metric, "PRE")])
            row[f"post_{metric}"] = float(values[(metric, "POST")])
            row[f"post_minus_pre_{metric}"] = float(values[(metric, "POST")] - values[(metric, "PRE")])
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_populations(event_summary: pd.DataFrame, contrasts: pd.DataFrame) -> pd.DataFrame:
    if event_summary.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for (population, phase), group in event_summary.groupby(["population", "phase"], sort=True):
        rows.append(
            {
                "population": population,
                "phase": phase,
                "events": int(len(group)),
                "animals": int(group["animal"].nunique()),
                "sessions": int(group["session"].nunique()),
                "validated_ordered_count": int(group["validated_ordered_trajectory"].sum()),
                "validated_ordered_fraction": float(group["validated_ordered_trajectory"].mean()),
                "validated_clean_imm_count": int(group["validated_clean_imm"].sum()),
                "validated_clean_imm_fraction": float(group["validated_clean_imm"].mean()),
                "median_map_specific_mode_mass_excess": float(
                    group["map_specific_excess_mean_nonstationary_mode_probability"].median()
                ),
                "median_map_specific_displacement_excess_cm": float(
                    group["map_specific_excess_posterior_net_displacement_cm"].median()
                ),
                "median_time_order_advantage": float(
                    group["time_order_advantage_imm_minus_fragmented"].median()
                ),
                "median_heldout_imm_minus_fragmented": float(
                    group["median_heldout_imm_minus_fragmented"].median()
                ),
            }
        )
    output = pd.DataFrame(rows)
    if not contrasts.empty:
        delta_columns = [column for column in contrasts if column.startswith("post_minus_pre_")]
        medians = contrasts.groupby("population")[delta_columns].median().reset_index()
        output = output.merge(medians, on="population", how="left")
    return output


def _reduce_metric(values: pd.Series, method: str) -> float:
    finite = pd.to_numeric(values, errors="coerce").dropna()
    if finite.empty:
        return np.nan
    if method == "mean":
        return float(finite.mean())
    if method == "median":
        return float(finite.median())
    raise ValueError(f"unknown reduction method {method}")


def learning_effects_by_session_and_animal(
    contrasts: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reduce paired effects within sessions, then weight animals equally."""

    if contrasts.empty:
        return pd.DataFrame(), pd.DataFrame()
    pair_frames: dict[str, pd.DataFrame] = {
        population: contrasts[contrasts["population"].eq(population)].copy()
        for population in learning.POPULATIONS
    }
    slow = pair_frames["slow_firing"]
    fast = pair_frames["fast_firing"]
    join_keys = ["animal", "session", "geometry", "match_pair_id"]
    if not slow.empty and not fast.empty:
        joined = slow.merge(fast, on=join_keys, suffixes=("_slow", "_fast"))
        for metric in LEARNING_METRICS:
            joined[metric] = joined[f"{metric}_slow"] - joined[f"{metric}_fast"]
        pair_frames["slow_minus_fast"] = joined[join_keys + list(LEARNING_METRICS)].copy()

    session_rows: list[dict[str, object]] = []
    for population_contrast, frame in pair_frames.items():
        if frame.empty:
            continue
        for (animal, session), group in frame.groupby(["animal", "session"], sort=True):
            for metric, method in LEARNING_METRICS.items():
                session_rows.append(
                    {
                        "animal": animal,
                        "session": session,
                        "population_contrast": population_contrast,
                        "metric": metric,
                        "within_session_reduction": method,
                        "n_matched_pairs": int(len(group)),
                        "estimate": _reduce_metric(group[metric], method),
                    }
                )
    session_effects = pd.DataFrame(session_rows)
    animal_rows: list[dict[str, object]] = []
    if not session_effects.empty:
        for (animal, population_contrast, metric), group in session_effects.groupby(
            ["animal", "population_contrast", "metric"], sort=True
        ):
            animal_rows.append(
                {
                    "animal": animal,
                    "population_contrast": population_contrast,
                    "metric": metric,
                    "sessions": int(group["session"].nunique()),
                    "matched_pairs": int(group["n_matched_pairs"].sum()),
                    "estimate": float(group["estimate"].mean()),
                }
            )
    return session_effects, pd.DataFrame(animal_rows)


def infer_equal_animal_learning_effects(
    animal_effects: pd.DataFrame,
    *,
    n_bootstraps: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return equal-animal bootstrap CIs and leave-one-animal-out estimates."""

    if animal_effects.empty:
        return pd.DataFrame(), pd.DataFrame()
    rng = np.random.default_rng(int(seed))
    inference_rows: list[dict[str, object]] = []
    leave_one_out_rows: list[dict[str, object]] = []
    for (population_contrast, metric), group in animal_effects.groupby(
        ["population_contrast", "metric"], sort=True
    ):
        ordered = group.sort_values("animal", kind="mergesort")
        values = ordered["estimate"].to_numpy(dtype=float)
        animals = ordered["animal"].astype(str).to_numpy()
        keep = np.isfinite(values)
        values = values[keep]
        animals = animals[keep]
        if values.size == 0:
            continue
        effect = float(np.mean(values))
        nonzero = values[values != 0.0]
        sign_positive = int(np.sum(nonzero > 0.0))
        sign_p = (
            float(
                sum(comb(len(nonzero), count) for count in range(sign_positive, len(nonzero) + 1))
                / (2 ** len(nonzero))
            )
            if len(nonzero)
            else np.nan
        )
        if int(n_bootstraps) > 0:
            samples = rng.choice(values, size=(int(n_bootstraps), len(values)), replace=True)
            bootstrap = np.mean(samples, axis=1)
            ci_low, ci_high = np.quantile(bootstrap, [0.025, 0.975])
        else:
            ci_low = np.nan
            ci_high = np.nan
        loo_values: list[float] = []
        if len(values) > 1:
            for heldout_index, heldout_animal in enumerate(animals):
                estimate = float(np.mean(np.delete(values, heldout_index)))
                loo_values.append(estimate)
                leave_one_out_rows.append(
                    {
                        "population_contrast": population_contrast,
                        "metric": metric,
                        "heldout_animal": heldout_animal,
                        "animals_retained": int(len(values) - 1),
                        "estimate": estimate,
                    }
                )
        inference_rows.append(
            {
                "population_contrast": population_contrast,
                "metric": metric,
                "animals": int(len(values)),
                "equal_animal_mean": effect,
                "rat_bootstrap_ci_low": float(ci_low),
                "rat_bootstrap_ci_high": float(ci_high),
                "positive_animals": int(np.sum(values > 0.0)),
                "all_animals_positive": bool(np.all(values > 0.0)),
                "one_sided_sign_test_n": int(len(nonzero)),
                "one_sided_sign_test_p": sign_p,
                "leave_one_animal_out_min": min(loo_values) if loo_values else np.nan,
                "n_bootstraps": int(n_bootstraps),
                "positive_robust": bool(
                    len(values) >= 4
                    and np.all(values > 0.0)
                    and np.isfinite(ci_low)
                    and float(ci_low) > 0.0
                    and loo_values
                    and min(loo_values) > 0.0
                ),
            }
        )
    return pd.DataFrame(inference_rows), pd.DataFrame(leave_one_out_rows)


def _inference_pass(inference: pd.DataFrame, population: str, metric: str) -> bool:
    if inference.empty:
        return False
    rows = inference[
        inference["population_contrast"].eq(population)
        & inference["metric"].eq(metric)
    ]
    return bool(len(rows) == 1 and rows.iloc[0]["positive_robust"])


def gate_summary(
    controls: pd.DataFrame,
    heldout: pd.DataFrame,
    event_summary: pd.DataFrame,
    contrasts: pd.DataFrame,
    n_map_permutations: int,
    n_time_shuffles: int,
    n_heldout_splits: int,
    inference: pd.DataFrame | None = None,
) -> pd.DataFrame:
    control_counts = controls.groupby("control_type")["replicate"].nunique() if not controls.empty else pd.Series(dtype=int)
    control_keys = ["session", "phase", "event_id", "population", "control_type"]
    per_event_controls = (
        controls[controls["status"].eq("success")]
        .groupby(control_keys)["replicate"]
        .nunique()
        if not controls.empty
        else pd.Series(dtype=int)
    )
    map_complete = (
        per_event_controls.xs("map_permutation", level="control_type")
        if not per_event_controls.empty and "map_permutation" in per_event_controls.index.get_level_values("control_type")
        else pd.Series(dtype=int)
    )
    time_complete = (
        per_event_controls.xs("time_shuffle", level="control_type")
        if not per_event_controls.empty and "time_shuffle" in per_event_controls.index.get_level_values("control_type")
        else pd.Series(dtype=int)
    )
    populations = set(event_summary["population"]) if not event_summary.empty else set()
    animals = int(event_summary["animal"].nunique()) if not event_summary.empty else 0
    heldout_success = heldout[heldout["status"].eq("success")] if not heldout.empty else heldout
    complete_heldout = (
        heldout_success.groupby(["session", "phase", "event_id", "population"])["split_index"].nunique()
        if not heldout.empty
        else pd.Series(dtype=int)
    )
    checks = [
        ("events_present", not event_summary.empty, f"events={len(event_summary)}"),
        ("no_control_scoring_failures", bool(not controls.empty and controls["status"].eq("success").all()), f"failures={int((~controls['status'].eq('success')).sum()) if not controls.empty else 0}"),
        ("no_heldout_scoring_failures", bool(not heldout.empty and heldout["status"].eq("success").all()), f"failures={int((~heldout['status'].eq('success')).sum()) if not heldout.empty else 0}"),
        ("map_permutations_complete", bool(len(map_complete) > 0 and (map_complete == int(n_map_permutations)).all()), f"complete={int((map_complete == int(n_map_permutations)).sum())}/{len(map_complete)}; global_replicates={int(control_counts.get('map_permutation', 0))}"),
        ("time_shuffles_complete", bool(len(time_complete) > 0 and (time_complete == int(n_time_shuffles)).all()), f"complete={int((time_complete == int(n_time_shuffles)).sum())}/{len(time_complete)}; global_replicates={int(control_counts.get('time_shuffle', 0))}"),
        ("heldout_splits_complete", bool(len(complete_heldout) > 0 and (complete_heldout == int(n_heldout_splits)).all()), f"complete={int((complete_heldout == int(n_heldout_splits)).sum())}/{len(complete_heldout)}"),
        ("all_populations_present", populations == set(learning.POPULATIONS), f"populations={sorted(populations)}"),
        ("multiple_animals_present", animals >= 2, f"animals={animals}"),
        ("paired_pre_post_contrasts_present", not contrasts.empty, f"pairs={len(contrasts)}"),
    ]
    technical = all(passed for _, passed, _ in checks)
    checks.append(("overall_technical", technical, "map/order/held-out controls on frozen events"))

    inference = inference if inference is not None else pd.DataFrame()
    strict_metric = "post_minus_pre_validated_ordered_trajectory"
    map_mass_metric = "post_minus_pre_map_specific_excess_mean_nonstationary_mode_probability"
    displacement_metric = "post_minus_pre_map_specific_excess_posterior_net_displacement_cm"
    order_metric = "post_minus_pre_time_order_advantage_ordered_minus_nonordered"
    heldout_metric = "post_minus_pre_median_heldout_ordered_minus_nonordered"
    all_four_animals = bool(
        not inference.empty and int(inference["animals"].max()) >= 4
    )
    all_core = {
        "validated_fraction": _inference_pass(inference, "all", strict_metric),
        "mode_mass": _inference_pass(inference, "all", map_mass_metric),
        "displacement": _inference_pass(inference, "all", displacement_metric),
        "time_order": _inference_pass(inference, "all", order_metric),
        "heldout_prediction": _inference_pass(inference, "all", heldout_metric),
    }
    slow_core = {
        "validated_fraction": _inference_pass(inference, "slow_minus_fast", strict_metric),
        "mode_mass": _inference_pass(inference, "slow_minus_fast", map_mass_metric),
        "displacement": _inference_pass(inference, "slow_minus_fast", displacement_metric),
        "time_order": _inference_pass(inference, "slow_minus_fast", order_metric),
        "heldout_prediction": _inference_pass(inference, "slow_minus_fast", heldout_metric),
    }
    all_positive = bool(technical and all_four_animals and all(all_core.values()))
    slow_specific = bool(all_positive and all(slow_core.values()))
    checks.extend(
        [
            ("four_animals_in_inference", all_four_animals, f"animals={int(inference['animals'].max()) if not inference.empty else 0}"),
            ("post_increase_in_validated_map_specific_dynamics", all_positive, json.dumps(all_core, sort_keys=True)),
            ("post_increase_selective_for_slow_firing_population", slow_specific, json.dumps(slow_core, sort_keys=True)),
            ("learning_dependent_trajectory_dynamics_supported", all_positive, "requires every all-cell component gate and four-rat robustness"),
            ("slow_plastic_population_selectivity_supported", slow_specific, "requires every slow-minus-fast component gate"),
        ]
    )
    return pd.DataFrame(checks, columns=["gate", "passed", "detail"])


def validate_selection_scoring_parameters(
    selection: pd.DataFrame,
    *,
    time_bin_s: float,
    event_padding_s: float,
) -> None:
    for column, requested in (
        ("scoring_time_bin_s", float(time_bin_s)),
        ("scoring_event_padding_s", float(event_padding_s)),
    ):
        if column not in selection:
            continue
        recorded = selection[column].dropna().astype(float).unique()
        if len(recorded) != 1 or not np.isclose(recorded[0], requested):
            raise ValueError(
                f"{column} mismatch: selection records {recorded.tolist()}, "
                f"control requested {requested}"
            )


def run(args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    dataset_root = Path(args.dataset_root).resolve()
    selection_path = Path(args.selection_csv).resolve()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selection = pd.read_csv(selection_path)
    validate_selection_scoring_parameters(
        selection,
        time_bin_s=args.time_bin_s,
        event_padding_s=args.event_padding_s,
    )
    if args.sessions:
        requested = {value.strip() for value in args.sessions.split(",") if value.strip()}
        selection = selection[selection["session"].isin(requested)].copy()
    if selection.empty:
        raise ValueError("selection contains no requested events")

    control_rows: list[dict[str, object]] = []
    heldout_rows: list[dict[str, object]] = []
    for session, selected in selection.groupby("session", sort=True):
        session_dirs = list(dataset_root.glob(f"*/{session}"))
        if len(session_dirs) != 1:
            raise ValueError(f"expected one dataset directory for {session}; found {len(session_dirs)}")
        session_dir = session_dirs[0]
        animal = session_dir.parent.name
        track = hc11.load_track_samples(session_dir)
        spikes = hc11.load_spikes(session_dir)
        encodings, _ = hc11.build_session_encodings(
            track,
            spikes,
            position_bin_size_cm=args.position_bin_size_cm,
            min_run_speed_cm_s=args.min_run_speed_cm_s,
            min_run_spikes=args.min_run_spikes,
            min_spatial_information=args.min_spatial_information,
            min_peak_rate_hz=args.min_peak_rate_hz,
            min_encoding_units=args.min_encoding_units,
            smoothing_sigma_bins=args.smoothing_sigma_bins,
        )
        selected_units = encodings["pooled"][0].unit_ids
        groups, _ = learning.offline_firing_rate_groups(
            session_dir,
            spikes,
            selected_units,
            scope=args.rate_group_scope,
        )
        population_maps = learning.population_encodings(encodings, groups)
        model_kwargs = _model_kwargs(args, track)

        map_permutations: dict[tuple[str, int], np.ndarray] = {}
        heldout_splits: dict[tuple[str, int], tuple[tuple[int, ...], tuple[int, ...]]] = {}
        for population, unit_ids in groups.items():
            if population not in population_maps:
                continue
            for replicate in range(args.n_map_permutations):
                rng = np.random.default_rng(stable_seed(args.random_seed, session, population, "map", replicate))
                map_permutations[(population, replicate)] = rng.permutation(len(unit_ids))
            for split_index in range(args.n_heldout_splits):
                heldout_splits[(population, split_index)] = split_unit_ids(
                    unit_ids,
                    args.heldout_fraction,
                    stable_seed(args.random_seed, session, population, "heldout", split_index),
                )

        for event in selected.itertuples(index=False):
            start_s = max(0.0, float(event.start_time_s) - float(args.event_padding_s))
            end_s = float(event.end_time_s) + float(args.event_padding_s)
            edges = hc11.event_bin_edges(start_s, end_s, args.time_bin_s)
            duration_s = end_s - start_s
            common = {
                "animal": animal,
                "session": session,
                "geometry": track.topology,
                "phase": str(event.phase),
                "match_pair_id": int(event.match_pair_id),
                "event_id": int(event.event_id),
            }
            time_permutations = [
                np.random.default_rng(
                    stable_seed(args.random_seed, session, event.phase, event.event_id, "time", replicate)
                ).permutation(len(edges) - 1)
                for replicate in range(args.n_time_shuffles)
            ]
            for population, unit_ids in groups.items():
                if population not in population_maps:
                    continue
                maps = population_maps[population]
                counts = hc11.spike_count_matrix(spikes, unit_ids, edges)
                control_specs: list[tuple[str, int, np.ndarray, np.ndarray, list[hc11.EncodingMap]]] = [
                    ("original", 0, counts, edges, maps)
                ]
                for replicate in range(args.n_map_permutations):
                    control_specs.append(
                        (
                            "map_permutation",
                            replicate,
                            counts,
                            edges,
                            permute_encoding_maps(maps, map_permutations[(population, replicate)]),
                        )
                    )
                for replicate, permutation in enumerate(time_permutations):
                    shuffled_counts, shuffled_edges = shuffled_event(counts, edges, permutation)
                    control_specs.append(
                        ("time_shuffle", replicate, shuffled_counts, shuffled_edges, maps)
                    )

                for control_type, replicate, current_counts, current_edges, current_maps in control_specs:
                    started = time.perf_counter()
                    try:
                        scores = hc11.score_encoding_variant(
                            current_counts,
                            current_edges,
                            current_maps,
                            **model_kwargs,
                        )
                        metrics = score_metrics(
                            scores,
                            current_maps[0],
                            track,
                            duration_s,
                            args.margin_threshold,
                        )
                        control_rows.append(
                            {
                                **common,
                                "population": population,
                                "control_type": control_type,
                                "replicate": int(replicate),
                                "status": "success",
                                "failure_reason": "",
                                "runtime_s": time.perf_counter() - started,
                                "n_time_bins": len(current_counts),
                                "n_spikes": int(current_counts.sum()),
                                "n_active_units": int(np.sum(current_counts.sum(axis=0) > 0)),
                                "n_encoding_units": len(unit_ids),
                                **metrics,
                            }
                        )
                    except Exception as exc:
                        control_rows.append(
                            {
                                **common,
                                "population": population,
                                "control_type": control_type,
                                "replicate": int(replicate),
                                "status": "failure",
                                "failure_reason": f"{type(exc).__name__}: {exc}",
                                "runtime_s": time.perf_counter() - started,
                            }
                        )

                for split_index in range(args.n_heldout_splits):
                    train_ids, test_ids = heldout_splits[(population, split_index)]
                    train_maps = [learning.subset_encoding_map(encoding, train_ids) for encoding in maps]
                    test_maps = [learning.subset_encoding_map(encoding, test_ids) for encoding in maps]
                    train_counts = hc11.spike_count_matrix(spikes, train_ids, edges)
                    test_counts = hc11.spike_count_matrix(spikes, test_ids, edges)
                    started = time.perf_counter()
                    try:
                        predictive, diagnostics = posterior_marginal_predictive_scores(
                            train_counts,
                            test_counts,
                            edges,
                            train_maps,
                            test_maps,
                            model_kwargs=model_kwargs,
                        )
                        heldout_rows.append(
                            {
                                **common,
                                "population": population,
                                "split_index": int(split_index),
                                "status": "success",
                                "failure_reason": "",
                                "runtime_s": time.perf_counter() - started,
                                "n_train_cells": len(train_ids),
                                "n_heldout_cells": len(test_ids),
                                "n_train_spikes": int(train_counts.sum()),
                                "n_heldout_spikes": int(test_counts.sum()),
                                **{f"heldout_{model}": value for model, value in predictive.items()},
                                **{f"train_{key}": value for key, value in diagnostics.items()},
                            }
                        )
                    except Exception as exc:
                        heldout_rows.append(
                            {
                                **common,
                                "population": population,
                                "split_index": int(split_index),
                                "status": "failure",
                                "failure_reason": f"{type(exc).__name__}: {exc}",
                                "runtime_s": time.perf_counter() - started,
                                "n_train_cells": len(train_ids),
                                "n_heldout_cells": len(test_ids),
                            }
                        )

    controls = pd.DataFrame(control_rows)
    heldout = pd.DataFrame(heldout_rows)
    successful_heldout = heldout[heldout["status"].eq("success")].copy() if not heldout.empty else heldout
    event_summary = build_event_summary(controls, successful_heldout, args.margin_threshold)
    contrasts = learning_contrasts(event_summary)
    populations = summarize_populations(event_summary, contrasts)
    session_effects, animal_effects = learning_effects_by_session_and_animal(contrasts)
    inference, leave_one_out = infer_equal_animal_learning_effects(
        animal_effects,
        n_bootstraps=args.n_animal_bootstraps,
        seed=args.random_seed,
    )
    gates = gate_summary(
        controls,
        heldout,
        event_summary,
        contrasts,
        args.n_map_permutations,
        args.n_time_shuffles,
        args.n_heldout_splits,
        inference,
    )
    outputs = {
        CONTROL_EVIDENCE_OUTPUT: controls,
        HELDOUT_OUTPUT: heldout,
        EVENT_OUTPUT: event_summary,
        CONTRAST_OUTPUT: contrasts,
        POPULATION_OUTPUT: populations,
        SESSION_EFFECT_OUTPUT: session_effects,
        ANIMAL_EFFECT_OUTPUT: animal_effects,
        INFERENCE_OUTPUT: inference,
        LEAVE_ONE_ANIMAL_OUT_OUTPUT: leave_one_out,
        GATE_OUTPUT: gates,
    }
    for name, frame in outputs.items():
        frame.to_csv(output_dir / name, index=False)

    provenance = build_script_provenance(
        input_paths={"selection_csv": selection_path},
        cwd=ROOT,
        argv=sys.argv,
    )
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(dataset_root),
        "selection_csv": str(selection_path),
        "rate_group_scope": args.rate_group_scope,
        "n_map_permutations": args.n_map_permutations,
        "n_time_shuffles": args.n_time_shuffles,
        "n_heldout_splits": args.n_heldout_splits,
        "n_animal_bootstraps": args.n_animal_bootstraps,
        "heldout_score_scope": "training_only_state_posterior_marginal_predictive",
        "ordered_models": list(learning.ORDERED_MODELS),
        "nonordered_models": list(learning.NONORDERED_MODELS),
        "outputs": list(outputs),
        **provenance,
    }
    (output_dir / MANIFEST_OUTPUT).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (output_dir / SUMMARY_OUTPUT).write_text(
        build_markdown_summary(populations, inference, gates)
    )
    return outputs


def build_markdown_summary(
    populations: pd.DataFrame,
    inference: pd.DataFrame,
    gates: pd.DataFrame,
) -> str:
    technical = bool(gates.loc[gates["gate"].eq("overall_technical"), "passed"].all())
    lines = [
        "# hc-11 PRE/POST learning control audit",
        "",
        f"Technical controls complete: **{technical}**.",
        "",
        "## Population summary",
        "",
        "```text",
        populations.to_string(index=False) if not populations.empty else "No events.",
        "```",
        "",
        "## Equal-animal inference",
        "",
        "```text",
        inference.to_string(index=False) if not inference.empty else "No four-rat inference.",
        "```",
        "",
        "## Claim boundary",
        "",
        "A POST learning claim requires a positive map-specific, order-sensitive, held-out-predictive shift across the final four-rat session set. Fragmented is treated as nonordered throughout.",
        "",
    ]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--selection-csv", required=True)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--sessions", help="Optional comma-separated session subset for sharded runs.")
    parser.add_argument(
        "--rate-group-scope",
        choices=("overall_session", "combined_offline_nrem", "pre_nrem"),
        default="pre_nrem",
    )
    parser.add_argument("--n-map-permutations", type=int, default=20)
    parser.add_argument("--n-time-shuffles", type=int, default=20)
    parser.add_argument("--n-heldout-splits", type=int, default=20)
    parser.add_argument("--n-animal-bootstraps", type=int, default=10000)
    parser.add_argument("--heldout-fraction", type=float, default=0.30)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument(
        "--time-bin-s",
        type=float,
        default=0.020,
        help="Must match the primary evidence scorer; 20 ms is paper-matched.",
    )
    parser.add_argument(
        "--event-padding-s",
        type=float,
        default=0.0,
        help="Must match the primary evidence scorer; paper-defined events are unpadded.",
    )
    parser.add_argument("--position-bin-size-cm", type=float, default=4.0)
    parser.add_argument("--min-run-speed-cm-s", type=float, default=5.0)
    parser.add_argument("--min-run-spikes", type=int, default=20)
    parser.add_argument("--min-spatial-information", type=float, default=0.1)
    parser.add_argument("--min-peak-rate-hz", type=float, default=1.0)
    parser.add_argument("--min-encoding-units", type=int, default=5)
    parser.add_argument("--smoothing-sigma-bins", type=float, default=1.5)
    parser.add_argument("--diffusion-sigma-cm-sqrt-s", type=float, default=85.0)
    parser.add_argument("--stationary-sigma-cm", type=float, default=2.0)
    parser.add_argument("--max-step-sigma", type=float, default=4.0)
    parser.add_argument("--imm-mode-stickiness", type=float, default=0.95)
    parser.add_argument("--margin-threshold", type=float, default=5.5)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    outputs = run(args)
    print(
        f"Audited {len(outputs[EVENT_OUTPUT])} event-population rows; "
        f"wrote outputs to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
