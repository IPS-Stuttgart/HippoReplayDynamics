"""Gap-safe paired readouts and explicit detector-cohort summaries."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .replay_coverage import continuity_metrics

IDENTITY = ["dataset", "animal", "session"]
READOUT = ["likelihood", "estimator", "bin_filter", "cell_fraction"]
COHORT = ["detector", "window_variant", "peak_threshold_z", "overlap_scope"]
MEASURES = ["continuity_fraction", "full_continuity_fraction", "paired_continuity_delta",
    "mean_large_jump_fraction", "median_event_speed_cm_s", "median_selected_speed_cm_s",
    "speed_measurable_event_fraction", "selected_speed_measurable_event_fraction",
    "common_speed_measurable_event_fraction", "median_common_step_speed_delta_cm_s"]


def valid_frames(counts, apply_support):
    counts = np.asarray(counts)
    if counts.ndim != 2:
        raise ValueError("counts must be time by cell")
    return (counts.sum(axis=1) >= 3) & ((counts > 0).sum(axis=1) >= 2) if apply_support else np.ones(len(counts), bool)


def nonoverlap_steps(path, valid):
    path, valid = np.asarray(path, float), np.asarray(valid, bool)
    if path.ndim != 2 or valid.shape != (len(path),) or not np.isfinite(path).all():
        raise ValueError("invalid decoded path or support")
    indices = np.arange(0, len(path), 4)
    speeds = np.linalg.norm(np.diff(path[indices], axis=0), axis=1) / .020
    bad_prefix = np.r_[0, np.cumsum(~valid)]
    # Both endpoints AND every intervening overlapping frame are required.
    measured = (bad_prefix[indices[1:] + 1] - bad_prefix[indices[:-1]]) == 0
    return indices, speeds, measured


def paired_metrics(path, counts, rms, entropy, full_path, full_counts, apply_support):
    valid, full_valid = valid_frames(counts, apply_support), valid_frames(full_counts, apply_support)
    if np.shape(path) != np.shape(full_path) or len(counts) != len(full_counts):
        raise ValueError("full and subset must use the identical time windows")
    continuity = continuity_metrics(path, valid_bins=valid)
    full_continuity = continuity_metrics(full_path, valid_bins=full_valid)
    indices, speeds, measured = nonoverlap_steps(path, valid)
    _, full_speeds, full_measured = nonoverlap_steps(full_path, full_valid)
    common = measured & full_measured
    inside = (indices >= continuity["continuous_start"]) & (indices < continuity["continuous_end_exclusive"])
    selected = measured & inside[:-1] & inside[1:] & continuity["continuity_pass"]
    jumps = np.linalg.norm(np.diff(path, axis=0), axis=1)
    adjacent = valid[:-1] & valid[1:]
    return {**continuity, "full_continuity_pass": full_continuity["continuity_pass"],
        "paired_continuity_delta": int(continuity["continuity_pass"]) - int(full_continuity["continuity_pass"]),
        "overlapping_frames": len(path), "supported_frames": int(valid.sum()),
        "large_jump_fraction": float(np.mean(jumps[adjacent] >= 20)) if adjacent.any() else np.nan,
        "nonoverlap_measurable_steps": int(measured.sum()),
        "median_event_speed_cm_s": float(np.median(speeds[measured])) if measured.any() else np.nan,
        "selected_measurable_steps": int(selected.sum()),
        "median_selected_speed_cm_s": float(np.median(speeds[selected])) if selected.any() else np.nan,
        "common_measurable_steps": int(common.sum()),
        "median_common_step_speed_delta_cm_s": float(np.median(speeds[common] - full_speeds[common])) if common.any() else np.nan,
        "median_posterior_rms_cm": float(np.median(np.asarray(rms)[valid])) if valid.any() else np.nan,
        "median_posterior_entropy_nats": float(np.median(np.asarray(entropy)[valid])) if valid.any() else np.nan}


def condition_cohorts(windows, dataset):
    """Emit even empty/unavailable detector strata, not just observed groups."""
    detectors = ["source_high_mua", "native_ripple_table" if dataset == "pfeiffer_foster" else "lfp_ripple_detected"]
    ripple_available = windows.ripple_status.eq("available").all()
    for detector in detectors:
        thresholds = [3., 4., 5.] if detector == "lfp_ripple_detected" else [0.]
        for variant in ["detected_core", "peak_centered_200ms"]:
            base = windows[windows.detector.eq(detector) & windows.window_variant.eq(variant) & windows.eligible]
            for threshold in thresholds:
                frame = base[base.ripple_peak_z >= threshold] if threshold else base
                for scope in ["all_eligible", "both_detectors", "detector_only", "ripple_unavailable"]:
                    names = {"both_detectors": "both_detectors", "detector_only": "mua_only" if detector == "source_high_mua" else "ripple_only", "ripple_unavailable": "ripple_unavailable"}
                    selected = frame if scope == "all_eligible" else frame[frame.overlap_class.eq(names[scope])]
                    available = detector == "source_high_mua" or ripple_available
                    if scope in {"both_detectors", "detector_only"}:
                        available &= ripple_available
                    if scope == "ripple_unavailable":
                        available = detector == "source_high_mua" and not ripple_available
                    yield dict(zip(COHORT, [detector, variant, threshold, scope], strict=True)), selected.window_uid.to_numpy(), bool(available)


def session_summary(metrics, windows, population_specs):
    if windows[IDENTITY].drop_duplicates().shape[0] != 1:
        raise ValueError("one source session required, including excluded rows")
    identity = windows.iloc[0][IDENTITY].to_dict()
    cohorts = list(condition_cohorts(windows, identity["dataset"]))
    rows = []
    for spec in population_specs:
        for likelihood in ["poisson", "conditional_multinomial"]:
            for estimator in ["map", "posterior_mean"]:
                for bin_filter in ["unfiltered", "at_least_2cells_3spikes"]:
                    selected = metrics[(metrics.cell_fraction == spec["cell_fraction"]) & (metrics.population_replicate == spec["population_replicate"]) & metrics.likelihood.eq(likelihood) & metrics.estimator.eq(estimator) & metrics.bin_filter.eq(bin_filter)]
                    for cohort, ids, available in cohorts:
                        local = selected[selected.window_uid.isin(ids)]
                        if len(local) != len(ids):
                            raise ValueError("missing/duplicated event condition rows")
                        n = len(local)
                        rows.append({**identity, **cohort, "detector_available": available,
                            "cell_fraction": spec["cell_fraction"], "population_replicate": spec["population_replicate"],
                            "retained_cells": len(spec["cell_ids"]), "likelihood": likelihood, "estimator": estimator, "bin_filter": bin_filter,
                            "eligible_events": n if available else np.nan, "metric_rows": n,
                            "continuity_fraction": float(local.continuity_pass.mean()) if n else np.nan,
                            "full_continuity_fraction": float(local.full_continuity_pass.mean()) if n else np.nan,
                            "paired_continuity_delta": float(local.paired_continuity_delta.mean()) if n else np.nan,
                            "mean_large_jump_fraction": float(local.large_jump_fraction.mean()) if n else np.nan,
                            "median_event_speed_cm_s": float(local.median_event_speed_cm_s.median()) if n else np.nan,
                            "median_selected_speed_cm_s": float(local.median_selected_speed_cm_s.median()) if n else np.nan,
                            "median_common_step_speed_delta_cm_s": float(local.median_common_step_speed_delta_cm_s.median()) if n else np.nan,
                            "speed_measurable_event_fraction": float((local.nonoverlap_measurable_steps > 0).mean()) if n else np.nan,
                            "selected_speed_measurable_event_fraction": float((local.selected_measurable_steps > 0).mean()) if n else np.nan,
                            "common_speed_measurable_event_fraction": float((local.common_measurable_steps > 0).mean()) if n else np.nan})
    return pd.DataFrame(rows)


def summarize_animals(population, bootstraps=5000, seed=20260906):
    keys = COHORT + READOUT
    session = population.groupby(IDENTITY + keys, as_index=False).agg(
        **{name: (name, "mean") for name in MEASURES}, eligible_events=("eligible_events", "first"),
        detector_available=("detector_available", "all"), recording_subsets=("population_replicate", "size"))
    animal = session.groupby(["dataset", "animal"] + keys, as_index=False).agg(
        **{name: (name, "mean") for name in MEASURES}, source_sessions=("session", "size"),
        available_sessions=("detector_available", "sum"), measurable_sessions=("continuity_fraction", "count"),
        eligible_events=("eligible_events", lambda x: x.sum(min_count=1)))
    rng = np.random.default_rng(seed)
    rows = []
    for group, local in animal.groupby(["dataset"] + keys, sort=True):
        for metric in MEASURES:
            values = local[metric].dropna().to_numpy(float)
            ci = [np.nan, np.nan]
            if len(values) >= 2:
                ci = np.quantile(rng.choice(values, (bootstraps, len(values)), replace=True).mean(axis=1), [.025, .975])
            rows.append({**dict(zip(["dataset"] + keys, group, strict=True)), "metric": metric,
                "animals_total": len(local), "animals_measurable": len(values),
                "equal_animal_mean": float(np.mean(values)) if len(values) else np.nan,
                "ci95_low": ci[0], "ci95_high": ci[1],
                "uncertainty_scope": "animal_bootstrap_after_session_and_recording_subset_averaging"})
    return session, animal, pd.DataFrame(rows)
