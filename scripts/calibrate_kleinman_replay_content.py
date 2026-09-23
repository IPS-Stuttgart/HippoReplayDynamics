#!/usr/bin/env python3
"""Known-map synthetic short-window calibration, never a reward-effect scorer."""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from _provenance import build_script_provenance, file_sha256
from validate_kleinman_run_decoder import (
    PARAMETERS as RUN_PARAMETERS,
)
from validate_kleinman_run_decoder import (
    align_behavior,
    fit_maps,
    interval_counts,
    interval_data,
    make_traversals,
    split_units,
)

ARMS = ("matched_gain_poisson", "run_rate_poisson", "count_conditioned")
PARAMETERS = {
    "seed": 20260923,
    "repeats": 16,
    "dt_s": 0.001,
    "window_ms": 40,
    "step_ms": 5,
    "durations_s": [0.1, 0.2, 0.4],
    "expected_spikes": [24, 48, 96],
    "extents": [0.25, 0.5, 0.75],
    "profiles": ["linear", "cosine", "pause_step"],
    "maximum_fraction_error": 0.1,
    "minimum_support": 0.95,
    "maximum_static_call_fraction": 0.1,
}


def select_sessions(frame):
    if frame.duplicated(["animal", "session"]).any():
        raise ValueError("duplicate RUN session")
    passed = frame.decoder_pass.astype(str).str.lower().eq("true")
    selected = frame.loc[passed].sort_values(["animal", "session"]).groupby("animal").head(1)
    if len(selected) != 6 or selected.animal.nunique() != 6:
        raise ValueError("six decoder-pass animals required")
    return selected


def full_map(folder):
    info = loadmat(folder / "session_info.mat", simplify_cells=True)["session_info"]
    t, x, speed, ends, visits, epochs = align_behavior(info)
    traversals = make_traversals(visits, epochs)
    keys, trains, _ = split_units(loadmat(folder / "spike_data.mat", simplify_cells=True)["spike_data"])
    step = RUN_PARAMETERS["bin_cm"]
    edges = np.arange(np.floor(x.min() / step) * step, np.ceil(x.max() / step) * step + step, step)
    dt, _, trainable, _, direction, spatial = interval_data(t, x, speed, traversals, edges)
    rates, support, units, occupancy = fit_maps(interval_counts(trains, t), dt, direction, spatial, trainable, len(edges) - 1)
    if units.sum() < 5 or not support.any():
        raise ValueError("full RUN encoding model insufficient")
    return {"rates": rates, "support": support, "edges": edges, "ends": ends, "units": keys[units], "occupancy": occupancy}


def trajectory(duration, extent, side, profile, ends):
    n = round(duration / PARAMETERS["dt_s"])
    u = (np.arange(n) + 0.5) / n
    if profile == "linear":
        progress = u
    elif profile == "cosine":
        progress = (1 - np.cos(np.pi * u)) / 2
    elif profile == "pause_step":
        progress = np.clip((u - 0.25) * 2, 0, 1)
    else:
        raise ValueError("unknown timing profile")
    return ends[side] + (1 - 2 * side) * extent * np.diff(ends).item() * progress


def sliding_windows(values):
    a = np.asarray(values)
    width, step = PARAMETERS["window_ms"], PARAMETERS["step_ms"]
    starts = np.arange(0, len(a) - width + 1, step)
    if not len(starts):
        raise ValueError("event shorter than decoding window")
    sums = np.concatenate([np.zeros_like(a[:1]), np.cumsum(a, axis=0)])
    return sums[starts + width] - sums[starts], (starts + width / 2) * PARAMETERS["dt_s"]


def observations(model, *, duration, extent, side, profile, expected_spikes, event_index):
    truth = trajectory(duration, extent, side, profile, model["ends"])
    n_bins = len(model["edges"]) - 1
    spatial = np.searchsorted(model["edges"], truth, side="right") - 1
    if np.any((spatial < 0) | (spatial >= n_bins)):
        raise ValueError("synthetic trajectory outside recorded map")
    states = spatial + side * n_bins
    raw_intensity = model["rates"][states] * PARAMETERS["dt_s"]
    gain = expected_spikes / raw_intensity.sum()
    rng = np.random.default_rng(np.random.SeedSequence([PARAMETERS["seed"], event_index]))
    fine_counts = rng.poisson(raw_intensity * gain)
    counts, centers = sliding_windows(fine_counts)
    truth_windows, _ = sliding_windows(truth)
    return {
        "counts": counts,
        "centers": centers,
        "truth_windows": truth_windows / PARAMETERS["window_ms"],
        "gain": float(gain),
        "n_spikes": int(fine_counts.sum()),
        "n_active_units": int((fine_counts.sum(axis=0) > 0).sum()),
        "true_support_fraction": float(model["support"][states].mean()),
    }


def posterior(counts, rates, support, gain, arm):
    if arm not in ARMS or not np.isfinite(gain) or gain <= 0:
        raise ValueError("invalid decoding arm or gain")
    if not support.any() or np.any(rates <= 0):
        raise ValueError("unsupported or nonpositive map")
    if arm == "count_conditioned":
        log_rates = np.log(rates) - np.log(rates.sum(axis=1, keepdims=True))
        values = counts @ log_rates.T
    else:
        use_gain = gain if arm == "matched_gain_poisson" else 1.0
        values = counts @ np.log(rates).T - 0.04 * use_gain * rates.sum(axis=1)
    values[:, ~support] = -np.inf
    return np.exp(values - logsumexp(values, axis=1, keepdims=True))


def early_late(values, centers):
    lo, hi = centers[0], centers[-1]
    early = centers <= lo + 0.25 * (hi - lo)
    late = centers >= hi - 0.25 * (hi - lo)
    return float(np.mean(values[late]) - np.mean(values[early]))


def weighted_correlation(weights, times, locations):
    total = weights.sum()
    if total <= 0:
        return 0.0
    w = weights / total
    mt = np.sum(w.sum(axis=1) * times)
    mx = np.sum(w.sum(axis=0) * locations)
    vt = np.sum(w.sum(axis=1) * (times - mt) ** 2)
    vx = np.sum(w.sum(axis=0) * (locations - mx) ** 2)
    if vt * vx <= 1e-20:
        return 0.0
    return float(np.sum(w * (times - mt)[:, None] * (locations - mx)[None, :]) / np.sqrt(vt * vx))


def measure(probability, model, centers, side):
    locations = (model["edges"][1:] + model["edges"][:-1]) / 2
    n_bins = len(locations)
    mean = probability @ np.tile(locations, 2)
    mass = probability.reshape(len(centers), 2, n_bins).sum(axis=(0, 2)) / len(centers)
    dominant = int(np.argmax(mass))
    corr = weighted_correlation(probability[:, dominant * n_bins : (dominant + 1) * n_bins], centers, locations)
    delta = (1 - 2 * side) * early_late(mean, centers)
    span = np.diff(model["ends"]).item()
    reverse = mass[dominant] >= 0.55 and abs(corr) >= 0.5 and corr * (2 * dominant - 1) < 0 and delta > 0.1 * span
    return {
        "displacement_cm": delta,
        "displacement_fraction": delta / span,
        "incoming_direction_mass": float(mass[side]),
        "dominant_direction": dominant,
        "weighted_correlation": corr,
        "reverse_content_call": bool(reverse),
    }


def summarize(frame):
    keys = ["animal", "session", "side", "arm", "duration_s", "expected_spikes", "extent", "profile"]
    return (
        frame.groupby(keys, dropna=False)
        .agg(
            n_events=("event_index", "size"),
            median_spikes=("n_spikes", "median"),
            median_active_units=("n_active_units", "median"),
            median_displacement_fraction=("displacement_fraction", "median"),
            true_displacement_fraction=("true_displacement_fraction", "first"),
            median_signed_error_fraction=("error_fraction", "median"),
            median_absolute_error_fraction=("absolute_error_fraction", "median"),
            median_incoming_direction_mass=("incoming_direction_mass", "median"),
            reverse_call_fraction=("reverse_content_call", "mean"),
            minimum_true_support=("true_support_fraction", "min"),
        )
        .reset_index()
    )


def readiness(frame):
    primary = frame.loc[(frame.duration_s == 0.2) & frame.expected_spikes.isin([48, 96]) & (frame.profile == "linear")]
    rows = []
    for arm in ARMS:
        arm_frame = primary.loc[primary.arm == arm]
        for (animal, side), f in arm_frame.groupby(["animal", "side"]):
            strata = f.groupby(["expected_spikes", "extent"])
            bias = strata.error_fraction.median()
            by_count = bias.unstack(0)
            displacement = strata.displacement_fraction.median().unstack(1)
            static = f.loc[f.extent == 0].groupby("expected_spikes").reverse_content_call.mean()
            complete = len(f) == 128 and len(strata) == 8 and strata.size().eq(16).all()
            checks = {
                "complete": bool(complete),
                "finite": bool(np.isfinite(f[["error_fraction", "weighted_correlation"]]).all().all()),
                "support": bool(f.true_support_fraction.min() >= 0.95),
                "absolute_error": bool(f.absolute_error_fraction.median() <= 0.1),
                "bias": bool(len(bias) == 8 and bias.abs().max() <= 0.1),
                "count_stability": bool(set(by_count.columns) == {48, 96} and (by_count[96] - by_count[48]).abs().max() <= 0.1),
                "short_long": bool({0.25, 0.75}.issubset(displacement.columns) and len(displacement) == 2 and (displacement[0.75] > displacement[0.25]).all()),
                "static_specificity": bool(len(static) == 2 and static.max() <= 0.1),
            }
            rows.append(dict(arm=arm, animal=animal, side=side, **checks, passed=all(checks.values())))
    gates = pd.DataFrame(rows)
    summary = []
    for arm in ARMS:
        f = gates.loc[gates.arm == arm] if len(gates) else pd.DataFrame()
        good = bool(len(f) == 12 and f.animal.nunique() == 6 and f.passed.all())
        summary.append({"arm": arm, "strata": len(f), "passing_strata": int(f.passed.sum()) if len(f) else 0, "passed": good, "oracle_gain": arm == "matched_gain_poisson"})
    return gates, pd.DataFrame(summary)


def plot_summary(summary, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(13, 8), sharex="row", sharey="row")
    colors = ["#28788e", "#be573e", "#63883e", "#705a98", "#af8022", "#344e66"]
    for col, arm in enumerate(ARMS):
        sub = summary.loc[(summary.arm == arm) & (summary.duration_s == 0.2) & (summary.profile == "linear")]
        for color, animal in zip(colors, sorted(sub.animal.unique()), strict=True):
            a = sub.loc[sub.animal == animal]
            for side, marker in [(0, "o"), (1, "^")]:
                b = a.loc[(a.side == side) & (a.expected_spikes == 48)]
                axes[0, col].scatter(b.true_displacement_fraction, b.median_displacement_fraction, color=color, marker=marker, s=25, label=animal if side == 0 else None)
            means = a.groupby("expected_spikes").median_signed_error_fraction.mean()
            axes[1, col].plot(means.index, means.values, "o-", color=color)
        axes[0, col].plot([0, 0.6], [0, 0.6], color="black", linestyle="--", linewidth=1)
        axes[0, col].set_title(arm.replace("_", " "), fontsize=11)
        axes[0, col].set_xlabel("True early-to-late displacement / span")
        axes[1, col].axhline(0, color="black", linewidth=1)
        axes[1, col].set_xlabel("Expected event spikes")
    axes[0, 0].set_ylabel("Median decoded displacement / span")
    axes[1, 0].set_ylabel("Mean condition-median signed error / span")
    axes[0, 2].legend(fontsize=8)
    fig.suptitle("Synthetic recovery only: 200 ms known-map events\nTop: 48 expected spikes, both ends; bottom: all extents/ends equally weighted")
    fig.tight_layout()
    fig.savefig(output / "kleinman_content_calibration.png", dpi=160)
    plt.close(fig)


def run(args):
    started = time.monotonic()
    selected = select_sessions(pd.read_csv(args.run_qc))
    inputs = {
        "run_qc": args.run_qc,
        "protocol": ROOT / "docs/kleinman_replay_content_calibration_protocol.md",
        "producer": Path(__file__),
        "run_encoder": ROOT / "scripts/validate_kleinman_run_decoder.py",
    }
    folders = {}
    for row in selected.itertuples():
        matches = list(args.dataset_root.glob(f"**/{row.animal}/{row.session}/spike_data.mat"))
        if len(matches) != 1:
            raise ValueError("nonunique source session " + row.animal + "/" + row.session)
        folders[row.animal] = matches[0].parent
        for name in ("spike_data.mat", "session_info.mat"):
            inputs[row.animal + "_" + name] = matches[0].parent / name
    provenance = build_script_provenance(cwd=ROOT, input_paths=inputs)
    if provenance["git_dirty"] or provenance["code_commit"] == "unavailable":
        raise ValueError("clean committed producer required")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    selected[["animal", "session", "mean_posterior_mean_error_cm", "minimum_fold_units"]].to_csv(args.output_dir / "selected_sessions.csv", index=False)
    rows, map_rows = [], []
    event_index = 0
    shapes = [(0.0, "linear")] + list(product(PARAMETERS["extents"], PARAMETERS["profiles"]))
    for row in selected.itertuples():
        model = full_map(folders[row.animal])
        np.savez_compressed(args.output_dir / (row.animal + "_known_map.npz"), **model)
        map_rows.append(
            {
                "animal": row.animal,
                "session": row.session,
                "n_units": len(model["units"]),
                "n_bins": len(model["edges"]) - 1,
                "track_span_cm": np.diff(model["ends"]).item(),
                "supported_fraction": float(model["support"].mean()),
            }
        )
        for side, shape, duration, target, replicate in product(range(2), shapes, PARAMETERS["durations_s"], PARAMETERS["expected_spikes"], range(PARAMETERS["repeats"])):
            extent, profile = shape
            observed = observations(model, duration=duration, extent=extent, side=side, profile=profile, expected_spikes=target, event_index=event_index)
            truth = (1 - 2 * side) * early_late(observed["truth_windows"], observed["centers"]) / np.diff(model["ends"]).item()
            for arm in ARMS:
                p = posterior(observed["counts"], model["rates"], model["support"], observed["gain"], arm)
                metrics = measure(p, model, observed["centers"], side)
                error = metrics["displacement_fraction"] - truth
                rows.append(
                    dict(
                        animal=row.animal,
                        session=row.session,
                        event_index=event_index,
                        side=side,
                        extent=extent,
                        profile=profile,
                        duration_s=duration,
                        expected_spikes=target,
                        replicate=replicate,
                        arm=arm,
                        **metrics,
                        true_displacement_fraction=truth,
                        error_fraction=error,
                        absolute_error_fraction=abs(error),
                        **{k: observed[k] for k in ("gain", "n_spikes", "n_active_units", "true_support_fraction")},
                    )
                )
            event_index += 1
        pd.DataFrame([r for r in rows if r["animal"] == row.animal]).to_csv(args.output_dir / (row.animal + "_events.csv"), index=False)
        print(json.dumps({"animal": row.animal, "completed_events": event_index, "elapsed_s": time.monotonic() - started}), flush=True)
    frame = pd.DataFrame(rows)
    summary = summarize(frame)
    gates, readiness_summary = readiness(frame)
    for name, data in (("map_inventory", pd.DataFrame(map_rows)), ("condition_summary", summary), ("stratum_gates", gates), ("readiness", readiness_summary)):
        data.to_csv(args.output_dir / ("kleinman_content_" + name + ".csv"), index=False)
    plot_summary(summary, args.output_dir)
    for name, path in inputs.items():
        if file_sha256(path) != provenance["input_file_sha256"][name]:
            raise ValueError("input changed during run")
    manifest = {
        **provenance,
        "parameters": PARAMETERS,
        "run_parameters": RUN_PARAMETERS,
        "host": socket.gethostname(),
        "runtime_s": time.monotonic() - started,
        "n_synthetic_events": event_index,
        "n_decoder_rows": len(frame),
        "real_replay_scored": False,
        "reward_contrast_scored": False,
        "outputs": {p.name: file_sha256(p) for p in args.output_dir.iterdir() if p.is_file()},
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--run-qc", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    run(parser.parse_args())
