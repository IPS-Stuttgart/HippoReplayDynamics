#!/usr/bin/env python3
"""Direct conditional spike likelihood on a fixed synthetic replay bank."""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from _provenance import build_script_provenance, file_sha256
from calibrate_kleinman_replay_content import trajectory

PARAMETERS = {"seed": 20260923, "fit_bin_ms": 10, "grid_points": 9, "minimum_support": 0.95, "fit_profiles": ["linear", "cosine"], "tie_tolerance": 1e-10, "error_limit": 0.1}


def replay_counts(model, row):
    x = trajectory(row.duration_s, row.extent, row.side, row.profile, model["ends"])
    n_bins = len(model["edges"]) - 1
    state = np.searchsorted(model["edges"], x, side="right") - 1 + row.side * n_bins
    intensity = 0.001 * model["rates"][state]
    rng = np.random.default_rng(np.random.SeedSequence([PARAMETERS["seed"], row.event_index]))
    fine = rng.poisson(intensity * (row.expected_spikes / intensity.sum()))
    if int(fine.sum()) != row.n_spikes or int((fine.sum(axis=0) > 0).sum()) != row.n_active_units:
        raise ValueError("old synthetic event reconstruction mismatch")
    width = PARAMETERS["fit_bin_ms"]
    return fine.reshape(-1, width, fine.shape[1]).sum(axis=1)


def template_library(model, duration):
    n_times = round(duration * 1000)
    if n_times % PARAMETERS["fit_bin_ms"]:
        raise ValueError("duration not a whole number of fit bins")
    fraction_time = (np.arange(n_times) + 0.5) / n_times
    fractions = np.linspace(0, 1, PARAMETERS["grid_points"])
    lo, hi = model["ends"]
    n_bins = len(model["edges"]) - 1
    metadata, logq = [], []
    for direction in range(2):
        for start in fractions:
            for end in fractions:
                for profile in ["static"] if start == end else PARAMETERS["fit_profiles"]:
                    progress = (1 - np.cos(np.pi * fraction_time)) / 2 if profile == "cosine" else fraction_time
                    x = lo + (hi - lo) * (start + (end - start) * progress)
                    states = np.searchsorted(model["edges"], x, side="right") - 1 + direction * n_bins
                    supported = model["support"][states].mean()
                    if supported < PARAMETERS["minimum_support"]:
                        continue
                    integrated = model["rates"][states].reshape(-1, PARAMETERS["fit_bin_ms"], model["rates"].shape[1]).sum(axis=1)
                    logq.append(np.log(integrated) - np.log(integrated.sum(axis=1, keepdims=True)))
                    metadata.append({"start": start, "end": end, "extent": abs(end - start), "direction": direction, "profile": profile, "supported_fraction": float(supported)})
    if not metadata or not any(m["extent"] == 0 for m in metadata) or not any(m["extent"] > 0 for m in metadata):
        raise ValueError("both static and moving hypotheses required")
    return np.asarray(logq), pd.DataFrame(metadata)


def fit_events(counts, logq, templates):
    # The fitter receives only observations and hypotheses, never generating parameters.
    counts = np.asarray(counts)
    if counts.ndim != 3 or counts.shape[1:] != logq.shape[1:]:
        raise ValueError("event/template shape mismatch")
    if np.any(counts < 0) or not np.isfinite(counts).all():
        raise ValueError("invalid counts")
    ll = counts.reshape(len(counts), -1) @ logq.reshape(len(logq), -1).T
    best = ll.max(axis=1)
    ties = ll >= best[:, None] - PARAMETERS["tie_tolerance"]
    weights = np.exp(ll - logsumexp(ll, axis=1, keepdims=True))
    extents = templates.extent.to_numpy()
    result = pd.DataFrame(
        {
            "estimated_extent": ties @ extents / ties.sum(axis=1),
            "likelihood_weighted_extent": weights @ extents,
            "estimated_start": ties @ templates.start.to_numpy() / ties.sum(axis=1),
            "estimated_end": ties @ templates.end.to_numpy() / ties.sum(axis=1),
            "n_best_ties": ties.sum(axis=1),
            "best_conditional_log_likelihood": best,
        }
    )
    static = extents == 0
    predictive = np.zeros(len(counts))
    train_indices = np.arange(counts.shape[1]) % 2 == 0
    for train in (train_indices, ~train_indices):
        train_ll = counts[:, train].reshape(len(counts), -1) @ logq[:, train].reshape(len(logq), -1).T
        test_ll = counts[:, ~train].reshape(len(counts), -1) @ logq[:, ~train].reshape(len(logq), -1).T
        scores = []
        for family in (static, ~static):
            f_train = train_ll[:, family]
            f_ties = f_train >= f_train.max(axis=1, keepdims=True) - PARAMETERS["tie_tolerance"]
            scores.append((f_ties * test_ll[:, family]).sum(axis=1) / f_ties.sum(axis=1))
        predictive += scores[1] - scores[0]
    result["crossfit_moving_minus_static"] = predictive
    result["n_templates"] = len(templates)
    return result


def condition_summary(frame):
    keys = ["animal", "session", "side", "profile", "duration_s", "expected_spikes", "extent"]
    f = frame.copy()
    f["error"] = f.estimated_extent - f.extent
    f["absolute_error"] = f.error.abs()
    f["moving_extent"] = f.estimated_extent > 0.1
    return (
        f.groupby(keys)
        .agg(
            n_events=("event_index", "size"),
            median_estimated_extent=("estimated_extent", "median"),
            median_error=("error", "median"),
            median_absolute_error=("absolute_error", "median"),
            moving_extent_fraction=("moving_extent", "mean"),
            median_predictive_delta=("crossfit_moving_minus_static", "median"),
            median_likelihood_weighted_extent=("likelihood_weighted_extent", "median"),
        )
        .reset_index()
    )


def screens(frame):
    rows = []
    for (animal, side), f in frame.groupby(["animal", "side"]):
        for label, profiles in [("matched_timing", ["linear", "cosine"]), ("unseen_timing", ["pause_step"])]:
            s = f.loc[(f.duration_s == 0.2) & f.expected_spikes.isin([48, 96]) & (f.profile.isin(profiles) | (f.extent == 0))].copy()
            summary = condition_summary(s)
            expected = (1 + 3 * len(profiles)) * 2
            bias = summary.median_error.abs().max()
            count_table = summary.pivot(index=["profile", "extent"], columns="expected_spikes", values="median_estimated_extent")
            count_shift = (count_table[96] - count_table[48]).abs().max() if {48, 96}.issubset(count_table.columns) else np.nan
            moving = summary.loc[summary.extent > 0]
            span_table = moving.pivot(index=["profile", "expected_spikes"], columns="extent", values="median_estimated_extent")
            ranked = bool({0.25, 0.75}.issubset(span_table.columns) and len(span_table) == 2 * len(profiles) and (span_table[0.75] > span_table[0.25]).all())
            static_false = summary.loc[summary.extent == 0, "moving_extent_fraction"].max()
            abs_error = (s.estimated_extent - s.extent).abs().median()
            checks = {
                "complete": bool(len(summary) == expected and summary.n_events.eq(16).all()),
                "finite": bool(np.isfinite(s[["estimated_extent", "crossfit_moving_minus_static"]]).all().all()),
                "absolute_error": bool(abs_error <= 0.1),
                "bias": bool(bias <= 0.1),
                "count_stability": bool(count_shift <= 0.1),
                "short_long": ranked,
                "static_specificity": bool(static_false <= 0.1),
            }
            rows.append(
                dict(
                    animal=animal,
                    side=side,
                    screen=label,
                    **checks,
                    passed=all(checks.values()),
                    median_absolute_error=float(abs_error),
                    maximum_condition_bias=float(bias),
                    maximum_count_shift=float(count_shift),
                    maximum_static_false_fraction=float(static_false),
                )
            )
        s = condition_summary(f.loc[f.expected_spikes.isin([48, 96])])
        groups = s.groupby(["profile", "extent", "expected_spikes"])
        ranges = groups.median_estimated_extent.agg(lambda v: v.max() - v.min())
        complete = len(groups) == 20 and groups.duration_s.nunique().eq(3).all() and s.n_events.eq(16).all()
        rows.append(
            {
                "animal": animal,
                "side": side,
                "screen": "duration_only",
                "complete": bool(complete),
                "passed": bool(complete and ranges.max() <= 0.1),
                "maximum_duration_shift": float(ranges.max()),
            }
        )
    return pd.DataFrame(rows)


def plot(summary, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True, sharey=True)
    for ax, animal in zip(axes.flat, sorted(summary.animal.unique()), strict=True):
        f = summary.loc[(summary.animal == animal) & (summary.duration_s == 0.2) & (summary.expected_spikes == 48)]
        for profile, color in [("linear", "#28788e"), ("cosine", "#9a651b"), ("pause_step", "#b54e59")]:
            for side, marker in [(0, "o"), (1, "^")]:
                a = f.loc[(f.profile == profile) & (f.side == side)]
                ax.scatter(a.extent, a.median_estimated_extent, color=color, marker=marker, label=profile if side == 0 else None)
        ax.plot([0, 1], [0, 1], "k--", linewidth=1)
        ax.set_title(animal)
        ax.set_xlabel("True path extent / span")
        ax.set_ylabel("Median fitted extent / span")
    axes[0, 2].legend(fontsize=8)
    fig.suptitle(
        "Direct integrated spike likelihood: synthetic recovery, 200 ms, 48 expected spikes\nStart, end and directional map unknown; pause-step absent from fitted templates"
    )
    fig.tight_layout()
    fig.savefig(output / "kleinman_integrated_extent.png", dpi=160)
    plt.close(fig)


def run(args):
    started = time.monotonic()
    old = json.loads((args.bank_dir / "manifest.json").read_text())
    verification = json.loads((args.bank_dir / "verification.json").read_text())
    if verification["status"] != "passed" or old["n_synthetic_events"] != 17280:
        raise ValueError("verified fixed bank required")
    for name, digest in old["outputs"].items():
        if file_sha256(args.bank_dir / name) != digest:
            raise ValueError("changed fixed bank " + name)
    inputs = {
        "bank_manifest": args.bank_dir / "manifest.json",
        "bank_verification": args.bank_dir / "verification.json",
        "protocol": ROOT / "docs/kleinman_integrated_extent_protocol.md",
        "producer": Path(__file__),
        "old_generator": ROOT / "scripts/calibrate_kleinman_replay_content.py",
    }
    for p in sorted(args.bank_dir.glob("*_known_map.npz")) + sorted(args.bank_dir.glob("*_events.csv")):
        inputs[p.name] = p
    provenance = build_script_provenance(cwd=ROOT, input_paths=inputs)
    if provenance["git_dirty"] or provenance["code_commit"] == "unavailable":
        raise ValueError("committed clean producer required")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    frames, inventory = [], []
    for path in sorted(args.bank_dir.glob("*_events.csv")):
        source = pd.read_csv(path)
        base = source.loc[source.arm == "matched_gain_poisson"].copy()
        animal = base.animal.iloc[0]
        model = dict(np.load(args.bank_dir / (animal + "_known_map.npz")))
        pieces = []
        for duration, sub in base.groupby("duration_s"):
            logq, templates = template_library(model, duration)
            templates.to_csv(args.output_dir / f"{animal}_{duration}_templates.csv", index=False)
            inventory.append({"animal": animal, "duration_s": duration, "n_templates": len(templates), "n_static": int((templates.extent == 0).sum())})
            counts = np.array([replay_counts(model, row) for row in sub.itertuples()])
            fit = fit_events(counts, logq, templates)
            keys = ["animal", "session", "event_index", "side", "profile", "extent", "duration_s", "expected_spikes", "replicate", "n_spikes", "n_active_units"]
            pieces.append(pd.concat([sub[keys].reset_index(drop=True), fit], axis=1))
        frame = pd.concat(pieces).sort_values("event_index").reset_index(drop=True)
        frame.to_csv(args.output_dir / f"{animal}_estimates.csv", index=False)
        frames.append(frame)
        print(json.dumps({"animal": animal, "events": len(frame), "elapsed_s": time.monotonic() - started}), flush=True)
    frame = pd.concat(frames, ignore_index=True)
    summary = condition_summary(frame)
    gates = screens(frame)
    summary.to_csv(args.output_dir / "kleinman_integrated_condition_summary.csv", index=False)
    gates.to_csv(args.output_dir / "kleinman_integrated_screens.csv", index=False)
    pd.DataFrame(inventory).to_csv(args.output_dir / "template_inventory.csv", index=False)
    plot(summary, args.output_dir)
    for key, path in inputs.items():
        if file_sha256(path) != provenance["input_file_sha256"][key]:
            raise ValueError("input changed during run")
    manifest = {
        **provenance,
        "host": socket.gethostname(),
        "parameters": PARAMETERS,
        "runtime_s": time.monotonic() - started,
        "n_reused_events": len(frame),
        "n_new_simulations": 0,
        "real_replay_scored": False,
        "reward_contrast_scored": False,
        "readiness_passed": bool(len(gates) == 36 and gates.passed.all()),
        "outputs": {p.name: file_sha256(p) for p in args.output_dir.iterdir() if p.is_file()},
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    run(parser.parse_args())
