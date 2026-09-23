#!/usr/bin/env python3
"""Blocked-lap RUN decoding feasibility; no replay or manipulation contrast."""

from __future__ import annotations

import argparse
import json
import socket
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter1d
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import itertools

from _provenance import build_script_provenance, file_sha256

PARAMETERS = {"bin_cm": 2.0, "smooth_cm": 4.0, "training_speed_cm_s": 8.0,
                  "test_speed_cm_s": 20.0, "maximum_speed_cm_s": 200.0,
                  "minimum_dt_s": 0.001, "maximum_dt_s": 0.1, "test_window_s": 0.25,
                  "interior_margin_cm": 20.0, "minimum_run_spikes": 10, "minimum_peak_hz": 1.0,
                  "minimum_occupancy_s": 0.1, "rate_floor_hz": 1e-5, "folds": 5,
                  "minimum_units": 5, "minimum_test_spikes": 6, "minimum_test_windows": 20,
                  "maximum_mean_error_cm": 35.0, "minimum_direction_accuracy": 0.6}


def matrix(value, width, name):
    a = np.asarray(value, dtype=float)
    if not a.size:
        return np.empty((0, width))
    a = np.atleast_2d(a)
    if a.ndim != 2 or a.shape[1] != width or not np.isfinite(a).all():
        raise ValueError("invalid " + name)
    return a


def integer_indices(value, low, high, name):
    a = np.asarray(value, float)
    if not np.isfinite(a).all() or np.any(a != np.floor(a)) or np.any((a < low) | (a > high)):
        raise ValueError("invalid " + name + " indices")
    return a.astype(int)


def align_behavior(info):
    x = np.asarray(info["position"], float).reshape(-1)
    v = matrix(info["velocity"], 2, "velocity")
    if len(v) < 3 or np.any(np.diff(v[:, 0]) <= 0):
        raise ValueError("nonmonotonic_or_insufficient_velocity_clock")
    if len(x) != len(v) + 1 or not np.isfinite(x).all():
        raise ValueError("unexpected_position_length_or_nonfinite_position")
    t = v[:, 0]
    ends = np.asarray(info["reward_ends"], float).reshape(-1)
    if ends.size != 2 or not np.isfinite(ends).all() or ends[1] <= ends[0]:
        raise ValueError("invalid reward ends")
    visits = []
    for side, name in enumerate(("left_visit", "right_visit")):
        indices = integer_indices(matrix(info[name], 2, name), 2, len(x), name)
        for entry, exit_ in indices:
            if exit_ <= entry:
                raise ValueError("nonpositive_visit_duration")
            # Visits index original position, unlike epoch_change which indexes velocity.
            visits.append({"side": side, "entry_index": int(entry - 2), "exit_index": int(exit_ - 2),
                               "start_s": t[entry - 2], "end_s": t[exit_ - 2],
                               "entry_position_cm": x[entry - 1], "exit_position_cm": x[exit_ - 1],
                               "threshold_cm": ends[side],
                               "entry_inside": bool(x[entry - 1] < ends[side] if side == 0 else x[entry - 1] > ends[side])})
    visits.sort(key=lambda r: r["start_s"])
    if any(b["start_s"] < a["end_s"] for a, b in itertools.pairwise(visits)):
        raise ValueError("overlapping_native_visits")
    changes = integer_indices(matrix(info["epoch_change"], 2, "epoch_change"), 1, len(t), "epoch_change") - 1
    if np.any(changes[:, 1] <= changes[:, 0]) or np.any(np.diff(changes.reshape(-1)) <= 0):
        raise ValueError("unordered_epoch_transitions")
    epochs = list(zip(np.r_[t[0], t[changes[:, 1]]], np.r_[t[changes[:, 0]], t[-1]], strict=True))
    return t, x[1:], v[:, 1], ends, visits, epochs


def make_traversals(visits, epochs, n_folds=5):
    rows = []
    for a, b in itertools.pairwise(visits):
        start, stop = a["end_s"], b["start_s"]
        epoch = [i for i, (lo, hi) in enumerate(epochs) if start >= lo and stop <= hi]
        if a["side"] == b["side"] or stop <= start or len(epoch) != 1:
            continue
        rows.append({"start_s": start, "end_s": stop, "direction": b["side"], "epoch": epoch[0] + 1})
    groups = np.arange(len(rows)) // 2
    if len(np.unique(groups)) < n_folds:
        raise ValueError("insufficient_complete_lap_groups")
    fold_by_group = {int(g): f for f, batch in enumerate(np.array_split(np.unique(groups), n_folds)) for g in batch}
    for i, row in enumerate(rows):
        row.update(traversal=i, lap_group=int(groups[i]), fold=fold_by_group[int(groups[i])])
    return rows


def split_units(spikes):
    a = matrix(spikes, 3, "spikes")
    if np.any(a[:, 1:] != np.floor(a[:, 1:])):
        raise ValueError("noninteger_unit_ids")
    keep = np.all(a[:, 1:] > 0, axis=1)
    excluded = int((~keep).sum())
    a = a[keep]
    keys, inverse = np.unique(a[:, [2, 1]].astype(int), axis=0, return_inverse=True)
    trains = [np.sort(a[inverse == i, 0]) for i in range(len(keys))]
    return keys, trains, excluded


def interval_counts(trains, t):
    return np.column_stack([np.diff(np.searchsorted(s, t, side="left")) for s in trains])


def interval_data(t, x, speed, traversals, edges):
    dt, dx = np.diff(t), np.diff(x)
    mid_speed = (speed[:-1] + speed[1:]) / 2
    valid = ((dt >= PARAMETERS["minimum_dt_s"]) & (dt <= PARAMETERS["maximum_dt_s"])
             & (np.maximum(speed[:-1], speed[1:]) <= PARAMETERS["maximum_speed_cm_s"])
             & (np.minimum(speed[:-1], speed[1:]) >= 0)
             & (np.abs(dx) / dt <= PARAMETERS["maximum_speed_cm_s"]))
    fold = np.full(len(dt), -1)
    direction = np.full(len(dt), -1)
    for row in traversals:
        mask = (t[:-1] >= row["start_s"]) & (t[1:] <= row["end_s"])
        if np.any(fold[mask] != -1):
            raise ValueError("overlapping_traversals")
        fold[mask], direction[mask] = row["fold"], row["direction"]
    spatial = np.searchsorted(edges, (x[:-1] + x[1:]) / 2, side="right") - 1
    spatial = np.clip(spatial, 0, len(edges) - 2)
    moving = (mid_speed > PARAMETERS["training_speed_cm_s"]) & (dx * (2 * direction - 1) > 0)
    trainable = valid & moving & (fold >= 0)
    return dt, valid, trainable, fold, direction, spatial


def fit_maps(counts, dt, direction, spatial, mask, n_bins):
    occupancy = np.zeros((2, n_bins))
    cell_counts = np.zeros((2, n_bins, counts.shape[1]))
    np.add.at(occupancy, (direction[mask], spatial[mask]), dt[mask])
    np.add.at(cell_counts, (direction[mask], spatial[mask]), counts[mask])
    sigma = PARAMETERS["smooth_cm"] / PARAMETERS["bin_cm"]
    smoothed_occ = gaussian_filter1d(occupancy, sigma, axis=1, mode="constant", truncate=4)
    smoothed_counts = gaussian_filter1d(cell_counts, sigma, axis=1, mode="constant", truncate=4)
    rates = smoothed_counts / np.maximum(smoothed_occ[:, :, None], 1e-12)
    units = ((cell_counts.sum(axis=(0, 1)) >= PARAMETERS["minimum_run_spikes"])
             & (rates.max(axis=(0, 1)) >= PARAMETERS["minimum_peak_hz"]))
    supported = occupancy >= PARAMETERS["minimum_occupancy_s"]
    rates = np.maximum(rates[:, :, units], PARAMETERS["rate_floor_hz"])
    return rates.reshape(2 * n_bins, -1), supported.reshape(-1), units, occupancy


def decode(counts, rates, supported, exposure):
    if not supported.any() or not rates.shape[1]:
        raise ValueError("empty_encoding_model")
    values = counts @ np.log(rates).T - exposure * rates.sum(axis=1)
    values[:, ~supported] = -np.inf
    return values - logsumexp(values, axis=1, keepdims=True)


def window_behavior(t, x, speed, start, end, valid):
    left = np.searchsorted(t, start, side="right") - 1
    right = np.searchsorted(t, end, side="left")
    if left < 0 or right >= len(t) or not valid[left:right].all():
        return None
    inside = t[(t > start) & (t < end)]
    knots = np.r_[start, inside, end]
    weights = np.diff(knots)
    px, pv = np.interp(knots, t, x), np.interp(knots, t, speed)
    return float(np.sum(weights * (px[:-1] + px[1:]) / 2) / (end - start)), float(np.sum(weights * (pv[:-1] + pv[1:]) / 2) / (end - start))


def session_pass(mean_error, direction_accuracy, n_windows, min_units, n_folds):
    return bool(np.isfinite(mean_error) and np.isfinite(direction_accuracy)
                and mean_error <= PARAMETERS["maximum_mean_error_cm"]
                and direction_accuracy >= PARAMETERS["minimum_direction_accuracy"]
                and n_windows >= PARAMETERS["minimum_test_windows"]
                and min_units >= PARAMETERS["minimum_units"] and n_folds == PARAMETERS["folds"])


def score_session(folder):
    info = loadmat(folder / "session_info.mat", simplify_cells=True)["session_info"]
    t, x, speed, ends, visits, epochs = align_behavior(info)
    runs = make_traversals(visits, epochs, PARAMETERS["folds"])
    keys, trains, excluded = split_units(loadmat(folder / "spike_data.mat", simplify_cells=True)["spike_data"])
    if not len(keys):
        raise ValueError("no_positive_unit_ids")
    step = PARAMETERS["bin_cm"]
    edges = np.arange(np.floor(x.min() / step) * step, np.ceil(x.max() / step) * step + step, step)
    n_bins = len(edges) - 1
    if n_bins < 2 or n_bins > 1000:
        raise ValueError("implausible_track_geometry")
    centers = (edges[:-1] + edges[1:]) / 2
    state_x = np.tile(centers, 2)
    counts = interval_counts(trains, t)
    dt, valid, trainable, fold, direction, spatial = interval_data(t, x, speed, runs, edges)
    folds, windows = [], []
    for f in range(PARAMETERS["folds"]):
        rates, supported, units, occ = fit_maps(counts, dt, direction, spatial, trainable & (fold != f), n_bins)
        selected = [trains[i] for i in np.flatnonzero(units)]
        diagnostics = {"fold": f, "n_units": int(units.sum()), "n_supported_states": int(supported.sum()),
                           "training_occupancy_s": float(occ.sum()), "n_tracking_rejected": 0,
                           "n_behavior_rejected": 0, "n_spike_rejected": 0, "n_windows": 0}
        if not selected or not supported.any():
            diagnostics["status"] = "empty_encoding_model"
            folds.append(diagnostics)
            continue
        starts, metadata, observations = [], [], []
        for row in runs:
            if row["fold"] != f:
                continue
            width = PARAMETERS["test_window_s"]
            for start in np.arange(row["start_s"], row["end_s"] - width + 1e-9, width):
                end = start + width
                behavior = window_behavior(t, x, speed, start, end, valid)
                if behavior is None:
                    diagnostics["n_tracking_rejected"] += 1
                    continue
                truth, velocity = behavior
                if (velocity <= PARAMETERS["test_speed_cm_s"] or truth <= ends[0] + PARAMETERS["interior_margin_cm"]
                        or truth >= ends[1] - PARAMETERS["interior_margin_cm"]):
                    diagnostics["n_behavior_rejected"] += 1
                    continue
                c = np.array([np.searchsorted(s, end, side="left") - np.searchsorted(s, start, side="left") for s in selected])
                if c.sum() < PARAMETERS["minimum_test_spikes"]:
                    diagnostics["n_spike_rejected"] += 1
                    continue
                starts.append(start)
                metadata.append({"traversal": row["traversal"], "lap_group": row["lap_group"], "epoch": row["epoch"],
                                     "true_direction": row["direction"], "true_position_cm": truth, "speed_cm_s": velocity,
                                     "start_s": start, "end_s": end, "n_spikes": int(c.sum()), "n_active_units": int((c > 0).sum())})
                observations.append(c)
        diagnostics.update(n_windows=len(starts), status="scored" if starts else "no_eligible_test_windows")
        folds.append(diagnostics)
        if not starts:
            continue
        logp = decode(np.asarray(observations), rates, supported, PARAMETERS["test_window_s"])
        probability = np.exp(logp)
        mean = probability @ state_x
        map_state = logp.argmax(axis=1)
        right_prob = probability[:, n_bins:].sum(axis=1)
        for i, meta in enumerate(metadata):
            truth = int(np.clip(np.searchsorted(edges, meta["true_position_cm"], side="right") - 1, 0, n_bins - 1)) + meta["true_direction"] * n_bins
            ent = -np.sum(probability[i, supported] * logp[i, supported]) / np.log(supported.sum()) if supported.sum() > 1 else 0.0
            windows.append(dict(**meta, fold=f, n_units=int(units.sum()), posterior_mean_cm=float(mean[i]),
                                map_cm=float(state_x[map_state[i]]), posterior_rms_cm=float(np.sqrt(probability[i] @ (state_x - mean[i]) ** 2)),
                                posterior_mean_error_cm=float(abs(mean[i] - meta["true_position_cm"])),
                                map_error_cm=float(abs(state_x[map_state[i]] - meta["true_position_cm"])),
                                right_direction_probability=float(right_prob[i]),
                                direction_correct=0.5 if np.isclose(right_prob[i], 0.5, atol=1e-12, rtol=0) else float((right_prob[i] > 0.5) == meta["true_direction"]),
                                posterior_entropy_normalized=float(ent), true_state_supported=bool(supported[truth]),
                                true_state_log_score_above_uniform=float(logp[i, truth] + np.log(supported.sum())) if supported[truth] else np.nan))
    frame = pd.DataFrame(windows)
    summary = {"n_units_raw": len(keys), "excluded_nonpositive_id_spikes": excluded, "n_traversals": len(runs),
                   "n_position_samples": len(x) + 1, "duration_s": float(t[-1] - t[0]),
                   "fraction_valid_tracking_intervals": float(valid.mean()),
                   "visit_entries_inside_fraction": float(np.mean([r["entry_inside"] for r in visits])),
                   "position_alignment": "position[1:]_matches_velocity_times_source_code",
                   "n_test_windows": len(frame), "minimum_fold_units": min(r["n_units"] for r in folds),
                   "n_scored_folds": sum(r["status"] == "scored" for r in folds), "status": "scored"}
    for metric in ("posterior_mean_error_cm", "map_error_cm", "direction_correct", "true_state_supported",
                   "posterior_rms_cm", "posterior_entropy_normalized"):
        summary["mean_" + metric] = float(frame[metric].mean()) if len(frame) else np.nan
    summary["decoder_pass"] = session_pass(summary["mean_posterior_mean_error_cm"], summary["mean_direction_correct"],
                                           len(frame), summary["minimum_fold_units"], summary["n_scored_folds"])
    summary["map_decoder_pass"] = session_pass(summary["mean_map_error_cm"], summary["mean_direction_correct"],
                                               len(frame), summary["minimum_fold_units"], summary["n_scored_folds"])
    return summary, folds, windows, visits


def run(args):
    source, output = args.dataset_root.resolve(), args.output_dir.resolve()
    paths = sorted(source.glob("Experiment_1/*/*/spike_data.mat"))
    if not paths:
        raise ValueError("no spike sessions")
    inputs = {str(p.relative_to(source)): p for spike in paths for p in spike.parent.glob("*.mat")}
    inputs.update(producer=Path(__file__), protocol=ROOT / "docs/kleinman_run_decoder_protocol.md", readme=source / "README.md")
    inputs.update({"author_code_" + p.name: p for p in args.author_code_dir.glob("*.m")})
    if "author_code_trodes_analysis.m" not in inputs:
        raise ValueError("author alignment source missing")
    provenance = build_script_provenance(cwd=ROOT, input_paths=inputs)
    if provenance["git_dirty"] or provenance["code_commit"] == "unavailable":
        raise ValueError("committed clean producer required")
    output.mkdir(parents=True, exist_ok=False)
    sessions, folds, windows, visits = [], [], [], []
    for spike in paths:
        folder = spike.parent
        key = {"animal": folder.parent.name, "session": folder.name}
        info = loadmat(folder / "session_info.mat", simplify_cells=True)["session_info"]
        key.update(drug=int(info["drug"]), novel=int(info["novel"]), incr_end=int(info["incr_end"]))
        native = {}
        for name in ("sdes", "ripple_events"):
            native["native_" + name] = len(matrix(loadmat(folder / (name + ".mat"), simplify_cells=True)[name], 4, name))
        try:
            summary, ff, ww, vv = score_session(folder)
            sessions.append({**key, **native, **summary})
            folds.extend({**key, **r} for r in ff)
            windows.extend({**key, **r} for r in ww)
            visits.extend({**key, **r} for r in vv)
        except (ValueError, KeyError, IndexError) as exc:
            sessions.append({**key, **native, "status": "failed", "failure_reason": str(exc), "decoder_pass": False, "map_decoder_pass": False})
        print(json.dumps({**key, "status": sessions[-1]["status"], "decoder_pass": sessions[-1]["decoder_pass"]}), flush=True)
    session_frame = pd.DataFrame(sessions)
    for name, rows in (("sessions", sessions), ("folds", folds), ("windows", windows), ("visit_alignment", visits)):
        pd.DataFrame(rows).to_csv(output / ("kleinman_run_" + name + ".csv"), index=False)
    groups = session_frame.groupby(["animal", "drug", "novel"], dropna=False).agg(
        n_sessions=("session", "size"), decoder_pass_sessions=("decoder_pass", "sum"),
        mean_session_error_cm=("mean_posterior_mean_error_cm", "mean"),
        mean_session_direction_accuracy=("mean_direction_correct", "mean"))
    groups.to_csv(output / "kleinman_run_by_animal_condition.csv")
    for name, p in inputs.items():
        if file_sha256(p) != provenance["input_file_sha256"][name]:
            raise ValueError("input changed during run")
    manifest = {**provenance, "host": socket.gethostname(), "parameters": PARAMETERS,
                "n_spike_sessions": len(paths), "n_sessions_scored": int((session_frame.status == "scored").sum()),
                "n_decoder_pass_sessions": int(session_frame.decoder_pass.sum()), "replay_scored": False,
                "biological_contrast_scored": False,
                "outputs": {p.name: file_sha256(p) for p in output.iterdir() if p.is_file()}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--author-code-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    run(parser.parse_args())
