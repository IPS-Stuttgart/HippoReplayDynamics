"""Crossed population sampling with fixed event windows and explicit truth."""

from __future__ import annotations

import numpy as np
from scipy.special import softmax

from hipporeplayimm.regional_blind_bank import BlindPath


def tile_ids(xy, bounds):
    xy, bounds = np.asarray(xy, float), np.asarray(bounds, float)
    if bounds.shape != (2, 2) or not np.all(np.isfinite(bounds)) or np.any(bounds[1] <= bounds[0]):
        raise ValueError("invalid tile bounds")
    q = np.clip(np.floor(3 * (xy - bounds[0]) / (bounds[1] - bounds[0])).astype(int), 0, 2)
    return 3 * q[:, 0] + q[:, 1]


def make_populations(ids, rates, region, tetrodes, rng):
    ids, rates = np.asarray(ids), np.asarray(rates, float)
    region, tetrodes = np.asarray(region, bool), np.asarray(tetrodes)
    if len(ids) < 20 or rates.shape != (len(ids), len(region)):
        raise ValueError("at least 20 cells and aligned rate maps required")
    if len(np.unique(ids)) != len(ids) or tetrodes.shape != ids.shape:
        raise ValueError("unique IDs and complete tetrode mapping required")
    if not region.any() or region.all() or not np.isfinite(rates).all() or np.any(rates <= 0):
        raise ValueError("positive rates and two nonempty spatial classes required")
    n, k = len(ids), len(ids) // 2
    score = rates[:, region].mean(axis=1) / rates.mean(axis=1)
    ranked = np.lexsort((ids, score))
    pops = [{"name": "full", "family": "full", "side": "full", "repeat": 0, "indices": list(range(n))}]

    def add(name, family, side, repeat, ix):
        pops.append({"name": name, "family": family, "side": side, "repeat": repeat, "indices": np.sort(ix).tolist()})

    add("targeted_low", "targeted", "low", 0, ranked[:k])
    add("targeted_high", "targeted", "high", 0, ranked[-k:])
    for repeat in range(2):
        order = rng.permutation(n)
        for side, ix in (("a", order[:k]), ("b", order[k : 2 * k])):
            add(f"random{repeat}_{side}", "random", side, repeat, ix)
    missing = []
    electrodes = np.unique(tetrodes)
    if len(electrodes) < 2:
        missing.append("whole_tetrode: fewer than two electrodes")
    else:
        for repeat in range(2):
            proposals = []
            for _ in range(128):
                order = rng.permutation(electrodes)
                sizes = np.array([np.count_nonzero(tetrodes == t) for t in order])
                cut = int(np.argmin(np.abs(np.cumsum(sizes)[:-1] - n / 2))) + 1
                ix = np.flatnonzero(np.isin(tetrodes, order[:cut]))
                proposals.append((abs(len(ix) - n / 2), tuple(ix)))
            _, chosen = min(proposals)
            a = np.array(chosen, int)
            b = np.setdiff1d(np.arange(n), a)
            if min(len(a), len(b)) < 5:
                missing.append(f"whole_tetrode{repeat}: fewer than five cells in one half")
                continue
            for side, ix in (("a", a), ("b", b)):
                add(f"whole_tetrode{repeat}_{side}", "whole_tetrode", side, repeat, ix)
    return pops, score, missing


def endpoint_windows(events):
    out = []
    for e in events:
        start, end = float(e["event_start_s"]), float(e["event_end_s"])
        n = int(np.floor((end - start) / 0.005 + 1e-8))
        if n < 4:
            raise ValueError("event shorter than four base bins")
        out.append((start + (n - 4) * 0.005, start + n * 0.005))
    return np.asarray(out, float).reshape(-1, 2)


def count_windows(spikes, ids, windows):
    spikes, windows = np.asarray(spikes), np.asarray(windows).reshape(-1, 2)
    if not np.isfinite(windows).all() or np.any(windows[:, 1] <= windows[:, 0]):
        raise ValueError("invalid windows")
    counts = np.zeros((len(windows), len(ids)), np.int64)
    for j, cell in enumerate(ids):
        ts = np.sort(spikes[spikes[:, 1] == cell, 0])
        counts[:, j] = np.searchsorted(ts, windows[:, 1], side="left") - np.searchsorted(ts, windows[:, 0], side="left")
    return counts


def posterior_readout(counts, rates, tiles, likelihood):
    n, r, tiles = np.asarray(counts), np.asarray(rates), np.asarray(tiles, int)
    if n.ndim != 2 or r.ndim != 2 or n.shape[1] != len(r) or r.shape[1] != len(tiles):
        raise ValueError("likelihood shape mismatch")
    if not np.isfinite(n).all() or np.any(n < 0) or np.any(n != np.floor(n)):
        raise ValueError("counts must be finite nonnegative integers")
    if not np.isfinite(r).all() or np.any(r <= 0) or len(tiles) < 2:
        raise ValueError("positive rates on at least two grid locations required")
    log_likelihood = n @ np.log(r)
    if likelihood == "poisson":
        log_likelihood -= 0.02 * r.sum(axis=0)
    elif likelihood == "conditional_multinomial":
        log_likelihood -= n.sum(axis=1)[:, None] * np.log(r.sum(axis=0))
    else:
        raise ValueError("unknown likelihood")
    p = softmax(log_likelihood, axis=1)
    return {
        "regional": p @ np.eye(9)[tiles],
        "entropy": -(p * np.log(np.maximum(p, 1e-300))).sum(axis=1) / np.log(len(tiles)),
        "spikes": n.sum(axis=1),
        "active": np.count_nonzero(n, axis=1),
    }


def truth_mass(truth_tiles, windows, dt=0.005):
    """Exact occupancy for a piecewise-constant truth process and arbitrary edges."""
    labels, w = np.asarray(truth_tiles, int), np.asarray(windows).reshape(-1, 2)
    if np.any((labels < 0) | (labels > 8)) or not len(labels) or dt <= 0:
        raise ValueError("invalid truth process")
    if not np.isfinite(w).all() or np.any(w[:, 0] < -1e-8) or np.any(w[:, 1] > len(labels) * dt + 1e-8) or np.any(w[:, 1] <= w[:, 0]):
        raise ValueError("truth windows outside process support")
    edges = np.arange(len(labels) + 1) * dt
    answer = np.zeros((len(w), 9))
    for region in range(9):
        primitive = np.r_[0.0, np.cumsum(labels == region) * dt]
        answer[:, region] = (np.interp(w[:, 1], edges, primitive) - np.interp(w[:, 0], edges, primitive)) / (w[:, 1] - w[:, 0])
    return answer


def factorial(qff, qsf, qfs, qss):
    values = np.asarray([qff, qsf, qfs, qss], float)
    if not np.isfinite(values).all():
        return {"status": "incomplete", "detection": np.nan, "decoding": np.nan, "interaction": np.nan, "total": np.nan}
    d, c = qsf - qff, qfs - qff
    interaction = qss - qsf - qfs + qff
    return {"status": "complete", "detection": d, "decoding": c, "interaction": interaction, "total": qss - qff}


def event_matches(reference, alternate):
    """Descriptive maximum-overlap links; non-one-to-one matches remain visible."""
    ref = np.asarray(reference, float).reshape(-1, 2)
    alt = np.asarray(alternate, float).reshape(-1, 2)
    ix, amount = np.full(len(alt), -1, int), np.zeros(len(alt))
    if len(ref):
        for j, (a, b) in enumerate(alt):
            overlap = np.maximum(0, np.minimum(ref[:, 1], b) - np.maximum(ref[:, 0], a))
            if overlap.max() > 0:
                ix[j], amount[j] = int(overlap.argmax()), float(overlap.max())
    return ix, amount


def detect(detector, spikes, ids, speed_times, speeds, intervals):
    selected = spikes[np.isin(spikes[:, 1], ids)]
    out = []
    for interval, (start, end) in enumerate(intervals):
        if end - start < 0.05:
            continue
        events = detector.detect_high_mua_in_interval(
            selected,
            speed_times,
            speeds,
            float(start),
            float(end),
            bin_s=0.001,
            gaussian_sd_s=0.010,
            z_threshold=3.0,
            maximum_speed_cm_s=5.0,
            minimum_duration_s=0.050,
            maximum_duration_s=2.0,
            minimum_active_cells=max(1, int(np.ceil(0.1 * len(ids)))),
        )
        base = len(out)
        out.extend(dict(e, interval=interval, event_id=base + j) for j, e in enumerate(events))
    return out


def simulate(geometry, rates, ids, kind, peak, duration, rng):
    """Region-blind paths and unconditioned Poisson observations on a 5-ms clock."""
    if kind not in ("stationary", "moving") or peak < 1 or duration < 2 or duration % 2:
        raise ValueError("invalid simulation settings")
    dt, epoch_s = 0.005, 2.0
    centers = (np.arange(400) + 0.5) * dt
    gain = 1 + (peak - 1) * np.exp(-0.5 * ((centers - 1) / 0.04) ** 2)
    spike_parts, positions, sampled_means = [], [], []
    for epoch in range(duration // 2):
        path = BlindPath(geometry, kind, 500.0, [20.0], [0.02], 1.0, rng).extend(epoch_s)
        rates_at = path.intensities(centers, rates)
        expected = rates_at * (gain * dt)[:, None]
        counts = rng.poisson(expected)
        time_bin, cell = np.nonzero(counts)
        number = counts[time_bin, cell]
        time_bin, cell = np.repeat(time_bin, number), np.repeat(cell, number)
        time = epoch * epoch_s + (time_bin + rng.random(len(time_bin))) * dt
        spike_parts.append(np.column_stack((time, ids[cell])))
        knots, nodes = np.asarray(path.ages), np.asarray(path.nodes)
        k = np.clip(np.searchsorted(knots, centers, side="left") - 1, 0, len(knots) - 2)
        xy = geometry.grid[nodes[k]].copy()
        if kind == "moving":
            f = (centers - knots[k]) / (knots[k + 1] - knots[k])
            xy += f[:, None] * (geometry.grid[nodes[k + 1]] - xy)
        positions.append(xy)
        sampled_means.append(expected.sum(axis=0))
    spikes = np.concatenate(spike_parts)
    spikes = spikes[np.argsort(spikes[:, 0], kind="stable")]
    return spikes, np.concatenate(positions), np.asarray(sampled_means)
