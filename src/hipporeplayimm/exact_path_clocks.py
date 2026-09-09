"""Exhaustive integration of the finite endpoint/curve replay-clock prior."""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

from hipporeplayimm.dense_path_clocks import PathAccumulator, RateIntegral
from hipporeplayimm.literal_replay_clock import NODES, clock_coordinates, normalize


def eligible_endpoints(centers, start):
    distance = np.linalg.norm(centers - centers[start], axis=1)
    return np.flatnonzero((distance >= 40) & (distance <= 120))


def path_points(centers, start, end, curve):
    a, b = centers[start], centers[end]
    delta = b - a
    distance = np.linalg.norm(delta)
    normal = np.array([-delta[1], delta[0]]) / distance
    amplitude = 0.20 * distance * curve
    u = np.linspace(0, 1, NODES)
    return a + u[:, None] * delta + amplitude * np.sin(np.pi * u[:, None]) * normal


def geometries(spatial, start, stop):
    """Visit every oriented proposal, including unsupported ones for accounting."""
    for a in range(start, stop):
        for b in eligible_endpoints(spatial.centers, a):
            for curve in (0, -1, 1):
                points = path_points(spatial.centers, a, b, curve)
                try:
                    values = spatial(points)
                except ValueError:
                    yield (a, int(b), curve), None
                    continue
                _, _, clock = clock_coordinates(points, values)
                yield (a, int(b), curve), (RateIntegral(clock["physical"], values), RateIntegral(clock["neural"], values))


def score_geometry_block(spatial, groups, start, stop, chunk=128):
    accumulators = {n: [[PathAccumulator(x) for _ in range(2)] for _ in range(2)] for n, (_index, x) in groups.items()}
    queues, descriptors, rejected, proposed = [[], []], [], np.zeros(2, int), np.zeros(2, int)

    def flush(family):
        if not queues[family]:
            return
        for n, clocks in accumulators.items():
            for clock in range(2):
                logp = np.log(np.array([normalize(pair[clock].average(n)) for pair in queues[family]]))
                clocks[family][clock].add(logp)
        queues[family].clear()

    for descriptor, integrals in geometries(spatial, start, stop):
        family = int(descriptor[2] != 0)
        proposed[family] += 1
        if integrals is None:
            rejected[family] += 1
            continue
        descriptors.append(descriptor)
        queues[family].append(integrals)
        if len(queues[family]) == chunk:
            flush(family)
    for family in range(2):
        flush(family)
    result = {
        "descriptors": np.asarray(descriptors, dtype=np.int32).reshape(-1, 3),
        "proposed": proposed,
        "rejected": rejected,
        "accepted": proposed - rejected,
        "start_stop": np.array([start, stop]),
    }
    for n, families in accumulators.items():
        for f in range(2):
            for c in range(2):
                acc = families[f][c]
                if acc.n_paths != result["accepted"][f]:
                    raise ValueError("incomplete geometry block")
                result[f"coherent_{f}_{c}_{n}"] = acc.coherent
                result[f"reset_{f}_{c}_{n}"] = acc.reset
    return result


def merge_blocks(blocks, groups, static_rates, n_observations):
    totals = np.zeros(2, np.int64)
    sums = {}
    for block in blocks:
        totals += block["accepted"]
        for key in block.files if hasattr(block, "files") else block:
            if key.startswith(("coherent_", "reset_")):
                sums[key] = np.logaddexp(sums.get(key, -np.inf), block[key])
    if (totals <= 0).any():
        raise ValueError("both complete geometry families required")
    scores = np.full((n_observations, 5), np.nan)
    for n, (index, counts) in groups.items():
        for c in range(2):
            coherent = [sums[f"coherent_{f}_{c}_{n}"] - np.log(totals[f]) for f in range(2)]
            reset = [sums[f"reset_{f}_{c}_{n}"] - np.log(totals[f]) for f in range(2)]
            scores[index, c] = np.logaddexp(*coherent) - np.log(2)
            scores[index, c + 3] = (np.logaddexp(*reset) - np.log(2)).sum(axis=1)
        logp = np.log(normalize(static_rates.T))
        scores[index, 2] = logsumexp(counts.sum(axis=1) @ logp.T, axis=1) - np.log(len(logp))
    if not np.isfinite(scores).all():
        raise ValueError("incomplete exhaustive scores")
    return scores, totals
