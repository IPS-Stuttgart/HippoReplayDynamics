"""Fixed-tuning Poisson trains and spike-selection counterfactuals."""
from __future__ import annotations

import hashlib
import importlib.util

import numpy as np


def stable_seed(*parts):
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "little")


def load_detector(path):
    spec = importlib.util.spec_from_file_location("frozen_pf_mua_detector", str(path))
    if spec is None or spec.loader is None:
        raise ValueError("detector not loadable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def gain_profile(peak, epoch_s=2., bin_s=.001):
    if peak < 1 or not np.isfinite(peak):
        raise ValueError("gain peak must be finite and >=1")
    edges = np.arange(round(epoch_s/bin_s)+1)*bin_s
    centers = (edges[:-1]+edges[1:])/2
    gain = 1+(peak-1)*np.exp(-.5*((centers-epoch_s/2)/.04)**2)
    return edges, gain


def poisson_train(rates, ids, states, gain, rng, epoch_s=2.):
    """Sample independent cell counts and their inhomogeneous Poisson times."""
    r, ids, states = np.asarray(rates), np.asarray(ids), np.asarray(states, int)
    if r.ndim != 2 or len(ids) != len(r) or np.any(r <= 0) or not np.isfinite(r).all():
        raise ValueError("invalid fixed rate maps")
    if np.any(states < 0) or np.any(states >= r.shape[1]):
        raise ValueError("invalid states")
    g = np.asarray(gain, float)
    if g.ndim != 1 or not len(g) or np.any(g <= 0) or not np.isfinite(g).all():
        raise ValueError("invalid gain profile")
    dt = epoch_s/len(g)
    exposure = g.sum()*dt
    cdf = np.cumsum(g)/g.sum()
    cdf[-1] = 1.
    chunks = []
    for epoch, state in enumerate(states):
        counts = rng.poisson(r[:, state]*exposure)
        cells = np.repeat(ids, counts)
        bins = np.searchsorted(cdf, rng.random(len(cells)))
        times = epoch*epoch_s+(bins+rng.random(len(cells)))*dt
        chunks.append(np.column_stack((times, cells))[np.argsort(times, kind="stable")])
    return np.concatenate(chunks) if chunks else np.empty((0, 2))


def detector_endpoints(events, epoch_s=2.):
    windows, event_ids, crossing = [], [], 0
    for j, event in enumerate(events):
        a, b = event["event_start_s"], event["event_end_s"]
        n = int(np.floor((b-a)/.005+1e-8))
        if n < 4:
            raise ValueError("detector returned less than 20 ms")
        end = a+n*.005
        if abs(end-b) <= 1e-9:
            end = b
        start = a+(n-4)*.005
        first = int(np.floor(start/epoch_s+1e-10))
        last = int(np.floor((end-1e-9)/epoch_s))
        if first != last:
            crossing += 1
            continue
        windows.append([start, end])
        event_ids.append(j)
    return np.asarray(windows).reshape(-1, 2), np.asarray(event_ids, int), crossing


def select_indices(keys, maximum, *seed_parts):
    if maximum < 1:
        raise ValueError("positive cap required")
    ranked = sorted(range(len(keys)), key=lambda i: (stable_seed(*seed_parts, int(keys[i])), int(keys[i])))
    return np.asarray(sorted(ranked[:maximum]), int)


def window_means(windows, states, rates, gain, epoch_s=2.):
    w = np.asarray(windows, float).reshape(-1, 2)
    if np.any(w[:, 1] <= w[:, 0]) or np.any(w < 0):
        raise ValueError("invalid windows")
    epochs = np.floor(w[:, 0]/epoch_s+1e-10).astype(int)
    last = np.floor((w[:, 1]-1e-9)/epoch_s).astype(int)
    if np.any(epochs != last) or np.any(epochs >= len(states)):
        raise ValueError("window crosses latent-state boundary")
    edges = np.arange(len(gain)+1)*(epoch_s/len(gain))
    primitive = np.r_[0., np.cumsum(gain)*(epoch_s/len(gain))]
    a = w[:, 0]-epochs*epoch_s
    b = w[:, 1]-epochs*epoch_s
    exposure = np.interp(b, edges, primitive)-np.interp(a, edges, primitive)
    latent = np.asarray(states)[epochs]
    return rates[:, latent].T*exposure[:, None], exposure, latent
