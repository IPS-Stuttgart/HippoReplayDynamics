"""Independent event paths sampled from an audited exhaustive finite prior."""

from __future__ import annotations

import hashlib

import numpy as np

from hipporeplayimm.dense_path_clocks import RateIntegral
from hipporeplayimm.exact_path_clocks import path_points
from hipporeplayimm.literal_replay_clock import clock_coordinates, normalize
from hipporeplayimm.unknown_path_clocks import MODELS


def rng(*parts):
    digest = hashlib.sha256(("fresh_path_clocks_v1|20260910|" + "|".join(map(str, parts))).encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def family_indices(descriptors):
    d = np.asarray(descriptors)
    if d.ndim != 2 or d.shape[1] != 3 or not np.isin(d[:, 2], [-1, 0, 1]).all():
        raise ValueError("finite-prior path descriptors required")
    families = (np.flatnonzero(d[:, 2] == 0), np.flatnonzero(d[:, 2] != 0))
    if min(map(len, families)) == 0:
        raise ValueError("both geometry families required")
    return families


def draw_path(families, generator):
    family = families[generator.integers(2)]
    return int(family[generator.integers(len(family))])


def generate_event(totals, model, spatial, rates, descriptors, families, generator):
    totals = np.asarray(totals)
    if totals.ndim != 1 or len(totals) < 1 or (totals < 0).any() or not np.equal(totals, np.floor(totals)).all() or model not in MODELS:
        raise ValueError("nonempty integer bin totals and known generator required")
    n = len(totals)
    if model == "stationary":
        latent = np.full(n, generator.integers(rates.shape[1]), dtype=np.int32)
        probabilities = normalize(rates[:, latent].T)
    else:
        latent = np.array([draw_path(families, generator) for _ in range(n if model.endswith("reset") else 1)], dtype=np.int32)
        if len(latent) == 1:
            latent = np.repeat(latent, n)
        probabilities = np.empty((n, rates.shape[0]))
        clock_name = "neural" if model.startswith("neural") else "physical"
        for index in np.unique(latent):
            points = path_points(spatial.centers, *descriptors[index])
            values = spatial(points)
            _, _, clocks = clock_coordinates(points, values)
            p = normalize(RateIntegral(clocks[clock_name], values).average(n))
            mask = latent == index
            probabilities[mask] = p[mask]
    counts = np.array([generator.multinomial(int(total), p) for total, p in zip(totals, probabilities, strict=True)], dtype=np.int32)
    return counts, latent
