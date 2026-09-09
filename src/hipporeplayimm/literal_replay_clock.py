"""Explicit physical and conditional-population-code clocks on a shared path."""

from __future__ import annotations

import hashlib

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay
from scipy.special import logsumexp

DT = 0.020
PATHS = 32
REPEATS = 5
NODES = 801
QUADRATURE = 128


def rng(*parts):
    seed = int.from_bytes(hashlib.sha256(b"literal_clock_v1|20260909|" + "|".join(map(str, parts)).encode()).digest()[:8], "little")
    return np.random.default_rng(seed)


def normalize(rates):
    rates = np.asarray(rates, dtype=float)
    if rates.ndim != 2 or not np.isfinite(rates).all() or (rates <= 0).any():
        raise ValueError("strictly positive finite rate matrix required")
    return rates / rates.sum(axis=1, keepdims=True)


class SpatialRates:
    def __init__(self, centers, rates, grid_cm=8):
        self.centers = np.asarray(centers)
        self.tri = Delaunay(centers)
        self.interpolate = LinearNDInterpolator(self.tri, np.asarray(rates).T)
        vertices = self.centers[self.tri.simplices]
        diameters = np.linalg.norm(vertices[:, :, None] - vertices[:, None, :], axis=-1).max(axis=(1, 2))
        self.local = diameters <= np.sqrt(2) * grid_cm * 1.001

    def __call__(self, points):
        ids = self.tri.find_simplex(points)
        if (ids < 0).any() or not self.local[ids].all():
            raise ValueError("path exits locally supported grid")
        values = self.interpolate(points)
        normalize(values)
        return values


def sample_geometry(spatial, generator, curved, attempts=10000):
    u = np.linspace(0, 1, NODES)
    for attempt in range(attempts):
        a, b = spatial.centers[generator.choice(len(spatial.centers), 2, replace=False)]
        delta = b - a
        distance = np.linalg.norm(delta)
        if not 40 <= distance <= 120:
            continue
        normal = np.array([-delta[1], delta[0]]) / distance
        amplitude = 0.20 * distance * generator.choice([-1, 1]) if curved else 0
        points = a + u[:, None] * delta + amplitude * np.sin(np.pi * u[:, None]) * normal
        try:
            rates = spatial(points)
        except ValueError:
            continue
        return points, rates, attempt + 1
    raise ValueError("no covered path found within frozen attempt limit")


def clock_coordinates(points, rates):
    ds = np.linalg.norm(np.diff(points, axis=0), axis=1)
    p = normalize(rates)
    dc = np.linalg.norm(np.diff(np.sqrt(p), axis=0), axis=1) / np.sqrt(2)
    if (ds <= 0).any() or dc.sum() < 1e-8 or (dc <= 0).any():
        raise ValueError("unresolved or degenerate code arc length")
    s, c = np.r_[0, ds.cumsum()], np.r_[0, dc.cumsum()]
    return s, c, {"physical": s / s[-1], "neural": c / c[-1]}


def time_samples(coordinate, values, times):
    return np.column_stack([np.interp(times, coordinate, values[:, j]) for j in range(values.shape[1])])


def bin_average(coordinate, values, n_bins, quadrature=QUADRATURE):
    nodes, weights = leggauss(quadrature)
    times = (np.arange(n_bins)[:, None] + (nodes + 1) / 2) / n_bins
    sampled = time_samples(coordinate, values, times.ravel()).reshape(n_bins, quadrature, -1)
    return np.einsum("tqc,q->tc", sampled, weights / 2)


def multinomial_samples(probabilities, totals, generator):
    p = normalize(probabilities)
    totals = np.asarray(totals)
    if totals.shape != (len(p),) or (totals < 0).any() or not np.equal(totals, np.floor(totals)).all():
        raise ValueError("nonnegative integer totals required")
    return np.array([generator.multinomial(int(n), row) for n, row in zip(totals, p, strict=True)], dtype=np.int32)


def path_log_score(counts, probabilities):
    # The multinomial coefficient cancels in every within-observation contrast.
    p = normalize(probabilities)
    if counts.shape != p.shape or (counts < 0).any():
        raise ValueError("incompatible count table")
    return float(np.sum(counts * np.log(p)))


def decode(counts, rates, centers):
    logp = np.log(normalize(np.asarray(rates).T))
    ll = counts @ logp.T
    posterior = np.exp(ll - logsumexp(ll, axis=1, keepdims=True))
    mean = posterior @ centers
    variance = np.maximum(posterior @ (centers**2).sum(axis=1) - (mean**2).sum(axis=1), 0)
    return mean, centers[posterior.argmax(axis=1)], np.sqrt(variance)


def coarse_grid(rates, centers):
    keys = np.floor((centers - centers.min(axis=0)) / 16 + 1e-9).astype(int)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    groups = [inverse == i for i in range(inverse.max() + 1)]
    return np.column_stack([rates[:, g].mean(axis=1) for g in groups]), np.vstack([centers[g].mean(axis=0) for g in groups])


def recovery_credit(delta, truth):
    directed = delta if truth == "neural" else -delta
    return 0.5 if abs(directed) < 1e-10 else float(directed > 0)
