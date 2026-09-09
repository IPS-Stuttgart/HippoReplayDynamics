"""Finite path-marginalized clock likelihoods, without observed path labels."""

from __future__ import annotations

import hashlib
from itertools import product

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from hipporeplayimm.metric_population_recovery import wilson

MODELS = ("physical", "neural", "stationary", "physical_reset", "neural_reset")
SCENARIOS = (0.25, 0.5, 0.75)
TEACHERS = ("matched", "independent")
SUPPORTS = (128, 256)
REPEATS = 50
EVENTS = 128
PATHS = 256
IDS = ["dataset", "animal", "session"]


def rng(*parts):
    digest = hashlib.sha256(("unknown_path_clocks_v1|20260909|" + "|".join(map(str, parts))).encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def exact_bin_average(coordinate, values, n_bins):
    """Integrate the piecewise-linear rate curve using its antiderivative."""
    t, y = np.asarray(coordinate, float), np.asarray(values, float)
    if t.ndim != 1 or y.ndim != 2 or len(t) != len(y) or len(t) < 2 or n_bins < 1 or int(n_bins) != n_bins:
        raise ValueError("aligned clock knots, rate columns and positive bin count required")
    if not np.isfinite(t).all() or not np.isfinite(y).all() or (np.diff(t) <= 0).any() or t[0] != 0 or t[-1] != 1:
        raise ValueError("finite strictly increasing normalized clock required")
    dt = np.diff(t)
    prefix = np.vstack([np.zeros(y.shape[1]), np.cumsum(dt[:, None] * (y[:-1] + y[1:]) / 2, axis=0)])
    edge = np.linspace(0, 1, int(n_bins) + 1)
    index = np.minimum(np.searchsorted(t, edge, side="right") - 1, len(t) - 2)
    offset = edge - t[index]
    slope = (y[index + 1] - y[index]) / dt[index, None]
    integral = prefix[index] + y[index] * offset[:, None] + slope * offset[:, None] ** 2 / 2
    return np.diff(integral, axis=0) * n_bins


def score_batch(counts, log_physical, log_neural, log_stationary, supports=SUPPORTS):
    """Counts only: no generator label or true path index reaches this scorer."""
    x = np.asarray(counts)
    banks = [np.asarray(log_physical), np.asarray(log_neural)]
    static = np.asarray(log_stationary)
    if x.ndim != 3 or x.shape[0] < 1 or (x < 0).any() or not np.equal(x, np.floor(x)).all():
        raise ValueError("nonempty event-time-cell integer counts required")
    if banks[0].ndim != 3 or banks[1].shape != banks[0].shape or banks[0].shape[1:] != x.shape[1:]:
        raise ValueError("aligned candidate path-time-cell banks required")
    if static.ndim != 2 or static.shape[1] != x.shape[2] or len(static) < 1:
        raise ValueError("aligned stationary location-cell bank required")
    if not supports or len(set(supports)) != len(supports) or any(int(h) != h or not 1 <= h <= len(banks[0]) for h in supports):
        raise ValueError("distinct nonempty candidate support sizes required")
    for bank in [*banks, static]:
        if not np.isfinite(bank).all() or not np.allclose(logsumexp(bank, axis=-1), 0, atol=1e-10):
            raise ValueError("normalized finite log probabilities required")
    stationary = logsumexp(x.sum(axis=1) @ static.T, axis=1) - np.log(len(static))
    result = {h: np.zeros((len(x), len(MODELS))) for h in supports}
    for h in supports:
        result[h][:, 2] = stationary
    for i, bank in enumerate(banks):
        # Batch matrix multiplication shares the candidate library across events.
        per_bin = (x.transpose(1, 0, 2) @ bank.transpose(1, 2, 0)).transpose(1, 2, 0)
        for h in supports:
            ll = per_bin[:, :h]
            result[h][:, i] = logsumexp(ll.sum(axis=2), axis=1) - np.log(h)
            result[h][:, i + 3] = (logsumexp(ll, axis=1) - np.log(h)).sum(axis=1)
    return result


def generate_counts(totals, model, physical, neural, stationary, generator):
    totals = np.asarray(totals)
    t = len(totals)
    if model not in MODELS or (totals < 0).any() or not np.equal(totals, np.floor(totals)).all():
        raise ValueError("known generator and integer total counts required")
    if model == "stationary":
        latent = np.repeat(generator.integers(len(stationary)), t)
        probabilities = stationary[latent]
    else:
        bank = neural if model.startswith("neural") else physical
        latent = generator.integers(len(bank), size=t) if model.endswith("reset") else np.repeat(generator.integers(len(bank)), t)
        probabilities = bank[latent, np.arange(t)]
    if probabilities.shape[0] != t or not np.isfinite(probabilities).all() or (probabilities <= 0).any() or not np.allclose(probabilities.sum(axis=1), 1):
        raise ValueError("positive normalized emission probabilities required")
    counts = np.array([generator.multinomial(int(n), p) for n, p in zip(totals, probabilities, strict=True)], dtype=np.int32)
    return counts, latent


def summarize(fits, repeats=REPEATS, supports=SUPPORTS, teachers=TEACHERS):
    keys = ["dataset", "scenario", "repeat", "teacher", "support"]
    expected = set(product(("pfeiffer_foster", "tanni2022"), SCENARIOS, range(repeats), teachers, supports))
    if fits.empty or fits.duplicated(keys).any() or set(fits[keys].itertuples(index=False, name=None)) != expected:
        raise ValueError("all distinct frozen population fits required")
    x = fits.copy()
    values = x[["phi_hat", "phi_low", "phi_high"]].to_numpy()
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any() or (x.phi_low > x.phi_high).any():
        raise ValueError("finite bounded ordered estimates required")
    x["error"] = x.phi_hat - x.scenario
    x["covered"] = (x.phi_low <= x.scenario + 1e-8) & (x.phi_high >= x.scenario - 1e-8)
    x["directional_claim"] = (x.phi_low > 0.5) | (x.phi_high < 0.5)
    x["correct_direction"] = ((x.scenario > 0.5) & (x.phi_low > 0.5)) | ((x.scenario < 0.5) & (x.phi_high < 0.5))
    rows = []
    for key, group in x.groupby(["dataset", "teacher", "support", "scenario"]):
        row = dict(zip(["dataset", "teacher", "support", "scenario"], key, strict=True))
        row.update(
            n_replicates=len(group),
            mean_estimate=group.phi_hat.mean(),
            bias=group.error.mean(),
            rmse=np.sqrt(np.mean(group.error**2)),
            median_interval_width=(group.phi_high - group.phi_low).median(),
            mean_coherent_weight=group.coherent_weight.mean(),
        )
        for col in ("covered", "directional_claim", "correct_direction"):
            count = int(group[col].sum())
            low, high = wilson(count, len(group))
            row.update({col + "_fraction": count / len(group), col + "_mc_low": low, col + "_mc_high": high})
        rows.append(row)
    summary = pd.DataFrame(rows)
    gates = []
    for key, group in summary.groupby(["dataset", "teacher", "support"]):
        null = group[group.scenario.eq(0.5)].iloc[0]
        power = group[group.scenario.ne(0.5)].correct_direction_fraction.min()
        coverage, bias = group.covered_fraction.min(), group.bias.abs().max()
        gates.append(
            dict(zip(["dataset", "teacher", "support"], key, strict=True))
            | {
                "min_direction_power": power,
                "min_coverage": coverage,
                "max_absolute_bias": bias,
                "null_false_direction_fraction": null.directional_claim_fraction,
                "practical_pass": bool(power >= 0.8 and coverage >= 0.9 and bias <= 0.1 and null.directional_claim_fraction <= 0.05),
            }
        )
    return x, summary, pd.DataFrame(gates)
