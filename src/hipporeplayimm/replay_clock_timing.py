"""Matched coarse/fine clock likelihoods conditional on parent-bin spike totals."""

from __future__ import annotations

import hashlib

import numpy as np

MODELS = ("coarse_identity", "fine_identity", "fine_timing", "fine_joint")
SUBBINS = 20
REPEATS = 10


def rng(*parts):
    digest = hashlib.sha256(("clock_timing_v1|20260909|" + "|".join(map(str, parts))).encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def probabilities(fine_rates):
    r = np.asarray(fine_rates, dtype=float)
    if r.ndim != 3 or not np.isfinite(r).all() or (r <= 0).any():
        raise ValueError("positive finite parent-time by subbin by cell rates required")
    joint = r / r.sum(axis=(1, 2), keepdims=True)
    timing = joint.sum(axis=2)
    identity = joint / timing[:, :, None]
    coarse = joint.sum(axis=1)
    return {"fine_joint": joint, "fine_identity": identity, "fine_timing": timing, "coarse_identity": coarse}


def sample(joint, totals, generator):
    if joint.ndim != 3 or not np.allclose(joint.sum(axis=(1, 2)), 1):
        raise ValueError("normalized parent-bin probabilities required")
    totals = np.asarray(totals)
    if totals.shape != (len(joint),) or (totals < 0).any() or not np.equal(totals, np.floor(totals)).all():
        raise ValueError("integer parent-bin totals required")
    return np.array([generator.multinomial(int(n), p.ravel()).reshape(p.shape) for n, p in zip(totals, joint, strict=True)], dtype=np.int32)


def log_scores(counts, p):
    x = np.asarray(counts)
    if x.shape != p["fine_joint"].shape or not np.isfinite(x).all() or (x < 0).any():
        raise ValueError("compatible nonnegative counts required")
    return {
        "fine_joint": float(np.sum(x * np.log(p["fine_joint"]))),
        "fine_identity": float(np.sum(x * np.log(p["fine_identity"]))),
        "fine_timing": float(np.sum(x.sum(axis=2) * np.log(p["fine_timing"]))),
        "coarse_identity": float(np.sum(x.sum(axis=1) * np.log(p["coarse_identity"]))),
    }


def signed_credit(delta, truth):
    d = delta if truth == "neural" else -delta
    return 0.5 if abs(d) < 1e-10 else float(d > 0)
