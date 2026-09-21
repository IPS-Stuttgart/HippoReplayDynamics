"""Count-conditioned, separate-neuron spatial-mixture identifiability pilot.

This is SIMULATION CALIBRATION, not a real replay classifier. Optionally use
Pfeiffer/Foster RUN place fields through the existing hipporeplayimm loader.
Nothing scores real replay content or changes any frozen manuscript evidence.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import logsumexp


MODELS = ("single", "mixture", "serial")
GENERATORS = ("single", "mixture", "serial", "single_gain_drift", "unresolved_serial")


def probability_rows(rates: np.ndarray) -> np.ndarray:
    """Normalize each (time x neuron) intensity array, given its total count."""
    r = np.asarray(rates, dtype=float)
    if r.ndim != 3 or min(r.shape) < 1 or not np.isfinite(r).all() or np.any(r <= 0):
        raise ValueError("rates must be finite, strictly positive: states x phases x cells")
    flat = r.reshape(r.shape[0], -1)
    return flat / flat.sum(axis=1, keepdims=True)


def log_predictive(qa: np.ndarray, qb: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Integrate latent uncertainty before scoring the JOINT held-out vector.

    The uniform latent prior and held-out multinomial coefficient cancel in
    paired model contrasts. Totals in A and B are conditioned on, not predicted.
    This is same-window cross-neuron prediction, not a future forecast.
    """
    if qa.shape[0] != qb.shape[0]:
        raise ValueError("inference and evaluation must have identical latent support")
    for q in (qa, qb):
        if q.ndim != 2 or not np.isfinite(q).all() or np.any(q <= 0):
            raise ValueError("invalid probability matrix")
        if not np.allclose(q.sum(axis=1), 1, atol=1e-12):
            raise ValueError("probability rows must sum to one")
    for counts, q in ((a, qa), (b, qb)):
        if counts.ndim != 2 or counts.shape[1] != q.shape[1]:
            raise ValueError("count and probability shapes disagree")
        if not np.isfinite(counts).all() or np.any(counts < 0) or np.any(counts != np.floor(counts)):
            raise ValueError("counts must be finite nonnegative integers")
    if a.shape[0] != b.shape[0]:
        raise ValueError("paired count matrices must have the same number of windows")
    la = a @ np.log(qa).T
    lb = b @ np.log(qb).T
    return logsumexp(la + lb, axis=1) - logsumexp(la, axis=1)


def make_library(rates: np.ndarray, xy: np.ndarray, max_anchors: int = 32,
                 min_separation: float = 40.0) -> dict[str, np.ndarray]:
    """Keep ALL single states; choose pair anchors geometrically, without spikes."""
    rates = np.asarray(rates, float)
    xy = np.asarray(xy, float)
    if rates.ndim != 2 or xy.shape != (rates.shape[0], 2):
        raise ValueError("expected states x cells rates and states x 2 coordinates")
    if rates.shape[1] < 4 or np.any(rates <= 0) or not np.isfinite(rates).all() or not np.isfinite(xy).all():
        raise ValueError("need four cells and finite, positive rates/finite geometry")
    if max_anchors < 2 or min_separation <= 0:
        raise ValueError("invalid pair-library settings")
    anchors = [0]
    distance = ((xy - xy[0]) ** 2).sum(axis=1)
    for _ in range(min(max_anchors, len(xy)) - 1):
        index = int(np.argmax(distance))
        if distance[index] <= 0:
            break
        anchors.append(index)
        distance = np.minimum(distance, ((xy - xy[index]) ** 2).sum(axis=1))
    pairs = [(a, b) for i, a in enumerate(anchors) for b in anchors[i + 1:]
             if np.linalg.norm(xy[a] - xy[b]) >= min_separation]
    if not pairs:
        raise ValueError("no sufficiently separated pair anchors")
    ab = np.asarray(pairs, int)
    pair_rates = np.stack((rates[ab[:, 0]], rates[ab[:, 1]]), axis=1)
    mixed = np.repeat(pair_rates.mean(axis=1, keepdims=True), 2, axis=1)
    serial = np.concatenate((pair_rates, pair_rates[:, ::-1]), axis=0)
    return {"single": np.repeat(rates[:, None, :], 2, axis=1),
            "mixture": mixed, "serial": serial}


def generate(library: dict[str, np.ndarray], kind: str, n: int, total_a: int,
             total_b: int, aidx: np.ndarray, bidx: np.ndarray,
             rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    base = "single" if kind == "single_gain_drift" else "mixture" if kind == "unresolved_serial" else kind
    source = library[base]
    rates = source[rng.integers(len(source), size=n)].copy()
    if kind == "single_gain_drift":
        # A deliberately misspecified single-source control: unit-specific,
        # within-window constant RUN-to-event gain, not a second location.
        rates *= np.exp(rng.normal(0, 0.5, (n, 1, rates.shape[2])))
    pa = probability_rows(rates[:, :, aidx])
    pb = probability_rows(rates[:, :, bidx])
    ca = np.asarray([rng.multinomial(total_a, p) for p in pa])
    cb = np.asarray([rng.multinomial(total_b, p) for p in pb])
    return ca.reshape(n, 2, len(aidx)), cb.reshape(n, 2, len(bidx))


def score(library: dict[str, np.ndarray], aidx: np.ndarray, bidx: np.ndarray,
          a: np.ndarray, b: np.ndarray, fine: bool) -> dict[str, np.ndarray]:
    ca = a.reshape(len(a), -1) if fine else a.sum(axis=1)
    cb = b.reshape(len(b), -1) if fine else b.sum(axis=1)
    output = {}
    for name, intensity in library.items():
        r = intensity if fine else intensity.sum(axis=1, keepdims=True)
        output[name] = log_predictive(probability_rows(r[:, :, aidx]),
                                     probability_rows(r[:, :, bidx]), ca, cb)
    return output


def self_test() -> dict[str, Any]:
    # Coarse time integration makes equal-duty serial and mixed sources EXACTLY
    # observationally equivalent, including joint held-out-vector likelihoods.
    rng = np.random.default_rng(1)
    xy = np.array([[0, 0], [0, 80], [80, 0], [80, 80]], float)
    lib = make_library(np.exp(rng.normal(size=(4, 8))), xy, 4)
    a, b = generate(lib, "mixture", 20, 10, 12, np.arange(4), np.arange(4, 8), rng)
    scores = score(lib, np.arange(4), np.arange(4, 8), a, b, False)
    alias_error = float(np.max(np.abs(scores["mixture"] - scores["serial"])))
    assert alias_error < 1e-10
    # Enumerate the full held-out vector: latent ambiguity is NOT replaced by
    # independent averaged single-spike probabilities.
    qa = np.array([[.9, .1], [.1, .9]])
    qb = np.array([[.8, .2], [.2, .8]])
    aa, bb = np.array([[1, 0]]), np.array([[1, 1]])
    actual = float(log_predictive(qa, qb, aa, bb)[0])
    expected = float(np.log((.9 * .8 * .2 + .1 * .2 * .8) / (.9 + .1)))
    assert abs(actual - expected) < 1e-12
    averaged_marginals = float(np.log(.9 * .8 + .1 * .2) + np.log(.9 * .2 + .1 * .8))
    assert abs(actual - averaged_marginals) > .01
    # No held-out observations means no predictive information.
    z = log_predictive(qa, qb, aa, np.zeros((1, 2), int))
    assert abs(z[0]) < 1e-12
    # Joint permutation of cells leaves the score unchanged.
    perm = [1, 0]
    assert np.allclose(log_predictive(qa[:, perm], qb[:, perm], aa[:, perm], bb[:, perm]), actual)
    try:
        log_predictive(qa, qb, np.array([[.2, 0]]), bb)
    except ValueError:
        pass
    else:
        raise AssertionError("fractional count accepted")
    return {"passed": True, "coarse_serial_mixture_max_error_nats": alias_error,
            "joint_vector_exact_enumeration_error_nats": abs(actual - expected),
            "checks": ["coarse_alias", "joint_enumeration", "not_product_of_marginals",
                       "zero_target", "cell_permutation", "invalid_count"]}


def calibrate(name: str, rates: np.ndarray, xy: np.ndarray, repeats: int,
              seed: int, counts: tuple[int, ...], max_anchors: int) -> tuple[list[dict], list[dict], dict]:
    if repeats < 40 or repeats % 2:
        raise ValueError("repeats must be even and at least 40")
    rng = np.random.default_rng(seed)
    lib = make_library(rates, xy, max_anchors)
    order = rng.permutation(rates.shape[1])
    aidx, bidx = np.array_split(order, 2)
    raw, summary = [], []
    ncal = repeats // 2
    alias_max = 0.0
    for total in counts:
        fine_gaps, coarse_gaps, serial_gaps = {}, {}, {}
        for kind in GENERATORS:
            a, b = generate(lib, kind, repeats, total, total, aidx, bidx, rng)
            coarse = score(lib, aidx, bidx, a, b, False)
            fine = score(lib, aidx, bidx, a, b, True)
            alias_max = max(alias_max, float(np.max(np.abs(coarse["mixture"] - coarse["serial"]))))
            fine_gaps[kind] = fine["mixture"] - np.maximum(fine["single"], fine["serial"])
            coarse_gaps[kind] = coarse["mixture"] - coarse["single"]
            serial_gaps[kind] = fine["serial"] - fine["mixture"]
            for j in range(repeats):
                raw.append({"encoder": name, "generator": kind, "count_per_half_population": total,
                            "replicate": j, "partition": "calibration" if j < ncal else "evaluation",
                            "coarse_mixture_minus_single_nats": float(coarse_gaps[kind][j]),
                            "fine_mixture_minus_best_single_or_serial_nats": float(fine_gaps[kind][j]),
                            "fine_serial_minus_mixture_nats": float(serial_gaps[kind][j])})
        # Thresholds use only the first independent simulation half, not evaluated
        # outcomes. Counts/library/models are frozen; no search for best settings.
        coarse_threshold = max(0.0, float(np.quantile(coarse_gaps["single"][:ncal], .95, method="higher")))
        fine_threshold = max(0.0, *(float(np.quantile(fine_gaps[k][:ncal], .95, method="higher"))
                                      for k in ("single", "serial")))
        for kind in GENERATORS:
            c, f = coarse_gaps[kind][ncal:], fine_gaps[kind][ncal:]
            summary.append({"encoder": name, "generator": kind, "count_per_half_population": total,
                            "evaluation_windows": len(c), "coarse_threshold_nats": coarse_threshold,
                            "fine_threshold_nats": fine_threshold,
                            "coarse_positive_fraction": float(np.mean(c > coarse_threshold)),
                            "fine_positive_fraction": float(np.mean(f > fine_threshold)),
                            "mean_coarse_mixture_minus_single_nats": float(c.mean()),
                            "mean_fine_mixture_minus_best_alternative_nats": float(f.mean())})
    return raw, summary, {"encoder": name, "n_cells": int(rates.shape[1]),
                          "n_single_states": len(lib["single"]), "n_pair_states": len(lib["mixture"]),
                          "a_neuron_indices": aidx.tolist(), "b_neuron_indices": bidx.tolist(),
                          "rates_sha256": hashlib.sha256(np.ascontiguousarray(rates).tobytes()).hexdigest(),
                          "max_coarse_serial_alias_error_nats": alias_max}


def toy_encoder() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(62)
    xy = np.stack(np.meshgrid(np.linspace(0, 160, 9), np.linspace(0, 160, 9)), -1).reshape(-1, 2)
    centers = rng.uniform(0, 160, (48, 2))
    distance = ((xy[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    rates = .1 + rng.uniform(3, 15, (1, 48)) * np.exp(-distance / (2 * 24 ** 2))
    return rates, xy


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--sessions", default="Rat1/Open1,Rat2/Open1,Rat3/Open1,Rat4/Open1")
    parser.add_argument("--repeats", type=int, default=256)
    parser.add_argument("--counts", default="4,12,32")
    parser.add_argument("--max-anchors", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260921)
    args = parser.parse_args()
    counts = tuple(int(c) for c in args.counts.split(","))
    if not counts or min(counts) <= 0:
        raise ValueError("counts must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    test = self_test()
    all_raw, all_summary, provenance = [], [], []
    specs = ["synthetic_place_fields"] if args.dataset_root is None else args.sessions.split(",")
    for i, name in enumerate(specs):
        if args.dataset_root is None:
            rates, xy = toy_encoder()
        else:
            from hipporeplayimm.data import load_replay_session
            from hipporeplayimm.encoding import EncodingConfig, fit_place_field_encoding
            session = load_replay_session(args.dataset_root / name)
            encoding = fit_place_field_encoding(session, EncodingConfig(bin_size_cm=6.0,
                        smoothing_sigma_bins=2.0, min_speed_cm_s=10.0, exclude_ripple_intervals=True))
            valid = np.asarray(encoding.occupancy_s) >= .02
            if valid.sum() < 4:
                raise ValueError(f"{name}: insufficient RUN spatial support")
            rates = np.asarray(encoding.rates_hz)[:, valid].T
            xy = np.asarray(encoding.bin_centers)[valid]
        raw, summary, metadata = calibrate(name, rates, xy, args.repeats, args.seed + i,
                                            counts, args.max_anchors)
        all_raw.extend(raw)
        all_summary.extend(summary)
        provenance.append(metadata)
        write_csv(args.output / "simulation_scores.csv", all_raw)
        write_csv(args.output / "simulation_summary.csv", all_summary)
        print(json.dumps({"completed": name, "n_cells": metadata["n_cells"],
                          "n_single_states": metadata["n_single_states"],
                          "coarse_alias_max_error": metadata["max_coarse_serial_alias_error_nats"]}), flush=True)
    manifest = {"scope": "simulation_only; empirical RUN encoders when requested; no real replay scores",
                "not_evidence_of": ["biological simultaneous replay", "independent animal replication",
                                    "general robustness to unknown within-bin switching"],
                "score": "joint time/cell identity vector in B conditional on A and separate whole-window totals",
                "latent_prior": "uniform within each finite library; explicit normalization; no MAP plug-in",
                "generator_note": "matched model-family/library is favorable recovery, not external validation",
                "fine_resolution": "two equal subwindows; serial unresolved within each subwindow is identical to mixture",
                "gain_drift_control": "single location with per-window unit gains exp(N(0,0.5^2))",
                "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "python": platform.python_version(), "numpy": np.__version__,
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "self_tests": test, "encoders": provenance}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"self_tests": test, "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
