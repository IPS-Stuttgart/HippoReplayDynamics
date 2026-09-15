#!/usr/bin/env python3
"""Exact Bayesian control for class-conditional probability-score safeguards."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from fractions import Fraction as F
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts._provenance import build_script_provenance, file_sha256


def risks(prior, values, weights):
    prior = F(prior)
    q, w = tuple(map(F, values)), tuple(map(F, weights))
    if not 0 < prior < 1 or not q or len(q) != len(w):
        raise ValueError("invalid channel dimensions or prior")
    if any(not 0 <= x <= 1 for x in q) or any(x <= 0 for x in w):
        raise ValueError("invalid probabilities")
    if sum(w) != 1 or sum(a * b for a, b in zip(q, w, strict=True)) != prior:
        raise ValueError("not a posterior refinement")
    positive = sum(weight * p * (1 - p) ** 2 for weight, p in zip(w, q, strict=True)) / prior
    negative = sum(weight * (1 - p) * p**2 for weight, p in zip(w, q, strict=True)) / (1 - prior)
    total = sum(weight * p * (1 - p) for weight, p in zip(w, q, strict=True))
    variance = sum(weight * (p - prior) ** 2 for weight, p in zip(w, q, strict=True))
    if prior * positive + (1 - prior) * negative != total or prior * (1 - prior) - total != variance:
        raise AssertionError("Bayes-risk identity failed")
    return dict(
        positive_class_brier=positive,
        negative_class_brier=negative,
        overall_brier=total,
        positive_class_absolute_error=total / prior,
        negative_class_absolute_error=total / (1 - prior),
        posterior_variance=variance,
    )


def channel(prior, low, high):
    prior, low, high = map(F, (prior, low, high))
    if not 0 <= low < prior < high <= 1:
        raise ValueError("posterior endpoints do not straddle prior")
    high_weight = (prior - low) / (high - low)
    return [low, high], [1 - high_weight, high_weight]


def cases():
    prior = F(3, 100)
    yield "uninformative", prior, [prior], [F(1)]
    for name, low, high in (
        ("weak_refinement", F(0), F(3, 50)),
        ("strong_refinement_full_support", F(1, 1000), F(9, 10)),
        ("high_precision_refinement", F(0), F(99, 100)),
        ("perfect_observation", F(0), F(1)),
    ):
        values, weights = channel(prior, low, high)
        yield name, prior, values, weights


def independent_joint_check(prior, values, weights, measured):
    accum = {name: F(0) for name in ("positive_class_brier", "negative_class_brier", "overall_brier", "positive_class_absolute_error", "negative_class_absolute_error")}
    joints = []
    for state, (posterior, marginal) in enumerate(zip(values, weights, strict=True)):
        for truth in (0, 1):
            mass = marginal * (posterior if truth else 1 - posterior)
            class_mass = prior if truth else 1 - prior
            label = "positive" if truth else "negative"
            loss = (posterior - truth) ** 2
            accum[f"{label}_class_brier"] += mass * loss / class_mass
            accum[f"{label}_class_absolute_error"] += mass * abs(posterior - truth) / class_mass
            accum["overall_brier"] += mass * loss
            joints.append(dict(observation=state, truth=truth, joint_mass=mass, conditional_observation_probability=mass / class_mass, posterior=posterior))
    if any(accum[k] != measured[k] for k in accum) or sum(v["joint_mass"] for v in joints) != 1:
        raise AssertionError("joint-distribution enumeration disagrees")
    for state in range(len(values)):
        subset = [v for v in joints if v["observation"] == state]
        numerator = sum(v["joint_mass"] for v in subset if v["truth"] == 1)
        denominator = sum(v["joint_mass"] for v in subset)
        if numerator / denominator != values[state]:
            raise AssertionError("not the true Bayesian posterior")
    return joints


def run(output):
    inputs = {str(p): file_sha256(p) for p in (Path(__file__), ROOT / "scripts/_provenance.py")}
    rows, channels = [], []
    baseline = risks(F(3, 100), [F(3, 100)], [F(1)])
    for name, prior, values, weights in cases():
        measured = risks(prior, values, weights)
        joints = independent_joint_check(prior, values, weights, measured)
        row = dict(
            case=name,
            prior=float(prior),
            full_support=all(v["joint_mass"] > 0 for v in joints),
            overall_brier_nonworsening=measured["overall_brier"] <= baseline["overall_brier"],
            both_class_briers_nonworsening=all(measured[k] <= baseline[k] for k in ("positive_class_brier", "negative_class_brier")),
        )
        for key, value in measured.items():
            row[key] = float(value)
            row[f"{key}_exact"] = str(value)
        rows.append(row)
        channels.extend(dict(case=name, **{k: str(v) if isinstance(v, F) else v for k, v in entry.items()}) for entry in joints)
    strong = next(r for r in rows if r["case"] == "strong_refinement_full_support")
    assert strong["full_support"] and strong["overall_brier_nonworsening"] and not strong["both_class_briers_nonworsening"]
    output.mkdir(parents=True, exist_ok=False)
    for name, values in (("risk_comparison", rows), ("joint_distribution", channels)):
        with (output / f"{name}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    report = """# Class-conditional Brier safeguard: exact Bayesian control

This is an exact-arithmetic counterexample, not a neural-data result or a remedy.
The event of interest has prior probability 0.03. A coarse observation gives no
information, so its true posterior is always 0.03. A genuinely informative new
observation has true posterior 0.001 or 0.90. All truth/observation combinations
have positive probability, and their marginals obey the posterior-refinement
identity exactly. The complete joint distribution is provided alongside risks.

The refined posterior has lower overall Brier risk and lower mean absolute error
in BOTH classes, but higher Brier risk in the common negative class. Thus a
class-wise no-increase safeguard is stronger than improvement of an oracle
Bayesian decoder. Its failure alone does not prove that the acquired observations
are uninformative or that the decoder is incorrect. Neither does overall
improvement imply every regional probability-score safeguard passes.

For the special posterior pair {0,h}, the negative-class Brier change is exactly

  prior/(1-prior) * (h-prior) * (1-prior-h).

When prior < 1/2 and prior < h < 1-prior, this is positive even though overall
Brier risk decreases. Rare high-posterior mistakes can dominate squared loss in
the common class while performance on the rare class improves substantially.

This is a standard Bayes-risk/conditioning phenomenon, not a new theorem. The
two-location unit-distance example does not establish the mechanism of any PF
result; its absolute-error identity need not describe a multidimensional decoder.
It does not relax, reinterpret as successful, or override the frozen acquisition
gates. No replay, Q4, held-out source bank or independent recording was used.
The active goal remains unmet. Any different validation criterion would require
a separately justified prospective design, not rescue of a failed run.
"""
    (output / "report.md").write_text(report)
    if any(file_sha256(p) != sha for p, sha in inputs.items()):
        raise ValueError("source changed")
    manifest = {
        **build_script_provenance(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "input_file_sha256": inputs,
        "exact_arithmetic": "fractions.Fraction",
        "independent_joint_enumeration": True,
        "cases": len(rows),
        "validated_remedy": False,
        "neural_data_used": False,
        "external_validation": False,
        "output_sha256": {p.name: file_sha256(p) for p in output.iterdir()},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    run(parser.parse_args().output_dir)
