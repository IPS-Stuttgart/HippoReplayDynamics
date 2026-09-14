#!/usr/bin/env python3
"""Frozen regional-mixture development screen; never selects replay events."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit, logsumexp

from scripts._provenance import build_script_provenance, file_sha256

SESSIONS = ("Rat1/Open1", "Rat1/Open2", "Rat2/Open1", "Rat4/Open2")
REAL = ("all_fixed_candidates", "full_accepted_segment")
TRUTH = ("run_q4", "test_poisson_gain1", "test_poisson_gain4", "test_conditional", "test_conditional_map_drift", "test_conditional_shared_assembly")
PREVALENCES = (0.05, 0.15, 0.30, 0.50, 0.75)


def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def regional_log_bf(counts, rates, near, conditional=False):
    counts, rates, near = np.asarray(counts), np.asarray(rates), np.asarray(near, bool)
    if (
        counts.ndim != 2
        or rates.ndim != 2
        or counts.shape[1] != rates.shape[0]
        or near.shape != (rates.shape[1],)
        or not near.any()
        or near.all()
        or not np.isfinite(counts).all()
        or np.any(counts < 0)
        or not np.isfinite(rates).all()
        or np.any(rates <= 0)
    ):
        raise ValueError("invalid counts, encoding, or regional mask")
    ll = counts @ np.log(rates)
    if conditional:
        ll -= counts.sum(axis=1)[:, None] * np.log(rates.sum(axis=0))[None, :]
    else:
        ll -= 0.02 * rates.sum(axis=0)[None, :]
    return logsumexp(ll[:, near], axis=1) - np.log(near.sum()) - logsumexp(ll[:, ~near], axis=1) + np.log((~near).sum())


def fit_mixture(log_bf, weights=None):
    b = np.asarray(log_bf, float)
    w = np.ones_like(b) if weights is None else np.asarray(weights, float)
    if b.ndim != 1 or not b.size or w.shape != b.shape or not np.isfinite(b).all() or not np.isfinite(w).all() or np.any(w < 0) or w.sum() <= 0:
        raise ValueError("finite nonempty likelihoods and nonnegative weights required")
    active = w > 0
    b, w = b[active], w[active] / w.sum()
    n_effective = 1 / np.dot(w, w)
    if np.max(np.abs(b)) <= 1e-12:
        return dict(
            estimate=np.nan, fit_status="nonidentified", score=np.nan, information=0.0, profile_low=0.0, profile_high=1.0, profile_width=1.0, effective_observations=n_effective
        )
    a, c = np.exp(np.minimum(b, 0)), np.exp(np.minimum(-b, 0))

    def derivative(p):
        with np.errstate(over="ignore", divide="ignore"):
            if p == 0:
                return float(np.dot(w, np.expm1(b)))
            if p == 1:
                return float(np.dot(w, -np.expm1(-b)))
            return float(np.dot(w, (a - c) / (p * a + (1 - p) * c)))

    def ll(p):
        with np.errstate(divide="ignore"):
            terms = np.logaddexp(np.log(p) + b, np.log1p(-p)) - np.maximum(b, 0)
        return float(np.dot(w, terms))

    if derivative(0) <= 0:
        p, status = 0.0, "boundary_zero"
    elif derivative(1) >= 0:
        p, status = 1.0, "boundary_one"
    else:
        p, status = brentq(derivative, 0, 1, xtol=1e-13), "interior"
    # Conditional profile support only: neither clustering nor model error is covered.
    top = ll(p)

    def cut(q):
        return 2 * n_effective * (top - ll(q)) - 3.841458820694124

    low = brentq(cut, 0, p) if p > 0 and cut(0) > 0 else 0.0
    high = brentq(cut, p, 1) if p < 1 and cut(1) > 0 else 1.0
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        information = float(np.dot(w, ((a - c) / (p * a + (1 - p) * c)) ** 2))
    return dict(
        estimate=float(p),
        fit_status=status,
        score=derivative(p),
        information=information,
        profile_low=low,
        profile_high=high,
        profile_width=high - low,
        effective_observations=n_effective,
    )


def panel_weights(labels, prevalence=None):
    labels = np.asarray(labels, bool)
    if not labels.any() or labels.all():
        raise ValueError("both truth classes required")
    if prevalence is None:
        return np.ones(len(labels)) / len(labels)
    if not 0 < prevalence < 1:
        raise ValueError("panel prevalence must be interior")
    return np.where(labels, prevalence / labels.sum(), (1 - prevalence) / (~labels).sum())


def summarize(estimates):
    """No event pooling across rats; missing estimates cause gates to fail."""
    d = estimates.copy()
    known = d[d.source.isin(TRUTH)].copy()
    known["raw_error"] = abs(known.raw_posterior_mass - known.true_prevalence)
    known["mixture_error"] = abs(known.estimate - known.true_prevalence)
    keys = ["observation", "encoding", "source", "panel_type", "animal", "session"]
    metrics = ["raw_error", "mixture_error"]
    session_truth = known.groupby(keys, dropna=False)[metrics].mean().reset_index()
    animal_truth = session_truth.groupby(keys[:-1], dropna=False)[metrics].mean().reset_index()
    truth_summary = animal_truth.groupby(keys[:4], dropna=False)[metrics].mean().reset_index()
    truth_summary["relative_error_reduction"] = 1 - truth_summary.mixture_error / truth_summary.raw_error
    pair_rows = []
    for key, g in d[d.source.isin(REAL)].groupby(keys[:3] + ["animal", "session"]):
        if len(g) != 2 or set(g.side) != {"high", "low"}:
            raise ValueError("missing or duplicate original population")
        hi, lo = g.set_index("side").loc[["high", "low"]].to_dict("records")
        pair_rows.append(
            dict(zip(keys[:3] + ["animal", "session"], key, strict=True))
            | dict(
                raw_signed_gap=hi["raw_posterior_mass"] - lo["raw_posterior_mass"],
                mixture_signed_gap=hi["estimate"] - lo["estimate"],
                raw_gap=abs(hi["raw_posterior_mass"] - lo["raw_posterior_mass"]),
                mixture_gap=abs(hi["estimate"] - lo["estimate"]),
            )
        )
    pair = pd.DataFrame(pair_rows)
    gap_metrics = ["raw_signed_gap", "mixture_signed_gap", "raw_gap", "mixture_gap"]
    animal_gap = pair.groupby(keys[:3] + ["animal"])[gap_metrics].mean().reset_index()
    gaps = animal_gap.groupby(keys[:3])[gap_metrics].mean().reset_index()
    gaps["relative_gap_reduction"] = 1 - gaps.mixture_gap / gaps.raw_gap
    gates = []

    def gate(name, passed, detail):
        gates.append(dict(gate=name, passed=bool(passed), detail=detail))

    expected = {
        (s, o, e, source, side, panel)
        for s in SESSIONS
        for o in ("poisson", "conditional")
        for source in REAL + TRUTH
        for e in (("early_run", "full_run") if source in REAL else ("early_run",))
        for side in ("high", "low")
        for panel in (("real",) if source in REAL else tuple(f"p{p:.2f}" for p in PREVALENCES) + (("natural",) if source == "run_q4" else ()))
    }
    actual = set(d[["session", "observation", "encoding", "source", "side", "panel"]].itertuples(index=False, name=None))
    gate("all_expected_estimates_present", actual == expected and len(d) == len(expected), f"rows={len(d)} expected={len(expected)}")
    gate("all_estimates_identified", np.isfinite(d.estimate).all(), "No missing estimate is discarded")
    for source in TRUTH:
        s = truth_summary[(truth_summary.observation == "poisson") & (truth_summary.source == source) & (truth_summary.panel_type == "fixed_prevalence")]
        a = animal_truth[(animal_truth.observation == "poisson") & (animal_truth.source == source) & (animal_truth.panel_type == "fixed_prevalence")]
        no_harm = len(a) == 3 and np.isfinite(a.mixture_error).all() and (a.mixture_error <= a.raw_error + 1e-10).all()
        strong = source in ("run_q4", "test_poisson_gain1")
        passed = len(s) == 1 and no_harm
        if strong and len(s) == 1:
            passed = passed and s.iloc[0].mixture_error <= 0.05 and s.iloc[0].relative_error_reduction >= 0.20
        gate(f"truth_{source}", passed, s.to_json(orient="records"))
    natural = animal_truth[(animal_truth.observation == "poisson") & (animal_truth.panel_type == "natural")]
    gate(
        "natural_run_no_harm",
        len(natural) == 3 and np.isfinite(natural.mixture_error).all() and (natural.mixture_error <= natural.raw_error + 1e-10).all(),
        natural.to_json(orient="records"),
    )
    for encoding in ("early_run", "full_run"):
        for source in REAL:
            g = gaps[(gaps.observation == "poisson") & (gaps.encoding == encoding) & (gaps.source == source)]
            a = animal_gap[(animal_gap.observation == "poisson") & (animal_gap.encoding == encoding) & (animal_gap.source == source)]
            reduction = 0.20 if source == "all_fixed_candidates" else -1e-10
            passed = len(g) == 1 and len(a) == 3 and np.isfinite(a.mixture_gap).all()
            passed = passed and (a.mixture_gap <= a.raw_gap + 1e-10).all() and g.iloc[0].relative_gap_reduction >= reduction
            gate(f"content_{encoding}_{source}", passed, g.to_json(orient="records"))
    r = d[(d.observation == "poisson") & d.source.isin(REAL)]
    gate("real_estimates_interior", len(r) == 32 and r.fit_status.eq("interior").all(), r.fit_status.value_counts().to_json())
    gate("development_numerical_screen", all(g["passed"] for g in gates), "Independent reconstruction is additionally required; never an external-validation claim")
    return dict(
        session_truth=session_truth,
        animal_truth=animal_truth,
        truth_summary=truth_summary,
        pair_contrasts=pair,
        animal_contrasts=animal_gap,
        content_summary=gaps,
        gate_summary=pd.DataFrame(gates),
    )


def measure(args):
    inputs = {}

    def checked(path, expected=None):
        path = Path(path)
        value = file_sha256(path)
        if value is None or (expected is not None and value != expected):
            raise ValueError(f"missing/changed source {path}")
        inputs[str(path.resolve())] = value
        return path

    audit = json.loads(checked(args.source_audit).read_text())
    if audit.get("status") != "pass" or {x["session"] for x in audit["results"] if x["status"] == "pass"} != set(SESSIONS):
        raise ValueError("source reconstruction not complete")
    checked(args.source_dir / "manifest.json", audit["input_file_sha256"]["source"])
    checked(ROOT / "docs/regional_likelihood_mixture_protocol.md")
    checked(Path(__file__))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    rows, reconstruction = [], []
    for session in SESSIONS:
        folder = args.source_dir / session.replace("/", "_")
        hashes = json.loads(checked(folder / "outputs.json").read_text())
        for name, digest in hashes.items():
            checked(folder / name, digest)
        encoding = load_npz(folder / "encoding.npz")
        near, prior = encoding["near"], float(encoding["near"].mean())
        for source in REAL + TRUTH:
            data = load_npz(folder / f"{source}.npz")
            for enc in ("early_run", "full_run") if source in REAL else ("early_run",):
                for observation in ("poisson", "conditional"):
                    b = np.column_stack(
                        [
                            regional_log_bf(data["counts"][:, encoding[f"{side}_indices"]], encoding[enc][encoding[f"{side}_indices"]], near, observation == "conditional")
                            for side in ("high", "low")
                        ]
                    )
                    s = expit(b + np.log(prior / (1 - prior)))
                    if observation == "poisson":
                        np.testing.assert_allclose(s, data[f"{enc}_scores"], atol=1e-9, rtol=1e-9)
                        reconstruction.append(dict(session=session, source=source, encoding=enc, maximum_error=float(abs(s - data[f"{enc}_scores"]).max())))
                    name = f"{session.replace('/', '_')}__{source}__{enc}__{observation}.npz"
                    np.savez_compressed(args.output_dir / name, log_bf=b, raw_posterior=s)
                    panels = (
                        [("real", "real", None, np.ones(len(b)) / len(b))]
                        if source in REAL
                        else [(f"p{p:.2f}", "fixed_prevalence", p, panel_weights(data["labels"], p)) for p in PREVALENCES]
                    )
                    if source == "run_q4":
                        panels.append(("natural", "natural", float(data["labels"].mean()), panel_weights(data["labels"])))
                    for panel, panel_type, truth, weights in panels:
                        for j, side in enumerate(("high", "low")):
                            row = dict(
                                session=session,
                                animal=session.split("/")[0],
                                source=source,
                                encoding=enc,
                                observation=observation,
                                side=side,
                                panel=panel,
                                panel_type=panel_type,
                                true_prevalence=truth,
                                n_observations=len(b),
                                area_prior=prior,
                                raw_posterior_mass=float(weights @ s[:, j]),
                                **fit_mixture(b[:, j], weights),
                            )
                            rows.append(row)
        print(f"completed {session}; {len(rows)} cumulative estimates", flush=True)
    estimates = pd.DataFrame(rows)
    estimates.to_csv(args.output_dir / "estimates.csv", index=False)
    tables = summarize(estimates)
    for name, table in tables.items():
        table.to_csv(args.output_dir / f"{name}.csv", index=False)
    for path, digest in inputs.items():
        checked(path, digest)
    manifest = dict(
        **build_script_provenance(),
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        source_dir=str(args.source_dir.resolve()),
        source_audit=str(args.source_audit.resolve()),
        input_sha256=inputs,
        output_sha256={p.name: file_sha256(p) for p in args.output_dir.iterdir()},
        baseline_reconstruction=reconstruction,
        independent_audit="pending",
        claim="development only; independent data and true replay content not validated",
    )
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    measure(parser.parse_args())


if __name__ == "__main__":
    main()
