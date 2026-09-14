#!/usr/bin/env python3
"""Independent count-to-likelihood, optimizer, aggregation and gate audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import gammaln, logsumexp

SESSIONS = ("Rat1/Open1", "Rat1/Open2", "Rat2/Open1", "Rat4/Open2")
SOURCES = (
    "all_fixed_candidates",
    "full_accepted_segment",
    "run_q4",
    "test_poisson_gain1",
    "test_poisson_gain4",
    "test_conditional",
    "test_conditional_map_drift",
    "test_conditional_shared_assembly",
)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return dict(z)


def independent_likelihood(counts, rates, home, observation):
    n, mu = np.asarray(counts, float), 0.02 * np.asarray(rates, float)
    if observation == "poisson":
        ll = np.einsum("nc,cx->nx", n, np.log(mu), optimize=False)
        ll -= mu.sum(axis=0)
        ll -= gammaln(n + 1).sum(axis=1)[:, None]
    elif observation == "conditional":
        probabilities = mu / mu.sum(axis=0)
        ll = np.einsum("nc,cx->nx", n, np.log(probabilities), optimize=False)
        ll += (gammaln(n.sum(axis=1) + 1) - gammaln(n + 1).sum(axis=1))[:, None]
    else:
        raise ValueError("unknown observation model")
    inside, outside = logsumexp(ll[:, home], axis=1), logsumexp(ll[:, ~home], axis=1)
    bf = inside - outside + np.log((~home).sum() / home.sum())
    raw = np.exp(inside - logsumexp(ll, axis=1))
    return bf, raw


def independent_fit(b, w):
    b, w = np.asarray(b), np.asarray(w)
    b, w = b[w > 0], w[w > 0] / w.sum()
    if np.max(abs(b)) <= 1e-12:
        return np.nan, "nonidentified"

    def negative_loglike(p):
        with np.errstate(divide="ignore"):
            value = np.logaddexp(np.log(p) + b, np.log1p(-p)) - np.maximum(b, 0)
        return -float(w @ value)

    interior = minimize_scalar(negative_loglike, bounds=(0, 1), method="bounded", options={"xatol": 1e-13})
    choices = [(negative_loglike(0), 0.0, "boundary_zero"), (negative_loglike(1), 1.0, "boundary_one"), (interior.fun, float(interior.x), "interior")]
    _, p, status = min(choices)
    return p, status


def check_table(root, name, expected, keys):
    saved = pd.read_csv(root / f"{name}.csv")
    if set(saved.columns) != set(expected.columns):
        raise AssertionError(f"{name}: changed schema")
    left = expected.sort_values(keys).reset_index(drop=True)
    right = saved.sort_values(keys).reset_index(drop=True)[left.columns]
    pd.testing.assert_frame_equal(left, right, check_dtype=False, rtol=1e-8, atol=1e-10, obj=name)


def audit_tables(root, d):
    """Rebuild tables without importing producer aggregation or gate code."""
    truth = d[~d.source.isin(SOURCES[:2])].copy()
    truth["raw_error"] = (truth.raw_posterior_mass - truth.true_prevalence).abs()
    truth["mixture_error"] = (truth.estimate - truth.true_prevalence).abs()
    k = ["observation", "encoding", "source", "panel_type", "animal", "session"]
    m = ["raw_error", "mixture_error"]
    st = truth.groupby(k, dropna=False)[m].mean().reset_index()
    at = st.groupby(k[:-1], dropna=False)[m].mean().reset_index()
    summary = at.groupby(k[:4], dropna=False)[m].mean().reset_index()
    summary["relative_error_reduction"] = (summary.raw_error - summary.mixture_error) / summary.raw_error
    for name, frame, keys in (("session_truth", st, k), ("animal_truth", at, k[:-1]), ("truth_summary", summary, k[:4])):
        check_table(root, name, frame, keys)
    pair_rows = []
    rk = ["observation", "encoding", "source", "animal", "session"]
    for key, frame in d[d.source.isin(SOURCES[:2])].groupby(rk):
        high, low = frame[frame.side == "high"].iloc[0], frame[frame.side == "low"].iloc[0]
        x, y = high.raw_posterior_mass - low.raw_posterior_mass, high.estimate - low.estimate
        pair_rows.append(dict(zip(rk, key, strict=True)) | dict(raw_signed_gap=x, mixture_signed_gap=y, raw_gap=abs(x), mixture_gap=abs(y)))
    pair = pd.DataFrame(pair_rows)
    metrics = ["raw_signed_gap", "mixture_signed_gap", "raw_gap", "mixture_gap"]
    animal = pair.groupby(rk[:-1])[metrics].mean().reset_index()
    gaps = animal.groupby(rk[:3])[metrics].mean().reset_index()
    gaps["relative_gap_reduction"] = (gaps.raw_gap - gaps.mixture_gap) / gaps.raw_gap
    for name, frame, keys in (("pair_contrasts", pair, rk), ("animal_contrasts", animal, rk[:-1]), ("content_summary", gaps, rk[:3])):
        check_table(root, name, frame, keys)
    expected_keys = set()
    for session in SESSIONS:
        for source in SOURCES:
            for obs in ("poisson", "conditional"):
                for enc in ("early_run", "full_run") if source in SOURCES[:2] else ("early_run",):
                    panels = ("real",) if source in SOURCES[:2] else ("p0.05", "p0.15", "p0.30", "p0.50", "p0.75")
                    if source == "run_q4":
                        panels += ("natural",)
                    expected_keys.update((session, obs, enc, source, side, panel) for side in ("high", "low") for panel in panels)
    actual = set(d[["session", "observation", "encoding", "source", "side", "panel"]].itertuples(index=False, name=None))
    flags = dict(all_expected_estimates_present=actual == expected_keys and len(d) == len(expected_keys), all_estimates_identified=bool(np.isfinite(d.estimate).all()))
    for source in SOURCES[2:]:
        a = at[(at.observation == "poisson") & (at.source == source) & (at.panel_type == "fixed_prevalence")]
        s = summary[(summary.observation == "poisson") & (summary.source == source) & (summary.panel_type == "fixed_prevalence")].iloc[0]
        ok = len(a) == 3 and np.isfinite(a.mixture_error).all() and (a.mixture_error <= a.raw_error + 1e-10).all()
        if source in ("run_q4", "test_poisson_gain1"):
            ok = ok and s.mixture_error <= 0.05 and s.relative_error_reduction >= 0.20
        flags[f"truth_{source}"] = bool(ok)
    a = at[(at.observation == "poisson") & (at.panel_type == "natural")]
    flags["natural_run_no_harm"] = bool(len(a) == 3 and np.isfinite(a.mixture_error).all() and (a.mixture_error <= a.raw_error + 1e-10).all())
    for enc in ("early_run", "full_run"):
        for source in SOURCES[:2]:
            a = animal[(animal.observation == "poisson") & (animal.encoding == enc) & (animal.source == source)]
            g = gaps[(gaps.observation == "poisson") & (gaps.encoding == enc) & (gaps.source == source)].iloc[0]
            threshold = 0.2 if source == SOURCES[0] else -1e-10
            flags[f"content_{enc}_{source}"] = bool(
                len(a) == 3 and np.isfinite(a.mixture_gap).all() and (a.mixture_gap <= a.raw_gap + 1e-10).all() and g.relative_gap_reduction >= threshold
            )
    r = d[(d.observation == "poisson") & d.source.isin(SOURCES[:2])]
    flags["real_estimates_interior"] = bool(len(r) == 32 and r.fit_status.eq("interior").all())
    flags["development_numerical_screen"] = all(flags.values())
    saved = pd.read_csv(root / "gate_summary.csv")
    if saved.gate.duplicated().any() or dict(zip(saved.gate, saved.passed, strict=True)) != flags:
        raise AssertionError("gate values do not match independent reconstruction")
    return flags


def audit(root):
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for path, sha in manifest["input_sha256"].items():
        if digest(path) != sha:
            raise AssertionError(f"source changed: {path}")
    for name, sha in manifest["output_sha256"].items():
        if digest(root / name) != sha:
            raise AssertionError(f"output changed: {name}")
    d = pd.read_csv(root / "estimates.csv")
    n_checked = 0
    for session in SESSIONS:
        folder = Path(manifest["source_dir"]) / session.replace("/", "_")
        enc = read_npz(folder / "encoding.npz")
        for source in SOURCES:
            bank = read_npz(folder / f"{source}.npz")
            for encoding in ("early_run", "full_run") if source in SOURCES[:2] else ("early_run",):
                for obs in ("poisson", "conditional"):
                    cache = read_npz(root / f"{session.replace('/', '_')}__{source}__{encoding}__{obs}.npz")
                    for j, side in enumerate(("high", "low")):
                        ix = enc[f"{side}_indices"]
                        b, s = independent_likelihood(bank["counts"][:, ix], enc[encoding][ix], enc["near"], obs)
                        np.testing.assert_allclose(b, cache["log_bf"][:, j], atol=1e-9, rtol=1e-9)
                        np.testing.assert_allclose(s, cache["raw_posterior"][:, j], atol=1e-9, rtol=1e-9)
                        n_checked += len(b)
                        group = d[(d.session == session) & (d.source == source) & (d.encoding == encoding) & (d.observation == obs) & (d.side == side)]
                        for row in group.itertuples():
                            w = np.ones(len(b)) / len(b)
                            if row.panel.startswith("p"):
                                p = float(row.panel[1:])
                                y = bank["labels"].astype(bool)
                                w = np.where(y, p / y.sum(), (1 - p) / (~y).sum())
                                np.testing.assert_allclose(row.true_prevalence, p)
                            elif row.panel == "natural":
                                np.testing.assert_allclose(row.true_prevalence, bank["labels"].mean())
                            elif row.panel != "real":
                                raise AssertionError("unknown panel")
                            np.testing.assert_allclose(row.raw_posterior_mass, w @ s, atol=1e-10)
                            value, status = independent_fit(b, w)
                            np.testing.assert_allclose(row.estimate, value, atol=2e-5, rtol=1e-6, equal_nan=True)
                            if status != row.fit_status:
                                raise AssertionError("optimizer boundary/identification mismatch")
        print(f"independently reconstructed {session}", flush=True)
    flags = audit_tables(root, d)
    return dict(
        status="pass",
        manifest_sha256=digest(manifest_path),
        reconstructed_likelihoods=n_checked,
        reconstructed_estimates=len(d),
        gates=flags,
        scope="all cached count-to-likelihood readouts, independent optimizer, tables and gates; not biological truth",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = audit(args.result_dir)
    result["audit_script_sha256"] = digest(__file__)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
