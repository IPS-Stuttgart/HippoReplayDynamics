#!/usr/bin/env python3
"""Separate probability reconstruction for within-bin spatial consistency."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import gammaln, logsumexp
from scipy.stats import spearmanr

SESSIONS = ("Rat1/Open1", "Rat1/Open2", "Rat2/Open1", "Rat4/Open2")
REAL = ("all_fixed_candidates", "full_accepted_segment")
TRUTH = ("run_q4", "test_poisson_gain1", "test_poisson_gain4", "test_conditional", "test_conditional_map_drift", "test_conditional_shared_assembly")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for b in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def read_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return dict(z)


def gain(n, rates, folds):
    answer = []
    for held in folds:
        train = np.array([j for j in range(rates.shape[0]) if j not in held])
        log_probs = []
        for indices in (train, held):
            counts = n[:, indices]
            probabilities = rates[indices] / rates[indices].sum(axis=0)
            lp = np.einsum("nc,cx->nx", counts, np.log(probabilities), optimize=False)
            lp += (gammaln(counts.sum(axis=1) + 1) - gammaln(counts + 1).sum(axis=1))[:, None]
            log_probs.append(lp)
        posterior = log_probs[0] - logsumexp(log_probs[0], axis=1)[:, None]
        value = logsumexp(posterior + log_probs[1], axis=1) - logsumexp(log_probs[1], axis=1) + np.log(rates.shape[1])
        value[(n[:, train].sum(axis=1) == 0) | (n[:, held].sum(axis=1) == 0) | (abs(value) < 1e-12)] = 0
        answer.append(value)
    return np.array(answer).T


def readouts(n, r, grid, mask, truth):
    mu = 0.02 * r
    ll = np.einsum("nc,cx->nx", n, np.log(mu), optimize=False) - mu.sum(axis=0)
    ll -= gammaln(n + 1).sum(axis=1)[:, None]
    p = np.exp(ll - logsumexp(ll, axis=1)[:, None])
    center = p @ grid
    xy = np.clip(np.floor(3 * (grid - grid.min(axis=0)) / np.maximum(grid.max(axis=0) - grid.min(axis=0), 1e-12)), 0, 2).astype(int)
    tiles = np.array([p[:, (xy[:, 0] + 3 * xy[:, 1]) == j].sum(axis=1) for j in range(9)]).T
    return dict(
        home=p[:, mask].sum(axis=1),
        mean=center,
        tiles=tiles,
        entropy=-(p * np.log(np.maximum(p, 1e-300))).sum(axis=1) / np.log(len(grid)),
        error=np.sqrt(np.sum((center - truth) ** 2, axis=1)),
    )


def independent_tables(d):
    rows = []
    for key, frame in d.groupby(["animal", "session", "source", "encoding"]):
        for method in ("all", "predictive_half", "spike_half", "entropy_half"):
            s = frame[frame[method]]
            r = dict(zip(("animal", "session", "source", "encoding"), key, strict=True))
            r.update(
                method=method,
                total_events=len(frame),
                retained_events=len(s),
                zero_score_fraction=np.mean(frame.pair_score == 0),
                home_gap=abs(np.mean(s.high_home) - np.mean(s.low_home)),
                high_home=s.high_home.mean(),
                low_home=s.low_home.mean(),
            )
            for metric in ("separation", "regional_tv", "high_entropy", "low_entropy"):
                r[metric] = s[metric].mean()
            for c in (0, 1):
                total = sum(frame.true_home == c)
                r[f"class{c}_retention"] = sum(s.true_home == c) / total if total else np.nan
            for metric in ("high_error", "low_error", "high_brier", "low_brier"):
                r[metric] = s[metric].mean()
                r[f"balanced_{metric}"] = 0.5 * (s[s.true_home == 0][metric].mean() + s[s.true_home == 1][metric].mean())
            rows.append(r)
    session = pd.DataFrame(rows)
    names = [c for c in session.select_dtypes("number").columns if c not in ("total_events", "retained_events")]
    animal = session.groupby(["animal", "source", "encoding", "method"])[names].mean().reset_index()
    summary = animal.groupby(["source", "encoding", "method"])[names].mean().reset_index()
    cs = []
    for rat, f in d[(d.source == REAL[0]) & (d.encoding == "early_run")].groupby("animal"):
        values = [spearmanr(s.pair_score, s.separation).statistic for _, s in f.groupby("session")]
        cs.append(dict(animal=rat, score_separation_rho=np.mean(values)))
    corr = pd.DataFrame(cs)
    return {"session_summary": session, "animal_summary": animal, "summary": summary, "correlations": corr}


def check_gates(root, d, tables):
    s, a, overall, corr = [tables[k] for k in ("session_summary", "animal_summary", "summary", "correlations")]
    expected = {(session, src, enc) for session in SESSIONS for src in REAL + TRUTH for enc in (("early_run", "full_run") if src in REAL else ("early_run",))}
    actual = set(d[["session", "source", "encoding"]].itertuples(index=False, name=None))
    f = dict(complete_source_coverage=actual == expected and not d.duplicated(["session", "source", "encoding", "observation_index"]).any())
    selected = s[s.method != "all"]
    f["fixed_half_coverage"] = len(selected) == 120 and (selected.retained_events == (selected.total_events + 1) // 2).all()
    for enc in ("early_run", "full_run"):
        for src in REAL:
            g = overall[(overall.encoding == enc) & (overall.source == src)].set_index("method")
            p = a[(a.encoding == enc) & (a.source == src)].pivot(index="animal", columns="method", values="home_gap")
            factor = 0.8 if src == REAL[0] else 1.0
            f[f"home_{enc}_{src}"] = (
                len(p) == 3
                and np.isfinite(p.predictive_half).all()
                and (p.predictive_half <= p["all"] + 1e-10).all()
                and g.loc["predictive_half", "home_gap"] <= factor * g.loc["all", "home_gap"] + 1e-10
            )
    p = a[(a.source == REAL[0]) & (a.encoding == "early_run")]
    for metric in ("separation", "regional_tv", "high_entropy", "low_entropy"):
        v = p.pivot(index="animal", columns="method", values=metric)
        ok = len(v) == 3 and np.isfinite(v.predictive_half).all()
        if metric in ("separation", "regional_tv"):
            ok = ok and (v.predictive_half < v["all"]).mean() >= 0.75 and v.predictive_half.mean() <= 0.9 * v["all"].mean()
        else:
            ok = ok and v.predictive_half.mean() <= v["all"].mean() + 1e-10
        f[f"real_{metric}"] = ok
    for src in TRUTH:
        for metric in ("balanced_high_error", "balanced_low_error", "balanced_high_brier", "balanced_low_brier"):
            v = a[a.source == src].pivot(index="animal", columns="method", values=metric)
            f[f"truth_{src}_{metric}"] = len(v) == 3 and np.isfinite(v.predictive_half).all() and (v.predictive_half <= v["all"] + 1e-10).all()
    c = s[s.source.isin(TRUTH) & (s.method == "predictive_half")][["class0_retention", "class1_retention"]]
    f["both_truth_classes_retained"] = len(c) == 24 and np.isfinite(c).all().all() and (c >= 0.2).all().all()
    f["predictive_correlation"] = (
        len(corr) == 3 and np.isfinite(corr.score_separation_rho).all() and corr.score_separation_rho.mean() < 0 and (corr.score_separation_rho < 0).mean() >= 0.75
    )
    f = {k: bool(v) for k, v in f.items()}
    f["development_numerical_screen"] = all(f.values())
    saved = pd.read_csv(root / "gates.csv")
    if saved.gate.duplicated().any() or dict(zip(saved.gate, saved.passed, strict=True)) != f:
        raise AssertionError("gate reconstruction mismatch")
    return f


def audit(root):
    manifest = json.loads((root / "manifest.json").read_text())
    for path, sha in manifest["input_sha256"].items():
        if digest(path) != sha:
            raise AssertionError(f"changed input: {path}")
    for name, sha in manifest["output_sha256"].items():
        if digest(root / name) != sha:
            raise AssertionError(f"changed result: {name}")
    d = pd.read_csv(root / "events.csv.gz", dtype={"event_id": str})
    n_folds = 0
    for session in SESSIONS:
        folder = Path(manifest["source_dir"]) / session.replace("/", "_")
        enc = read_npz(folder / "encoding.npz")
        ids = json.loads((root / f"{session.replace('/', '_')}_folds.json").read_text())
        for src in REAL + TRUTH:
            bank = read_npz(folder / f"{src}.npz")
            for encoding in ("early_run", "full_run") if src in REAL else ("early_run",):
                frame = d[(d.session == session) & (d.source == src) & (d.encoding == encoding)].sort_values("observation_index")
                np.testing.assert_array_equal(frame.observation_index, np.arange(len(bank["counts"])))
                np.testing.assert_array_equal(frame.event_id.astype(str), bank.get("event_ids", np.arange(len(frame))).astype(str))
                truth = bank.get("labels", np.full(len(frame), np.nan)).astype(float)
                np.testing.assert_allclose(frame.true_home, truth, equal_nan=True)
                outs, scores = [], []
                for side in ("high", "low"):
                    idx = enc[f"{side}_indices"]
                    cell_ids = enc["cell_ids"][idx]
                    raw_seed = int.from_bytes(hashlib.sha256(f"20260915|{session}|{side}".encode()).digest()[:8], "little")
                    folds = np.array_split(np.random.default_rng(raw_seed).permutation(len(idx)), 3)
                    for recorded, fold in zip(ids[side], folds, strict=True):
                        np.testing.assert_array_equal(recorded, cell_ids[fold])
                    counts, rates = bank["counts"][:, idx], enc[encoding][idx]
                    values = gain(counts, rates, folds)
                    np.testing.assert_allclose(frame[[f"{side}_fold{j}_gain" for j in range(3)]], values, atol=1e-9, rtol=1e-8)
                    scores.append(np.median(values, axis=1))
                    np.testing.assert_allclose(frame[f"{side}_score"], scores[-1], atol=1e-9)
                    out = readouts(counts, rates, enc["grid_cm"], enc["near"], bank["truth_cm"])
                    for field in ("home", "entropy", "error"):
                        np.testing.assert_allclose(frame[f"{side}_{field}"], out[field], atol=1e-9, rtol=1e-8, equal_nan=True)
                    np.testing.assert_allclose(frame[f"{side}_brier"], (out["home"] - truth) ** 2, atol=1e-9, equal_nan=True)
                    np.testing.assert_array_equal(frame[f"{side}_spikes"], counts.sum(axis=1))
                    np.testing.assert_array_equal(frame[f"{side}_active"], (counts > 0).sum(axis=1))
                    outs.append(out)
                    n_folds += values.size
                pair_score = np.minimum(scores[0], scores[1])
                np.testing.assert_allclose(frame.pair_score, pair_score, atol=1e-9)
                np.testing.assert_allclose(frame.separation, np.linalg.norm(outs[0]["mean"] - outs[1]["mean"], axis=1), atol=1e-9)
                np.testing.assert_allclose(frame.regional_tv, 0.5 * abs(outs[0]["tiles"] - outs[1]["tiles"]).sum(axis=1), atol=1e-9)
                for method, score in (
                    ("predictive_half", frame.pair_score.to_numpy()),
                    ("spike_half", np.minimum(frame.high_spikes, frame.low_spikes)),
                    ("entropy_half", -np.maximum(frame.high_entropy, frame.low_entropy)),
                ):
                    tie = [int.from_bytes(hashlib.sha256(f"20260915|{session}|{src}|{j}".encode()).digest()[:8], "little") for j in frame.observation_index]
                    order = sorted(range(len(frame)), key=lambda j: (-float(np.asarray(score)[j]), tie[j]))
                    chosen = set(order[: (len(frame) + 1) // 2])
                    mask = np.array([j in chosen for j in range(len(frame))])
                    np.testing.assert_array_equal(frame[method], mask)
                assert frame["all"].all()
        print(f"independently reconstructed {session}", flush=True)
    tables = independent_tables(d)
    keys = {
        "session_summary": ["animal", "session", "source", "encoding", "method"],
        "animal_summary": ["animal", "source", "encoding", "method"],
        "summary": ["source", "encoding", "method"],
        "correlations": ["animal"],
    }
    for name, expected in tables.items():
        actual = pd.read_csv(root / f"{name}.csv")
        pd.testing.assert_frame_equal(
            expected.sort_values(keys[name]).reset_index(drop=True),
            actual[expected.columns].sort_values(keys[name]).reset_index(drop=True),
            check_dtype=False,
            atol=1e-10,
            rtol=1e-8,
            obj=name,
        )
    flags = check_gates(root, d, tables)
    return dict(
        status="pass",
        manifest_sha256=digest(root / "manifest.json"),
        events=len(d),
        fold_predictions_reconstructed=n_folds,
        gates=flags,
        scope="all count-derived fold scores, posteriors, selection, truth metrics, summaries and gates; not biological replay truth",
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--result-dir", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = audit(args.result_dir)
    result["audit_script_sha256"] = digest(__file__)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
