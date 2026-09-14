#!/usr/bin/env python3
"""Frozen within-bin split-cell spatial consistency and fixed-half screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import numpy as np
import pandas as pd
from scipy.special import logsumexp, softmax
from scipy.stats import rankdata

from scripts._provenance import build_script_provenance, file_sha256

SESSIONS = ("Rat1/Open1", "Rat1/Open2", "Rat2/Open1", "Rat4/Open2")
REAL = ("all_fixed_candidates", "full_accepted_segment")
TRUTH = ("run_q4", "test_poisson_gain1", "test_poisson_gain4", "test_conditional", "test_conditional_map_drift", "test_conditional_shared_assembly")
METHODS = ("all", "predictive_half", "spike_half", "entropy_half")


def seed(*items):
    return int.from_bytes(hashlib.sha256("|".join(map(str, (20260915, *items))).encode()).digest()[:8], "little")


def read_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return dict(data)


def make_folds(n, session, side):
    if n < 3:
        raise ValueError("at least three cells needed")
    return tuple(np.array_split(np.random.default_rng(seed(session, side)).permutation(n), 3))


def consistency(counts, rates, folds):
    n, r = np.asarray(counts), np.asarray(rates)
    if n.ndim != 2 or r.ndim != 2 or n.shape[1] != r.shape[0] or np.any(n < 0) or np.any(r <= 0):
        raise ValueError("invalid population arrays")
    if not np.isfinite(n).all() or not np.isfinite(r).all():
        raise ValueError("nonfinite counts/rates")
    if len(folds) != 3 or not np.array_equal(np.sort(np.concatenate(folds)), np.arange(n.shape[1])):
        raise ValueError("folds must partition cells exactly")
    scores = []
    for held in folds:
        train = np.setdiff1d(np.arange(n.shape[1]), held)
        if not len(held) or not len(train):
            raise ValueError("empty cell fold")
        pieces = [n[:, ix] @ np.log(r[ix] / r[ix].sum(axis=0)) for ix in (train, held)]
        value = logsumexp(pieces[0] + pieces[1], axis=1) - logsumexp(pieces[0], axis=1) - logsumexp(pieces[1], axis=1) + np.log(r.shape[1])
        silent = (n[:, train].sum(axis=1) == 0) | (n[:, held].sum(axis=1) == 0)
        value[silent | (np.abs(value) < 1e-12)] = 0
        scores.append(value)
    return np.column_stack(scores)


def decode(counts, rates, grid, near, truth):
    p = softmax(counts @ np.log(rates) - 0.02 * rates.sum(axis=0), axis=1)
    mean = p @ grid
    xy = np.minimum(2, np.floor(3 * (grid - grid.min(axis=0)) / np.maximum(np.ptp(grid, axis=0), 1e-12))).astype(int)
    regions = xy[:, 0] + 3 * xy[:, 1]
    tile = np.column_stack([p[:, regions == j].sum(axis=1) for j in range(9)])
    h = p[:, near].sum(axis=1)
    return dict(mean=mean, tiles=tile, home=h, entropy=-(p * np.log(np.maximum(p, 1e-300))).sum(axis=1) / np.log(p.shape[1]), error=np.linalg.norm(mean - truth, axis=1))


def select_half(score, tie):
    score, tie = np.asarray(score), np.asarray(tie, np.uint64)
    if score.ndim != 1 or not len(score) or score.shape != tie.shape or not np.isfinite(score).all():
        raise ValueError("invalid selection scores")
    selected = np.zeros(len(score), bool)
    selected[np.lexsort((tie, -score))[: (len(score) + 1) // 2]] = True
    return selected


def event_table(session, source, encoding, bank, enc):
    rows = dict(
        session=session,
        animal=session.split("/")[0],
        source=source,
        encoding=encoding,
        observation_index=np.arange(len(bank["counts"])),
        event_id=bank.get("event_ids", np.arange(len(bank["counts"]))).astype(str),
    )
    truth = bank["truth_cm"]
    outputs, scores = [], []
    for side in ("high", "low"):
        ix = enc[f"{side}_indices"]
        counts, rates = bank["counts"][:, ix], enc[encoding][ix]
        folds = make_folds(len(ix), session, side)
        gains = consistency(counts, rates, folds)
        out = decode(counts, rates, enc["grid_cm"], enc["near"], truth)
        np.testing.assert_allclose(out["home"], bank[f"{encoding}_scores"][:, len(outputs)], atol=1e-9, rtol=1e-9)
        for j in range(3):
            rows[f"{side}_fold{j}_gain"] = gains[:, j]
        scores.append(np.median(gains, axis=1))
        for key in ("home", "entropy", "error"):
            rows[f"{side}_{key}"] = out[key]
        rows[f"{side}_score"] = scores[-1]
        rows[f"{side}_spikes"] = counts.sum(axis=1)
        rows[f"{side}_active"] = (counts > 0).sum(axis=1)
        outputs.append(out)
    rows["pair_score"] = np.minimum(*scores)
    rows["separation"] = np.linalg.norm(outputs[0]["mean"] - outputs[1]["mean"], axis=1)
    rows["regional_tv"] = 0.5 * np.abs(outputs[0]["tiles"] - outputs[1]["tiles"]).sum(axis=1)
    labels = bank.get("labels", np.full(len(truth), np.nan)).astype(float)
    rows["true_home"] = labels
    for side in ("high", "low"):
        rows[f"{side}_brier"] = (rows[f"{side}_home"] - labels) ** 2
    tie = np.array([seed(session, source, int(j)) for j in rows["observation_index"]], dtype=np.uint64)
    rows["all"] = np.ones(len(truth), bool)
    rows["predictive_half"] = select_half(rows["pair_score"], tie)
    rows["spike_half"] = select_half(np.minimum(rows["high_spikes"], rows["low_spikes"]), tie)
    rows["entropy_half"] = select_half(-np.maximum(rows["high_entropy"], rows["low_entropy"]), tie)
    return pd.DataFrame(rows)


def correlations(frame):
    values = []
    for _, group in frame.groupby("session"):
        if group.pair_score.nunique() < 2 or group.separation.nunique() < 2:
            values.append(np.nan)
        else:
            values.append(float(np.corrcoef(rankdata(group.pair_score), rankdata(group.separation))[0, 1]))
    return float(np.mean(values))


def summarize(data):
    metrics = ("separation", "regional_tv", "high_entropy", "low_entropy")
    rows = []
    for key, frame in data.groupby(["animal", "session", "source", "encoding"]):
        for method in METHODS:
            selected = frame[frame[method]]
            row = dict(zip(("animal", "session", "source", "encoding"), key, strict=True))
            row.update(
                method=method,
                total_events=len(frame),
                retained_events=len(selected),
                zero_score_fraction=float(frame.pair_score.eq(0).mean()),
                home_gap=abs(selected.high_home.mean() - selected.low_home.mean()),
                high_home=selected.high_home.mean(),
                low_home=selected.low_home.mean(),
            )
            row.update({m: selected[m].mean() for m in metrics})
            for label in (0, 1):
                denominator = int(frame.true_home.eq(label).sum())
                row[f"class{label}_retention"] = float(selected.true_home.eq(label).sum() / denominator) if denominator else np.nan
            for m in ("high_error", "low_error", "high_brier", "low_brier"):
                row[m] = selected[m].mean()
                class_means = [selected.loc[selected.true_home.eq(label), m].mean() for label in (0, 1)]
                row[f"balanced_{m}"] = np.mean(class_means)
            rows.append(row)
    session = pd.DataFrame(rows)
    numeric = [x for x in session.select_dtypes("number").columns if x not in ("total_events", "retained_events")]
    by_animal = session.groupby(["animal", "source", "encoding", "method"])[numeric].mean().reset_index()
    summary = by_animal.groupby(["source", "encoding", "method"])[numeric].mean().reset_index()
    corr_rows = []
    for animal, frame in data[data.source.eq(REAL[0]) & data.encoding.eq("early_run")].groupby("animal"):
        corr_rows.append(dict(animal=animal, score_separation_rho=correlations(frame)))
    corr = pd.DataFrame(corr_rows)
    flags = []

    def gate(name, value, detail=""):
        flags.append(dict(gate=name, passed=bool(value), detail=detail))

    expected = {(s, source, enc) for s in SESSIONS for source in REAL + TRUTH for enc in (("early_run", "full_run") if source in REAL else ("early_run",))}
    actual = set(data[["session", "source", "encoding"]].itertuples(index=False, name=None))
    gate("complete_source_coverage", actual == expected and not data.duplicated(["session", "source", "encoding", "observation_index"]).any())
    sel = session[session.method.ne("all")]
    gate("fixed_half_coverage", len(sel) == 120 and (sel.retained_events == (sel.total_events + 1) // 2).all())
    for enc in ("early_run", "full_run"):
        for source in REAL:
            s = summary[summary.source.eq(source) & summary.encoding.eq(enc)].set_index("method")
            a = by_animal[by_animal.source.eq(source) & by_animal.encoding.eq(enc)].pivot(index="animal", columns="method", values="home_gap")
            reduction = 0.2 if source == REAL[0] else 0.0
            gate(
                f"home_{enc}_{source}",
                len(a) == 3
                and np.isfinite(a.predictive_half).all()
                and (a.predictive_half <= a["all"] + 1e-10).all()
                and s.loc["predictive_half", "home_gap"] <= (1 - reduction) * s.loc["all", "home_gap"] + 1e-10,
            )
    s = summary[summary.source.eq(REAL[0]) & summary.encoding.eq("early_run")].set_index("method")
    a = by_animal[by_animal.source.eq(REAL[0]) & by_animal.encoding.eq("early_run")]
    for m in metrics:
        v = a.pivot(index="animal", columns="method", values=m)
        ok = len(v) == 3 and np.isfinite(v.predictive_half).all()
        if m in ("separation", "regional_tv"):
            ok = ok and (v.predictive_half < v["all"]).mean() >= 0.75 and s.loc["predictive_half", m] <= 0.9 * s.loc["all", m]
        else:
            ok = ok and s.loc["predictive_half", m] <= s.loc["all", m] + 1e-10
        gate(f"real_{m}", ok)
    for source in TRUTH:
        a = by_animal[by_animal.source.eq(source)]
        for m in ("balanced_high_error", "balanced_low_error", "balanced_high_brier", "balanced_low_brier"):
            v = a.pivot(index="animal", columns="method", values=m)
            gate(f"truth_{source}_{m}", len(v) == 3 and np.isfinite(v.predictive_half).all() and (v.predictive_half <= v["all"] + 1e-10).all())
    cls = session[session.source.isin(TRUTH) & session.method.eq("predictive_half")]
    gate(
        "both_truth_classes_retained",
        len(cls) == 24 and np.isfinite(cls[["class0_retention", "class1_retention"]]).all().all() and (cls[["class0_retention", "class1_retention"]] >= 0.2).all().all(),
    )
    gate(
        "predictive_correlation",
        len(corr) == 3 and np.isfinite(corr.score_separation_rho).all() and corr.score_separation_rho.mean() < 0 and (corr.score_separation_rho < 0).mean() >= 0.75,
        corr.to_json(orient="records"),
    )
    gate("development_numerical_screen", all(x["passed"] for x in flags), "Independent reconstruction and external validation additionally required")
    return dict(session_summary=session, animal_summary=by_animal, summary=summary, correlations=corr, gates=pd.DataFrame(flags))


def measure(args):
    inputs = {}

    def checked(path, expected=None):
        sha = file_sha256(path)
        if sha is None or (expected is not None and sha != expected):
            raise ValueError(f"changed or missing source: {path}")
        inputs[str(Path(path).resolve())] = sha
        return Path(path)

    upstream = json.loads(checked(args.source_audit).read_text())
    if upstream.get("status") != "pass":
        raise ValueError("source audit did not pass")
    checked(args.source_dir / "manifest.json", upstream["input_file_sha256"]["source"])
    checked(ROOT / "docs/spatial_predictive_content_protocol.md")
    checked(__file__)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    frames = []
    for session in SESSIONS:
        folder = args.source_dir / session.replace("/", "_")
        hashes = json.loads(checked(folder / "outputs.json").read_text())
        for name, sha in hashes.items():
            checked(folder / name, sha)
        enc = read_npz(folder / "encoding.npz")
        assignments = {
            side: [[int(enc["cell_ids"][enc[f"{side}_indices"]][j]) for j in fold] for fold in make_folds(len(enc[f"{side}_indices"]), session, side)] for side in ("high", "low")
        }
        (args.output_dir / f"{session.replace('/', '_')}_folds.json").write_text(json.dumps(assignments, indent=2) + "\n")
        for source in REAL + TRUTH:
            bank = read_npz(folder / f"{source}.npz")
            for encoding in ("early_run", "full_run") if source in REAL else ("early_run",):
                frames.append(event_table(session, source, encoding, bank, enc))
        print(f"completed {session}", flush=True)
    d = pd.concat(frames, ignore_index=True)
    d.to_csv(args.output_dir / "events.csv.gz", index=False)
    for name, table in summarize(d).items():
        table.to_csv(args.output_dir / f"{name}.csv", index=False)
    for path, sha in inputs.items():
        checked(path, sha)
    manifest = dict(
        **build_script_provenance(),
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        source_dir=str(args.source_dir.resolve()),
        input_sha256=inputs,
        output_sha256={p.name: file_sha256(p) for p in args.output_dir.iterdir()},
        independent_audit="pending",
        external_validation=False,
    )
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-dir", required=True, type=Path)
    p.add_argument("--source-audit", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    measure(p.parse_args())


if __name__ == "__main__":
    main()
