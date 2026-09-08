#!/usr/bin/env python3
"""Independent knot fitting, multinomial predictions and aggregation checks."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy.stats import multinomial

from scripts._provenance import build_script_provenance, file_sha256

ID = ["dataset", "animal", "session"]
KEY = ID + ["event_id", "split"]
CONTRASTS = {
    "phase_minus_global": ("score_phase", "score_global"),
    "spatial_imm_minus_phase": ("score_first_order_imm", "score_phase"),
    "learned_hmm_minus_phase": ("score_learned_hmm", "score_phase"),
    "spatial_iid_minus_phase": ("score_iid_position", "score_phase"),
    "learned_iid_minus_phase": ("score_same_emissions_iid", "score_phase"),
    "phase_order_advantage": ("score_phase", "mean_shuffle_phase"),
    "learned_order_advantage": ("score_learned_hmm", "mean_shuffle_hmm"),
}
CONTRASTS["phase_minus_phase_averaged"] = ("score_phase", "score_phase_averaged")


def phase_from_cache(cache, eid):
    times, edges = cache[f"times_{eid}"], cache[f"edges_{eid}"]
    assert len(edges) == len(times) + 1 and (np.diff(edges) > 0).all()
    np.testing.assert_allclose(times, (edges[:-1] + edges[1:]) / 2, atol=1e-8, rtol=0)
    return (times - edges[0]) / (edges[-1] - edges[0])


def weights(phase, n):
    centers = (np.arange(n) + 0.5) / n
    return np.array([np.interp(phase, centers, np.eye(n)[j]) for j in range(n)]).T


def score(x, p, held):
    y = x[:, held]
    q = p[:, held] / p[:, held].sum(axis=1, keepdims=True)
    return float(multinomial.logpmf(y, y.sum(axis=1), q).sum())


def independent_order(identity, eid, n, shuffle):
    parts = (20260908, *identity, eid, "whole-bin-order", shuffle)
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little")).permutation(n)


def assert_frame(expected, path, keys, values):
    actual = pd.read_csv(path)
    assert not actual.duplicated(keys).any()
    a, b = actual.set_index(keys).sort_index(), expected.set_index(keys).sort_index()
    assert a.index.equals(b.index), path.name
    np.testing.assert_allclose(a[values], b[values], rtol=1e-9, atol=1e-8, equal_nan=True)


def audit_summaries(root, original):
    pieces = []
    for name, (a, b) in CONTRASTS.items():
        x = original.copy() if name != "phase_order_advantage" else original[original.n_knots.eq(5)].copy()
        x["delta"] = x[a] - x[b]
        x["delta_per_spike"] = np.divide(x.delta, x.n_heldout_spikes, out=np.full(len(x), np.nan), where=x.n_heldout_spikes.to_numpy() > 0)
        x["contrast"] = name
        pieces.append(x[KEY + ["fold", "n_knots", "contrast", "delta", "delta_per_spike", "n_heldout_spikes"]])
    splits = pd.concat(pieces, ignore_index=True)
    assert_frame(splits, root / "burst_phase_splits.csv.gz", KEY + ["n_knots", "contrast"], ["fold", "delta", "delta_per_spike", "n_heldout_spikes"])
    event_rows = []
    ek = ID + ["event_id", "n_knots", "contrast"]
    for key, g in splits.groupby(ek):
        assert sorted(g.split) == list(range(5))
        values = g.delta_per_spike.dropna()
        event_rows.append(
            dict(zip(ek, key, strict=True))
            | {"delta": float(np.median(g.delta)), "delta_per_spike": float(np.median(values)) if len(values) else np.nan, "valid_neural_splits": len(values)}
        )
    events = pd.DataFrame(event_rows)
    assert_frame(events, root / "burst_phase_events.csv.gz", ek, ["delta", "delta_per_spike", "valid_neural_splits"])
    sk = ID + ["n_knots", "contrast"]
    sessions = events.groupby(sk)[["delta", "delta_per_spike"]].mean().reset_index()
    assert_frame(sessions, root / "burst_phase_sessions.csv.gz", sk, ["delta", "delta_per_spike"])
    ak = ["dataset", "animal", "n_knots", "contrast"]
    animals = sessions.groupby(ak)[["delta", "delta_per_spike"]].mean().reset_index()
    assert_frame(animals, root / "burst_phase_animals.csv.gz", ak, ["delta", "delta_per_spike"])
    rows = []
    for key, g in animals.groupby(["dataset", "n_knots", "contrast"]):
        for metric in ("delta", "delta_per_spike"):
            v = g.sort_values("animal")[metric].to_numpy()
            assert np.isfinite(v).all()
            boot = [float(np.mean(v[list(indices)])) for indices in itertools.product(range(len(v)), repeat=len(v))]
            low, high = np.percentile(boot, [2.5, 97.5])
            rows.append(
                {
                    "dataset": key[0],
                    "n_knots": key[1],
                    "contrast": key[2],
                    "metric": metric,
                    "mean": v.mean(),
                    "ci_low": low,
                    "ci_high": high,
                    "positive_animals": int((v > 0).sum()),
                    "animals": len(v),
                }
            )
    summary = pd.DataFrame(rows)
    assert_frame(summary, root / "burst_phase_summary.csv.gz", ["dataset", "n_knots", "contrast", "metric"], ["mean", "ci_low", "ci_high", "positive_animals", "animals"])
    return summary


def verify(root, out):
    mp = root / "burst_phase_manifest.json"
    manifest = json.loads(mp.read_text())
    assert manifest["status"] == "complete" and not manifest["git_dirty"]
    for name, path in manifest["input_file_paths"].items():
        assert file_sha256(path) == manifest["input_file_sha256"][name], name
    for name, digest in manifest["output_sha256"].items():
        assert file_sha256(root / name) == digest, name
    parent = Path(manifest["input_file_paths"]["parent"]).parent
    rate = Path(manifest["input_file_paths"]["rate"]).parent
    learned = Path(manifest["input_file_paths"]["learned"]).parent
    frames = []
    checked = 0
    maximum = 0.0
    for item in manifest["completed"]:
        tag = item["tag"]
        cache = dict(np.load(parent / f"{tag}_cache.npz"))
        selection = pd.read_csv(parent / f"{tag}_selection.csv").sort_values(["start_s", "event_id"])
        published = json.loads((root / f"{tag}_folds.json").read_text())
        actual = pd.read_csv(root / f"{tag}_original.csv.gz")
        shuffles = pd.read_csv(root / f"{tag}_shuffles.csv.gz")
        assert not actual.target_observations_used_for_phase_fit.any()
        assert len(actual) == len(selection) * 15 and len(shuffles) == len(selection) * 100
        assert not actual.duplicated(["event_id", "split", "n_knots"]).any()
        assert not shuffles.duplicated(["event_id", "split", "shuffle"]).any()
        spatial = pd.read_csv(rate / f"{tag}_original.csv.gz")
        spatial = spatial[spatial.alpha.eq(100) & spatial["map"].eq("real")].set_index(["event_id", "split"])
        assert set(actual.event_id) == set(selection.event_id) == set(shuffles.event_id)
        for col in ID:
            assert actual[col].eq(item[col]).all() and shuffles[col].eq(item[col]).all()
        for fold, index in enumerate(np.array_split(np.arange(len(selection)), 5)):
            test = selection.iloc[index]
            possible = selection[~selection.event_id.isin(test.event_id)]
            keep = []
            for c in possible.itertuples():
                keep.append(all(c.end_s + 1 <= t.start_s or c.start_s >= t.end_s + 1 for t in test.itertuples()))
            cal = possible[np.array(keep)]
            excluded = possible[~np.array(keep)]
            detail = next(v for v in published if v["fold"] == fold)
            assert detail["test_ids"] == test.event_id.tolist() and detail["calibration_ids"] == cal.event_id.tolist() and detail["excluded_ids"] == excluded.event_id.tolist()
            sequences = [cache[f"counts_{eid}"] for eid in cal.event_id]
            xcal = np.concatenate(sequences)
            phases = np.concatenate([phase_from_cache(cache, eid) for eid in cal.event_id])
            nc = xcal.shape[1]
            gp = (xcal.sum(axis=0) + 100 / nc) / (xcal.sum() + 100)
            saved = dict(np.load(root / f"{tag}__fold{fold}_phase_fit.npz"))
            np.testing.assert_allclose(saved["global_probability"], gp, atol=1e-12)
            assert detail["calibration_bins"] == len(xcal) and detail["calibration_spikes"] == xcal.sum()
            profiles = {}
            for n in (3, 5, 10):
                accumulated = weights(phases, n).T @ xcal
                profiles[n] = accumulated + 100 * gp
                profiles[n] /= profiles[n].sum(axis=1, keepdims=True)
                np.testing.assert_allclose(saved[f"weighted_counts_{n}"], accumulated, atol=1e-7)
                np.testing.assert_allclose(saved[f"probabilities_{n}"], profiles[n], atol=1e-10)
            old = pd.read_csv(learned / f"{tag}__fold{fold}__k50_scores.csv.gz")
            original_hmm = old[old.shuffle.eq(-1)].set_index(["event_id", "split"])
            average_hmm = old[old.shuffle.ge(0)].groupby(["event_id", "split"]).score_learned_hmm.mean()
            original_lookup = actual[actual.fold.eq(fold)].set_index(["event_id", "split", "n_knots"]).sort_index()
            shuffled_lookup = shuffles[shuffles.fold.eq(fold)].set_index(["event_id", "split", "shuffle"]).sort_index()
            for e in test.itertuples():
                x = cache[f"counts_{e.event_id}"]
                assert np.isfinite(x).all() and (x >= 0).all() and (x == np.floor(x)).all()
                assert x.sum() == e.n_spikes_qc_units and np.count_nonzero(x.sum(axis=0)) == e.n_active_qc_units
                u = phase_from_cache(cache, e.event_id)
                for split in range(5):
                    held, train = cache[f"held_{split}"], cache[f"train_{split}"]
                    np.testing.assert_array_equal(np.sort(np.r_[train, held]), np.arange(nc))
                    for n in (3, 5, 10):
                        row = original_lookup.loc[(e.event_id, split, n)]
                        assert row.n_heldout_spikes == x[:, held].sum() and row.n_train_spikes == x[:, train].sum() and row.n_time_bins == len(x)
                        np.testing.assert_allclose(row.duration_s, e.end_s - e.start_s, atol=1e-8)
                        p = weights(u, n) @ profiles[n]
                        computed = score(x, p, held)
                        maximum = max(maximum, abs(computed - row.score_phase))
                        np.testing.assert_allclose(computed, row.score_phase, atol=1e-8, rtol=1e-9)
                        np.testing.assert_allclose(score(x, np.tile(p.mean(axis=0), (len(x), 1)), held), row.score_phase_averaged, atol=1e-8)
                        np.testing.assert_allclose(score(x, np.tile(gp, (len(x), 1)), held), row.score_global, atol=1e-8)
                        for name in ("score_first_order_imm", "score_iid_position"):
                            np.testing.assert_allclose(row[name], spatial.loc[(e.event_id, split), name], atol=1e-10)
                        for name in ("score_learned_hmm", "score_same_emissions_iid"):
                            np.testing.assert_allclose(row[name], original_hmm.loc[(e.event_id, split), name], atol=1e-10)
                        np.testing.assert_allclose(row.mean_shuffle_hmm, average_hmm.loc[(e.event_id, split)], atol=1e-10)
                        checked += 1
                    group = shuffled_lookup.loc[(e.event_id, split)]
                    assert set(group.index) == set(range(20))
                    assert np.isfinite(group.score_phase).all() and group.score_phase.le(1e-8).all()
                    observed_mean = original_lookup.loc[(e.event_id, split, 5), "mean_shuffle_phase"]
                    np.testing.assert_allclose(observed_mean, group.score_phase.mean(), atol=1e-10)
                    # Reproduce the recorded parent permutation seed without calling its helper.
                    order = independent_order(tuple(item[k] for k in ID), int(e.event_id), len(x), 0)
                    computed = score(x[order], weights(u, 5) @ profiles[5], held)
                    maximum = max(maximum, abs(computed - group.loc[0, "score_phase"]))
                    np.testing.assert_allclose(computed, group.loc[0, "score_phase"], atol=1e-8, rtol=1e-9)
                    checked += 1
        frames.append(actual)
        print("verified " + tag, flush=True)
    original = pd.concat(frames, ignore_index=True)
    assert original.groupby("dataset").event_id.size().to_dict() == {"pfeiffer_foster": 4001 * 15, "tanni2022": 5224 * 15}
    summary = audit_summaries(root, original)
    gates = {"all_events_present": len(original) == manifest["events"] * 15, "independent_audit_required": True}
    for name in list(CONTRASTS)[:3]:
        g = summary[summary.n_knots.eq(5) & summary.metric.eq("delta_per_spike") & summary.contrast.eq(name)]
        gates[name + "_both_datasets_positive"] = bool(len(g) == 2 and g.ci_low.gt(0).all() and g.positive_animals.eq(g.animals).all())
    gates["high_importance_discovery_established"] = False
    for name in ("phase_minus_phase_averaged", "phase_order_advantage"):
        g = summary[summary.n_knots.eq(5) & summary.metric.eq("delta_per_spike") & summary.contrast.eq(name)]
        gates[name + "_both_datasets_positive"] = bool(len(g) == 2 and g.ci_low.gt(0).all() and g.positive_animals.eq(g.animals).all())
    gates["recruitment_lead"] = all(gates[name + "_both_datasets_positive"] for name in ("phase_minus_global", "phase_minus_phase_averaged", "phase_order_advantage"))
    assert gates == manifest["gates"] == pd.read_csv(root / "burst_phase_gates.csv").iloc[0].to_dict()
    out.mkdir(parents=True, exist_ok=False)
    m = build_script_provenance(input_paths={"source_manifest": mp, "verifier": Path(__file__)}, cwd=ROOT)
    m.update(
        status="pass", independent_predictions=checked, max_error=maximum, all_summaries_reconstructed=True, native_sources_reopened=False, source_HMM_spatial_models_rescored=False
    )
    (out / "burst_phase_audit.json").write_text(json.dumps(m, indent=2) + "\n")
    print(json.dumps(m), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    verify(a.run_dir, a.output_dir)


if __name__ == "__main__":
    main()
