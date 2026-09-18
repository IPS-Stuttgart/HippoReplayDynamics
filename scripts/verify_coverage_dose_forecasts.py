#!/usr/bin/env python3
"""Independent subset, geometry, forecast and aggregation reconstruction."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import binomtest

from scripts._provenance import build_script_provenance, file_sha256
from scripts.verify_independent_rejected_forecasts import ref_filter, ref_geometry, ref_ll, ref_null

ID = ["dataset", "animal", "session"]
BASELINES = ["matched_own", "matched_shared", "frozen", "no_history", "global"]


def reference_score(counts, train, held, fit, null):
    a, initial = fit["transition"], fit["initial"]
    ll = ref_ll(counts[:, train], fit["probabilities"][train])
    filtered = ref_filter(ll, initial, a)
    matched = ref_filter(ll, initial, null)
    q = {
        "dynamic": filtered[:-2] @ a @ a,
        "matched_own": matched[:-2] @ null @ null,
        "matched_shared": filtered[:-2] @ null @ null,
        "frozen": filtered[:-2],
        "no_history": np.array([initial @ np.linalg.matrix_power(a, i) for i in range(2, len(counts))]),
    }
    held_counts = counts[2:, held]
    hl = ref_ll(held_counts, fit["probabilities"][held])
    totals = held_counts.sum(axis=1)
    result = {}
    for key, probabilities in q.items():
        values = logsumexp(np.log(np.maximum(probabilities, 1e-300)) + hl, axis=1)
        values[totals == 0] = 0
        result["score_" + key] = float(values.sum())
    result["score_global"] = float(ref_ll(held_counts, fit["global_probability"][held, None]).sum())
    return result


def independent_events(rows):
    keys = ID + ["event_id", "split"]
    cols = ["dynamic_minus_" + b for b in BASELINES]
    x = rows.copy()
    for b, col in zip(BASELINES, cols, strict=True):
        x[col] = (x.score_dynamic - x["score_" + b]) / x.n_heldout_target_spikes.replace(0, np.nan)
    baseline = x[x.fraction.eq(1)].set_index(keys)
    index = pd.MultiIndex.from_frame(x[keys])
    for col in cols + ["geometric_pass", "valid_frames"]:
        x["full_" + col] = baseline[col].reindex(index).to_numpy()
    for col in cols:
        x["paired_change_" + col] = x[col] - x["full_" + col]
    x["pass_change"] = x.geometric_pass.astype(int) - x.full_geometric_pass.astype(int)
    x["acceptance"] = x.geometric_pass.astype(float)
    x["mean_inference_cells"], x["mean_inference_spikes"] = x.n_train_cells, x.n_train_spikes
    full, current = x.full_geometric_pass.astype(bool), x.geometric_pass.astype(bool)
    masks = {
        "all": x.fraction.notna(),
        "full_pass": full,
        "full_supported": x.full_valid_frames.ge(10),
        "lost": full & ~current,
        "lost_supported": full & ~current & x.valid_frames.ge(10),
        "gained": ~full & current,
        "retained": full & current,
    }
    medians = cols + ["full_" + c for c in cols] + ["paired_change_" + c for c in cols]
    means = ["acceptance", "pass_change", "mean_inference_cells", "mean_inference_spikes"]
    pieces = []
    for group, mask in masks.items():
        local = x[mask]
        if local.empty:
            continue
        split_keys = keys + ["arm", "fraction"]
        med = local.groupby(split_keys)[medians].median()
        avg = local.groupby(split_keys)[means].mean()
        split = med.join(avg).reset_index()
        event_keys = ID + ["event_id", "arm", "fraction"]
        event = split.groupby(event_keys)[medians].median().join(split.groupby(event_keys)[means].mean()).reset_index()
        pieces.append(event.assign(group=group))
    return pd.concat(pieces, ignore_index=True)


def independent_summary(events):
    factors = ["arm", "fraction", "group"]
    metrics = [c for c in events if c.startswith(("dynamic_minus_", "full_dynamic_minus_", "paired_change_"))]
    metrics += ["acceptance", "pass_change", "mean_inference_cells", "mean_inference_spikes"]
    sessions = events.groupby(ID + factors)[metrics].mean().reset_index()
    animals = sessions.groupby(["dataset", "animal"] + factors)[metrics].mean().reset_index()
    out = []
    for keys, group in animals.groupby(["dataset"] + factors):
        for metric in metrics:
            v = group.sort_values("animal")[metric].dropna().to_numpy()
            if len(v) == 0:
                continue
            lo, hi = np.quantile(np.array(list(itertools.product(v, repeat=len(v)))).mean(axis=1), [0.025, 0.975])
            nonzero = v[v != 0]
            out.append(
                dict(zip(["dataset"] + factors, keys, strict=True))
                | {
                    "metric": metric,
                    "mean": v.mean(),
                    "ci_low": lo,
                    "ci_high": hi,
                    "animals": len(v),
                    "positive_animals": int((v > 0).sum()),
                    "sign_p_value": binomtest(int((nonzero > 0).sum()), len(nonzero)).pvalue if len(nonzero) else 1.0,
                }
            )
    return animals, pd.DataFrame(out)


def compare_frames(actual, expected, keys, columns):
    a, b = actual.set_index(keys).sort_index(), expected.set_index(keys).sort_index()
    if not a.index.equals(b.index):
        raise AssertionError("mismatched aggregate identities")
    np.testing.assert_allclose(a[columns].to_numpy(float), b[columns].to_numpy(float), atol=1e-9, rtol=1e-9, equal_nan=True)


def audit(root, output):
    mp = root / "coverage_dose_manifest.json"
    manifest = json.loads(mp.read_text())
    if manifest["status"] != "complete" or len(manifest["completed"]) != 33:
        raise AssertionError("complete full-cohort experiment required")
    for key, path in manifest["input_file_paths"].items():
        assert file_sha256(path) == manifest["input_file_sha256"][key]
    for name, sha in manifest["output_sha256"].items():
        assert file_sha256(root / name) == sha
    source_root = Path(manifest["input_file_paths"]["source_manifest"]).parent
    source_manifest = json.loads((source_root / "independent_rejected_forecast_manifest.json").read_text())
    previous = {x["tag"]: x for x in source_manifest["completed"]}
    records, aggregate_frames = [], []
    for rec in manifest["completed"]:
        local, source = root / rec["tag"], source_root / rec["tag"]
        for name, sha in rec["output_sha256"].items():
            assert file_sha256(local / name) == sha
        for name, sha in previous[rec["tag"]]["output_sha256"].items():
            assert file_sha256(source / name) == sha
        with np.load(source / "cache.npz", allow_pickle=False) as z:
            cache = {k: z[k] for k in z.files}
        configurations = json.loads((local / "subsets.json").read_text())
        lookup = {(c["split"], c["repeat"], c["fraction"]): c for c in configurations}
        assert len(lookup) == 155
        for split in range(5):
            train, half, held = (cache[f"{k}_{split}"] for k in ["train", "half", "held"])
            for repeat in range(10):
                key = "|".join(map(str, (20260918, *[rec[k] for k in ID], split, repeat)))
                rng = np.random.default_rng(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little"))
                order = np.concatenate([rng.permutation(half), rng.permutation(np.setdiff1d(train, half))]) if repeat == 0 else rng.permutation(train)
                for fraction in [0.75, 0.5, 0.25]:
                    c = lookup[(split, repeat, fraction)]
                    np.testing.assert_array_equal(c["train"], sorted(order[: max(2, int(len(train) * fraction))]))
                    np.testing.assert_array_equal(c["held"], held)
                    assert set(c["train"]).isdisjoint(held)
        frames, n_scores, n_geometry, max_error = [], 0, 0, 0.0
        folds = json.loads((source / "folds.json").read_text())
        for fold in folds:
            scores = pd.read_csv(local / f"scores_{fold['fold']}.csv.gz")
            frames.append(scores)
            eligible = [eid for eid in fold["test_ids"] if len(cache[f"counts_{eid}"]) > 2]
            if not eligible:
                continue
            eid = eligible[0]
            chosen = scores[scores.event_id.eq(eid)]
            fits = {}
            for row in chosen.itertuples(index=False):
                c = lookup[(row.split, row.repeat, row.fraction)]
                path, geometry = ref_geometry(cache[f"base_{eid}"], cache[f"durations_{eid}"], cache["rates"], cache["centers"], np.array(c["train"]))
                for k, value in geometry.items():
                    assert abs(float(getattr(row, k)) - float(value)) < 1e-8
                n_geometry += 1
                name = f"source_{fold['fold']}" if row.fit_file == "source" else row.fit_file
                if name not in fits:
                    fp = source / f"fit_{fold['fold']}.npz" if row.fit_file == "source" else local / name
                    with np.load(fp, allow_pickle=False) as z:
                        fit = {k: z[k] for k in z.files}
                    fits[name] = fit, ref_null(fit["transition"])
                fit, null = fits[name]
                cells = fit.get("population_indices", np.arange(len(cache["unit_ids"])))
                if row.arm == "restricted_calibration":
                    np.testing.assert_array_equal(cells, sorted(c["train"] + c["held"]))
                train, held = np.searchsorted(cells, c["train"]), np.searchsorted(cells, c["held"])
                values = reference_score(cache[f"counts_{eid}"][:, cells], train, held, fit, null)
                for k, value in values.items():
                    err = abs(value - getattr(row, k))
                    assert err < 1e-8, (rec["tag"], eid, k, err)
                    max_error = max(max_error, err)
                    n_scores += 1
        combined = pd.concat(frames, ignore_index=True)
        assert len(combined) == rec["events"] * 170
        assert not combined.duplicated(["event_id", "split", "repeat", "fraction", "arm"]).any()
        event = independent_events(combined)
        saved = pd.read_csv(local / "events.csv.gz")
        keys = ID + ["event_id", "arm", "fraction", "group"]
        compare_frames(saved, event, keys, [c for c in event if c not in keys])
        aggregate_frames.append(event)
        records.append(
            {k: rec[k] for k in ID}
            | {"scores_reconstructed": n_scores, "geometries_reconstructed": n_geometry, "all_event_aggregates_reconstructed": True, "maximum_score_error": max_error}
        )
        print(json.dumps(records[-1]), flush=True)
    animals, summary = independent_summary(pd.concat(aggregate_frames, ignore_index=True))
    keys = ["dataset", "animal", "arm", "fraction", "group"]
    compare_frames(pd.read_csv(root / "coverage_animals.csv"), animals, keys, [c for c in animals if c not in keys])
    keys = ["dataset", "arm", "fraction", "group", "metric"]
    compare_frames(pd.read_csv(root / "coverage_summary.csv"), summary, keys, [c for c in summary if c not in keys])
    output.mkdir(parents=True, exist_ok=False)
    animals.to_csv(output / "independent_animals.csv", index=False)
    summary.to_csv(output / "independent_summary.csv", index=False)
    provenance = build_script_provenance(input_paths={"run_manifest": mp, "verifier": Path(__file__)}, cwd=ROOT)
    provenance.update(
        status="pass",
        sessions=records,
        all_subsets_checked=True,
        all_aggregates_reconstructed=True,
        model_fitting_independently_repeated=False,
        forecast_sampling_rule="first eligible event per fold, all configurations and arms",
    )
    (output / "coverage_audit.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    audit(args.run_dir, args.output_dir)
