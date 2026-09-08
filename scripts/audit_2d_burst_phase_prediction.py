#!/usr/bin/env python3
"""Cross-event burst-time recruitment comparator on frozen PF/Tanni events."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from hipporeplayimm.burst_phase_prediction import event_phase, fit_burst_phase, phase_weights, predict_burst_phase, predictive_score, score_orders
from scripts._provenance import build_script_provenance, file_sha256
from scripts.audit_2d_count_conditioned_prediction import IDENTITY, KEYS, folds
from scripts.audit_2d_predictive_order_map import shuffled_indices

HASHES = {
    "parent": ("conditional_2d_manifest.json", "d16542d0bf935929a05a335f84cd2060167c76dd8e255378beb202fb84e97386"),
    "rate": ("mua_rate_transfer_manifest.json", "84418e7db33df939b9609dae312eed19d72b7dc6e8d382d7cbdc91e4c1744943"),
    "learned": ("learned_assembly_manifest.json", "a8507a6db14d07e7e6f8d5d709a571da949c9394a6c31e2ce8f8e54c88f7efa6"),
}
KNOTS = (3, 5, 10)
DIFFERENCES = {
    "phase_minus_global": ("score_phase", "score_global"),
    "spatial_imm_minus_phase": ("score_first_order_imm", "score_phase"),
    "learned_hmm_minus_phase": ("score_learned_hmm", "score_phase"),
    "spatial_iid_minus_phase": ("score_iid_position", "score_phase"),
    "learned_iid_minus_phase": ("score_same_emissions_iid", "score_phase"),
    "phase_order_advantage": ("score_phase", "mean_shuffle_phase"),
    "learned_order_advantage": ("score_learned_hmm", "mean_shuffle_hmm"),
    "phase_minus_phase_averaged": ("score_phase", "score_phase_averaged"),
}


def worker(job):
    item, parent, rate, learned, out = job
    parent, rate, learned, out = map(Path, (parent, rate, learned, out))
    tag = item["tag"]
    fields = {k: item[k] for k in IDENTITY}
    identity = tuple(fields.values())
    selected = pd.read_csv(parent / f"{tag}_selection.csv")
    with np.load(parent / f"{tag}_cache.npz") as z:
        cache = dict(z)
    source = pd.read_csv(rate / f"{tag}_original.csv.gz")
    source = source[source.alpha.eq(100) & source["map"].eq("real")].set_index(["event_id", "split"])
    if source.index.duplicated().any():
        raise ValueError("duplicate spatial source")
    original, shuffles, details = [], [], []
    tick = time.monotonic()
    for fold, test, cal, excluded in folds(selected):
        stem = f"{tag}__fold{fold}__k50"
        hm = json.loads((learned / f"{stem}_manifest.json").read_text())
        if hm["test_ids"] != test.event_id.tolist() or hm["calibration_ids"] != cal.event_id.tolist() or hm["excluded_ids"] != excluded.event_id.tolist():
            raise ValueError("different calibration/test split")
        if not hm["fit_converged"]:
            raise ValueError("unconverged frozen HMM")
        old = pd.read_csv(learned / f"{stem}_scores.csv.gz")
        h = old[old.shuffle.eq(-1)].set_index(["event_id", "split"])
        hs = old[old.shuffle.ge(0)].groupby(["event_id", "split"]).score_learned_hmm.mean()
        sequences = [cache[f"counts_{eid}"] for eid in cal.event_id]
        phases = [event_phase(cache[f"times_{eid}"], cache[f"edges_{eid}"]) for eid in cal.event_id]
        fits = {n: fit_burst_phase(sequences, phases, n) for n in KNOTS}
        with np.load(learned / f"{stem}_fit.npz") as z:
            np.testing.assert_allclose(fits[5].global_probability, z["global_probability"], rtol=0, atol=1e-12)
        payload = {}
        for n, fit in fits.items():
            payload[f"probabilities_{n}"] = fit.probabilities
            payload[f"weighted_counts_{n}"] = fit.weighted_counts
        payload["global_probability"] = fits[5].global_probability
        np.savez_compressed(out / f"{tag}__fold{fold}_phase_fit.npz", **payload)
        details.append(
            fields
            | {
                "fold": fold,
                "test_ids": test.event_id.tolist(),
                "calibration_ids": cal.event_id.tolist(),
                "excluded_ids": excluded.event_id.tolist(),
                "calibration_bins": fits[5].n_calibration_bins,
                "calibration_spikes": int(sum(x.sum() for x in sequences)),
            }
        )
        for e in test.itertuples():
            x = cache[f"counts_{e.event_id}"]
            u = event_phase(cache[f"times_{e.event_id}"], cache[f"edges_{e.event_id}"])
            orders = np.array([shuffled_indices(identity, int(e.event_id), len(x), j) for j in range(20)])
            p = phase_weights(u, 5) @ fits[5].probabilities
            for split in range(5):
                held, train = cache[f"held_{split}"], cache[f"train_{split}"]
                np.testing.assert_array_equal(np.sort(np.r_[train, held]), np.arange(x.shape[1]))
                ns = int(x[:, held].sum())
                s, hrow = source.loc[(e.event_id, split)], h.loc[(e.event_id, split)]
                if ns != s.n_heldout_spikes or ns != hrow.n_heldout_spikes:
                    raise ValueError("held-out support differs")
                global_score = predictive_score(x, np.tile(fits[5].global_probability, (len(x), 1)), held)
                np.testing.assert_allclose(global_score, hrow.score_nonspatial_global, atol=1e-9)
                sv = score_orders(x, p, held, orders)
                common = fields | {
                    "event_id": int(e.event_id),
                    "fold": fold,
                    "split": split,
                    "n_heldout_spikes": ns,
                    "n_train_spikes": int(x[:, train].sum()),
                    "n_time_bins": len(x),
                    "duration_s": float(e.end_s - e.start_s),
                }
                shuffles.extend(common | {"shuffle": j, "score_phase": float(v)} for j, v in enumerate(sv))
                for n, fit in fits.items():
                    event_profile = phase_weights(u, n) @ fit.probabilities
                    averaged_profile = np.tile(event_profile.mean(axis=0), (len(x), 1))
                    original.append(
                        common
                        | {
                            "n_knots": n,
                            "score_phase": predict_burst_phase(x, u, held, fit),
                            "score_global": global_score,
                            "score_phase_averaged": predictive_score(x, averaged_profile, held),
                            "score_learned_hmm": hrow.score_learned_hmm,
                            "score_same_emissions_iid": hrow.score_same_emissions_iid,
                            "score_first_order_imm": s.score_first_order_imm,
                            "score_iid_position": s.score_iid_position,
                            "mean_shuffle_phase": float(sv.mean()) if n == 5 else np.nan,
                            "mean_shuffle_hmm": float(hs.loc[(e.event_id, split)]),
                            "target_observations_used_for_phase_fit": False,
                        }
                    )
    pd.DataFrame(original).to_csv(out / f"{tag}_original.csv.gz", index=False)
    pd.DataFrame(shuffles).to_csv(out / f"{tag}_shuffles.csv.gz", index=False)
    (out / f"{tag}_folds.json").write_text(json.dumps(details, indent=2) + "\n")
    result = fields | {"tag": tag, "events": len(selected), "rows": len(original), "shuffle_rows": len(shuffles), "runtime_s": time.monotonic() - tick}
    print(json.dumps(result), flush=True)
    return result


def summarize(original):
    if original.empty or original.duplicated(KEYS + ["n_knots"]).any():
        raise ValueError("empty or duplicate paired rows")
    if not original.groupby(KEYS).n_knots.apply(lambda x: set(x) == set(KNOTS)).all():
        raise ValueError("missing knot sensitivity")
    if not original.groupby(IDENTITY + ["event_id", "n_knots"]).split.apply(lambda x: set(x) == set(range(5))).all():
        raise ValueError("missing neural split")
    rows = []
    for name, (a, b) in DIFFERENCES.items():
        frame = original if name != "phase_order_advantage" else original[original.n_knots.eq(5)]
        if not np.isfinite(frame[[a, b]]).all().all():
            raise ValueError("missing comparator scores")
        part = frame[KEYS + ["fold", "n_knots", "n_heldout_spikes"]].copy()
        part["contrast"] = name
        part["delta"] = frame[a] - frame[b]
        part["delta_per_spike"] = part.delta / part.n_heldout_spikes.replace(0, np.nan)
        rows.append(part)
    splits = pd.concat(rows, ignore_index=True)
    by = IDENTITY + ["event_id", "n_knots", "contrast"]
    events = splits.groupby(by, as_index=False)[["delta", "delta_per_spike"]].median()
    support = splits.groupby(by).delta_per_spike.count().rename("valid_neural_splits").reset_index()
    events = events.merge(support, validate="one_to_one")
    sessions = events.groupby(IDENTITY + ["n_knots", "contrast"], as_index=False)[["delta", "delta_per_spike"]].mean()
    animals = sessions.groupby(["dataset", "animal", "n_knots", "contrast"], as_index=False)[["delta", "delta_per_spike"]].mean()
    summary = []
    for key, g in animals.groupby(["dataset", "n_knots", "contrast"]):
        for metric in ("delta", "delta_per_spike"):
            v = g.sort_values("animal")[metric].to_numpy()
            if not np.isfinite(v).all():
                raise ValueError("animal with no predictive support")
            draws = np.array(list(itertools.product(range(len(v)), repeat=len(v))))
            limits = np.quantile(v[draws].mean(axis=1), [0.025, 0.975])
            summary.append(
                {
                    "dataset": key[0],
                    "n_knots": key[1],
                    "contrast": key[2],
                    "metric": metric,
                    "mean": float(v.mean()),
                    "ci_low": float(limits[0]),
                    "ci_high": float(limits[1]),
                    "positive_animals": int((v > 0).sum()),
                    "animals": len(v),
                }
            )
    return splits, events, sessions, animals, pd.DataFrame(summary)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("parent", "rate", "learned"):
        p.add_argument(f"--{name}-dir", type=Path, required=True)
    for name in ("rate", "learned"):
        p.add_argument(f"--{name}-audit", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    inputs = {
        "script": Path(__file__),
        "protocol": ROOT / "docs/burst_phase_prediction_protocol.md",
        "kernel": ROOT / "src/hipporeplayimm/burst_phase_prediction.py",
        "fold_logic": ROOT / "scripts/audit_2d_count_conditioned_prediction.py",
        "shuffle_logic": ROOT / "scripts/audit_2d_predictive_order_map.py",
    }
    manifests = {}
    for name, (filename, digest) in HASHES.items():
        path = getattr(args, name + "_dir") / filename
        if file_sha256(path) != digest:
            raise ValueError("wrong frozen " + name)
        manifests[name] = json.loads(path.read_text())
        if manifests[name]["status"] != "complete":
            raise ValueError("incomplete parent")
        inputs[name] = path
    for name, sourcekey in (("rate", "run_manifest"), ("learned", "scoring_manifest")):
        audit = json.loads(getattr(args, name + "_audit").read_text())
        if audit["status"] != "pass" or audit["input_file_sha256"][sourcekey] != HASHES[name][1]:
            raise ValueError("matching passing audit required")
        inputs[name + "_audit"] = getattr(args, name + "_audit")
    items = sorted(manifests["parent"]["completed"], key=lambda x: x["tag"])
    for item in items:
        tag = item["tag"]
        needed = {
            "parent": [tag + "_cache.npz", tag + "_selection.csv"],
            "rate": [tag + "_original.csv.gz"],
            "learned": [f"{tag}__fold{f}__k50_{suffix}" for f in range(5) for suffix in ("fit.npz", "scores.csv.gz", "manifest.json")],
        }
        for source, names in needed.items():
            for name in names:
                path = getattr(args, source + "_dir") / name
                if file_sha256(path) != manifests[source]["output_sha256"][name]:
                    raise ValueError("changed source " + name)
                inputs[name] = path
    m = build_script_provenance(input_paths=inputs, cwd=ROOT)
    if m["git_dirty"]:
        raise ValueError("commit frozen code before scoring")
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    mp = out / "burst_phase_manifest.json"
    m.update(status="running", n_knots=KNOTS, primary_knots=5, shuffles=20, source_models_rescored=False, independent_confirmation=False)
    mp.write_text(json.dumps(m, indent=2) + "\n")
    start = time.monotonic()
    try:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            m["completed"] = list(pool.map(worker, [(i, args.parent_dir, args.rate_dir, args.learned_dir, out) for i in items]))
        original = pd.concat([pd.read_csv(out / f"{i['tag']}_original.csv.gz") for i in items], ignore_index=True)
        tables = summarize(original)
        for name, table in zip(("splits", "events", "sessions", "animals", "summary"), tables, strict=True):
            table.to_csv(out / f"burst_phase_{name}.csv.gz", index=False)
        expected = sum(i["events"] for i in items)
        gates = {"all_events_present": len(original) == expected * 5 * 3, "independent_audit_required": True}
        main = tables[-1].query('n_knots == 5 and metric == "delta_per_spike"')
        for contrast in list(DIFFERENCES)[:3]:
            g = main[main.contrast.eq(contrast)]
            gates[contrast + "_both_datasets_positive"] = bool(len(g) == 2 and g.ci_low.gt(0).all() and g.positive_animals.eq(g.animals).all())
        for contrast in ("phase_minus_phase_averaged", "phase_order_advantage"):
            g = main[main.contrast.eq(contrast)]
            gates[contrast + "_both_datasets_positive"] = bool(len(g) == 2 and g.ci_low.gt(0).all() and g.positive_animals.eq(g.animals).all())
        gates["recruitment_lead"] = all(gates[name + "_both_datasets_positive"] for name in ("phase_minus_global", "phase_minus_phase_averaged", "phase_order_advantage"))
        gates["high_importance_discovery_established"] = False
        pd.DataFrame([gates]).to_csv(out / "burst_phase_gates.csv", index=False)
        m.update(status="complete", events=expected, original_rows=len(original), gates=gates, runtime_s=time.monotonic() - start)
    except BaseException as exc:
        m.update(status="failed", error=repr(exc))
        raise
    finally:
        m["output_sha256"] = {x.name: file_sha256(x) for x in out.iterdir() if x.is_file() and x != mp}
        mp.write_text(json.dumps(m, indent=2) + "\n")
    print(json.dumps(m["gates"]), flush=True)


if __name__ == "__main__":
    main()
