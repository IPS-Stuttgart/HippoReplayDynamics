#!/usr/bin/env python3
"""Fixed-event neuron dose curves with future-neuron predictive validation."""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from hipporeplayimm.coverage_dose_forecast import FRACTIONS, REPEATS, event_tables, subsets, summaries
from hipporeplayimm.independent_rejected_forecast import BASELINES, ID, forecast_distributions, score_predictions, seed
from hipporeplayimm.lagged_neural_prediction import NeuralOperator
from hipporeplayimm.learned_assembly_prediction import fit_learned_assembly
from hipporeplayimm.occupancy_matched_forecast import OccupancyMatchedNull
from hipporeplayimm.training_continuity import classify_training
from scripts._provenance import build_script_provenance, file_sha256


def checked(path, digest):
    if file_sha256(path) != digest:
        raise ValueError("input hash mismatch: " + str(path))


def configurations(cache, identity):
    result = []
    for split in range(5):
        for repeat in range(REPEATS):
            for fraction, train in subsets(cache[f"train_{split}"], cache[f"half_{split}"], identity, split, repeat).items():
                if fraction == 1 and repeat:
                    continue
                held = cache[f"held_{split}"]
                if set(train) & set(held):
                    raise ValueError("inference/evaluation overlap")
                result.append({"split": split, "repeat": repeat, "fraction": fraction, "train": train, "held": held})
    return result


def save_fit(path, fit, cells):
    np.savez_compressed(
        path,
        probabilities=fit.probabilities,
        initial=fit.initial,
        transition=fit.transition,
        occupancy=fit.occupancy,
        global_probability=fit.global_probability,
        objective_trace=fit.objective_trace,
        population_indices=cells,
    )


def process(record, source, root):
    identity = {k: record[k] for k in ID}
    source, output = Path(source) / record["tag"], Path(root) / record["tag"]
    output.mkdir(exist_ok=False)
    started = time.monotonic()
    result = identity | {"tag": record["tag"], "status": "running"}
    try:
        for name, sha in record["output_sha256"].items():
            checked(source / name, sha)
        with np.load(source / "cache.npz", allow_pickle=False) as z:
            cache = {k: z[k] for k in z.files}
        selected = pd.read_csv(source / "selection.csv")
        folds = json.loads((source / "folds.json").read_text())
        config = configurations(cache, tuple(identity.values()))
        selections = [{k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in c.items()} for c in config]
        (output / "subsets.json").write_text(json.dumps(selections) + "\n")
        labels, paths = [], {}
        for e in selected.itertuples(index=False):
            for c in config:
                path, _, classifications = classify_training(
                    cache[f"base_{e.event_id}"], cache[f"durations_{e.event_id}"], cache["rates"], cache["centers"], c["train"], np.ones(len(cache["centers"]), bool)
                )
                key = (int(e.event_id), c["split"], c["repeat"], c["fraction"])
                labels.append(identity | dict(event_id=e.event_id, **{k: c[k] for k in ["split", "repeat", "fraction"]}) | classifications[0])
                paths["_".join(map(str, key))] = path
        label_table = pd.DataFrame(labels)
        label_table.to_csv(output / "labels.csv.gz", index=False)
        np.savez_compressed(output / "paths.npz", **paths)
        lookup = label_table.set_index(["event_id", "split", "repeat", "fraction"])
        result["events"] = len(selected)
        fit_metadata, pieces = [], []
        for fold_meta in folds:
            fold = fold_meta["fold"]
            if not fold_meta["converged"]:
                raise ValueError("unconverged source fit")
            with np.load(source / f"fit_{fold}.npz", allow_pickle=False) as z:
                fit = {k: z[k] for k in z.files}
            original = NeuralOperator(fit["initial"], fit["transition"], fit["occupancy"])
            original_null = OccupancyMatchedNull.from_operator(original)
            rows = []
            for c in config:
                full_population = np.arange(len(cache["unit_ids"]))
                arms = [("fixed_codebook", fit, original, original_null, full_population, "source")]
                if c["repeat"] == 0 and c["fraction"] < 1:
                    cells = np.sort(np.r_[c["train"], c["held"]])
                    sequences = [cache[f"counts_{eid}"][:, cells] for eid in fold_meta["calibration_ids"]]
                    fitted = fit_learned_assembly(sequences, 50, seed(*identity.values(), "coverage_refit", fold, c["split"], c["fraction"]))
                    name = f"refit_{fold}_{c['split']}_{c['fraction']}.npz"
                    save_fit(output / name, fitted, cells)
                    fit_metadata.append(
                        {
                            "file": name,
                            "fold": fold,
                            "split": c["split"],
                            "fraction": c["fraction"],
                            "converged": bool(fitted.converged),
                            "restart": int(fitted.restart),
                            "restart_converged": list(fitted.restart_converged),
                            "restart_objectives": list(fitted.restart_objectives),
                        }
                    )
                    (output / "fit_metadata.json").write_text(json.dumps(fit_metadata, indent=2) + "\n")
                    if not fitted.converged:
                        raise ValueError("restricted-calibration nonconvergence: " + name)
                    reduced = {k: getattr(fitted, k) for k in ["initial", "transition", "occupancy", "probabilities", "global_probability"]}
                    operator = NeuralOperator(reduced["initial"], reduced["transition"], reduced["occupancy"])
                    null = OccupancyMatchedNull.from_operator(operator)
                    arms.append(("restricted_calibration", reduced, operator, null, cells, name))
                for arm, parameters, operator, null, cells, fit_file in arms:
                    train, held = c["train"], c["held"]
                    ti, hi = np.searchsorted(cells, train), np.searchsorted(cells, held)
                    np.testing.assert_array_equal(cells[ti], train)
                    np.testing.assert_array_equal(cells[hi], held)
                    for eid in fold_meta["test_ids"]:
                        counts = cache[f"counts_{eid}"]
                        label = lookup.loc[(eid, c["split"], c["repeat"], c["fraction"])].to_dict()
                        row = (
                            identity
                            | {k: c[k] for k in ["split", "repeat", "fraction"]}
                            | dict(
                                event_id=eid,
                                arm=arm,
                                fold=fold,
                                fit_file=fit_file,
                                n_full_bins=len(counts),
                                n_train_cells=len(train),
                                n_heldout_cells=len(held),
                                n_train_spikes=int(counts[:, train].sum()),
                                **{k: v for k, v in label.items() if k not in ID},
                            )
                        )
                        if len(counts) > 2:
                            pred = forecast_distributions(counts[:, train], parameters["probabilities"][ti], operator, null, (2,))
                            score = score_predictions(pred, counts[:, held], parameters["probabilities"][hi], parameters["global_probability"][hi])[0]
                            row.update(score, status="scored")
                        else:
                            row.update(status="insufficient_full_bins", n_target_bins=0, n_heldout_target_spikes=0, **{"score_" + b: np.nan for b in ("dynamic", *BASELINES)})
                        rows.append(row)
            frame = pd.DataFrame(rows)
            frame.to_csv(output / f"scores_{fold}.csv.gz", index=False)
            pieces.append(frame)
            print(json.dumps(identity | {"fold": fold, "rows": len(frame), "refits": len(fit_metadata)}), flush=True)
        scores = pd.concat(pieces, ignore_index=True)
        if len(scores) != len(selected) * 5 * (1 + 3 * REPEATS + 3):
            raise ValueError("missing coverage rows")
        # New full/half forecasts must reproduce every previously computed row.
        legacy = pd.read_csv(source / "scores.csv.gz")
        errors = []
        for fraction, level in [(1, "full"), (0.5, "half")]:
            old = legacy[legacy.model.eq("learned_hmm") & legacy.horizon.eq(2) & legacy.level.eq(level)].set_index(["event_id", "split"])
            new = scores[scores.arm.eq("fixed_codebook") & scores.repeat.eq(0) & scores.fraction.eq(fraction)].set_index(["event_id", "split"]).reindex(old.index)
            for col in ["score_dynamic", *["score_" + b for b in BASELINES]]:
                np.testing.assert_allclose(new[col], old[col], rtol=0, atol=1e-8, equal_nan=True)
                errors.append(float(np.nanmax(abs(new[col] - old[col]))))
            expected = old["full_geometric_pass" if fraction == 1 else "half_geometric_pass"]
            if not new.geometric_pass.eq(expected).all():
                raise ValueError("legacy geometry disagreement")
        events = event_tables(scores)
        events.to_csv(output / "events.csv.gz", index=False)
        failures = label_table.groupby(ID + ["fraction", "failure_reason"], as_index=False).size()
        failures.to_csv(output / "failure_reasons.csv", index=False)
        result.update(status="complete", rows=len(scores), refits=len(fit_metadata), maximum_legacy_error=max(errors))
    except Exception as exc:  # noqa: BLE001 - each worker must preserve terminal failure status.
        result.update(status="failed", error=repr(exc))
    finally:
        result["runtime_s"] = time.monotonic() - started
        result["output_sha256"] = {p.name: file_sha256(p) for p in output.iterdir() if p.name != "session_manifest.json"}
        (output / "session_manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--workers", type=int, default=12)
    a = p.parse_args()
    started = time.monotonic()
    source_manifest = a.source_dir / "independent_rejected_forecast_manifest.json"
    source = json.loads(source_manifest.read_text())
    if source["status"] != "complete" or source["subset_debug"] or len(source["completed"]) != 33:
        raise ValueError("audited complete cohort required")
    if any(r["status"] != "complete" for r in source["completed"]):
        raise ValueError("failed source sessions")
    a.output_dir.mkdir(parents=True, exist_ok=False)
    provenance = build_script_provenance(
        input_paths={
            "source_manifest": source_manifest,
            "protocol": ROOT / "docs/coverage_dose_forecast_protocol.md",
            "runner": Path(__file__),
            "module": ROOT / "src/hipporeplayimm/coverage_dose_forecast.py",
        },
        cwd=ROOT,
    )
    if provenance["git_dirty"]:
        raise ValueError("freeze source before real scoring")
    provenance.update(status="running", fractions=FRACTIONS, repeats=REPEATS, forecast_horizon_ms=40, workers=a.workers, completed=[])
    manifest = a.output_dir / "coverage_dose_manifest.json"
    manifest.write_text(json.dumps(provenance, indent=2) + "\n")
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        jobs = [pool.submit(process, record, a.source_dir, a.output_dir) for record in source["completed"]]
        for f in as_completed(jobs):
            result = f.result()
            provenance["completed"].append(result)
            manifest.write_text(json.dumps(provenance, indent=2) + "\n")
            print(json.dumps(result | {"output_sha256": "recorded"}), flush=True)
    status = pd.DataFrame([{k: v for k, v in x.items() if k != "output_sha256"} for x in provenance["completed"]])
    status.to_csv(a.output_dir / "session_status.csv", index=False)
    provenance.update(status="complete" if status.status.eq("complete").all() else "failed", runtime_s=time.monotonic() - started)
    if provenance["status"] == "complete":
        events = pd.concat([pd.read_csv(a.output_dir / x["tag"] / "events.csv.gz") for x in provenance["completed"]], ignore_index=True)
        events.to_csv(a.output_dir / "coverage_events.csv.gz", index=False)
        for name, table in zip(["sessions", "animals", "summary", "leave_one_animal_out"], summaries(events), strict=True):
            table.to_csv(a.output_dir / f"coverage_{name}.csv", index=False)
    provenance["output_sha256"] = {p.name: file_sha256(p) for p in a.output_dir.iterdir() if p.is_file() and p != manifest}
    manifest.write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({k: provenance[k] for k in ["status", "runtime_s"]}), flush=True)
    return 0 if provenance["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
