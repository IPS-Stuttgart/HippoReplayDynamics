#!/usr/bin/env python3
"""Redetect independent MUA candidates and forecast evaluation-cell identities."""

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

from hipporeplayimm.independent_rejected_forecast import (
    BASELINES,
    ID,
    aggregate,
    classify,
    detect_candidates,
    detection_grid,
    event_counts,
    forecast_distributions,
    full_counts,
    partitions,
    score_predictions,
    seed,
)
from hipporeplayimm.lagged_neural_prediction import NeuralOperator, SpatialOperator
from hipporeplayimm.learned_assembly_prediction import fit_learned_assembly
from hipporeplayimm.occupancy_matched_forecast import OccupancyMatchedNull
from scripts._provenance import build_script_provenance, file_sha256
from scripts.audit_2d_count_conditioned_prediction import folds

SOURCE_MANIFEST_SHA = "2ed1c4d910a3ddc4027eefb398d48a55bae6829c3bf16e8f57955beb9ee55613"
NAMES = ("splits", "events", "sessions", "animals", "summary", "leave_one_animal_out")
HORIZONS = (1, 2, 4)
MODELS = ("learned_hmm", "spatial_diffusion")
STATUS_FIELDS = ["event_id", "start_s", "end_s", "peak_s", "duration_s", "detector_spikes", "detector_active_cells", "detector_peak_z", "mean_speed_cm_s"]


def checked(path, digest):
    if file_sha256(path) != digest:
        raise ValueError("changed input: " + str(path))


def prepare_session(record, output):
    source = Path(record["artifact_path"])
    checked(source, record["artifact_sha256"])
    with np.load(source, allow_pickle=False) as z:
        raw = {
            k: z[k]
            for k in [
                "spikes",
                "position",
                "supported_run_intervals",
                "cell_ids",
                "unit_qc_mask",
                "valid_spatial_bins",
                "rates_hz",
                "bin_centers_cm",
            ]
        }
    identity = tuple(record[k] for k in ID)
    units = raw["cell_ids"][raw["unit_qc_mask"].astype(bool)]
    rates = raw["rates_hz"][raw["unit_qc_mask"].astype(bool)][:, raw["valid_spatial_bins"].astype(bool)]
    centers = raw["bin_centers_cm"][raw["valid_spatial_bins"].astype(bool)]
    det, pop, splits = partitions(len(units), identity)
    clock, speed = detection_grid(raw["position"], raw["supported_run_intervals"])
    candidates, diagnostics = detect_candidates(raw["spikes"], units[det], clock, speed)
    selected = pd.DataFrame(candidates, columns=STATUS_FIELDS)
    selected.to_csv(output / "selection.csv", index=False)
    saved = dict(detector_unit_ids=units[det], unit_ids=units[pop], rates=rates[pop], centers=centers)
    for split, (tr, half, held) in enumerate(splits):
        saved.update({f"train_{split}": tr, f"half_{split}": half, f"held_{split}": held})
    indexed = {int(i): np.sort(raw["spikes"][raw["spikes"][:, 1] == i, 0]) for i in units[pop]}
    for e in selected.itertuples(index=False):
        base, durations = event_counts(indexed, units[pop], e.start_s, e.end_s)
        saved[f"base_{e.event_id}"], saved[f"durations_{e.event_id}"] = base, durations
        counts, discarded = full_counts(base, durations)
        saved[f"counts_{e.event_id}"] = counts
        saved[f"discarded_{e.event_id}"] = np.array(discarded)
    np.savez_compressed(output / "cache.npz", **saved)
    checked(source, record["artifact_sha256"])
    diagnostics.update(
        n_detector_cells=len(det),
        n_population_cells=len(pop),
        n_candidates=len(selected),
        detector_uses_evaluation_cells=False,
        full_session_source_sha256=record["artifact_sha256"],
    )
    (output / "detection.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    return selected, saved


def process(record, root):
    identity = {k: record[k] for k in ID}
    tag = "__".join(str(identity[k]).replace("/", "_") for k in ID)
    output = Path(root) / tag
    output.mkdir(exist_ok=False)
    started = time.monotonic()
    result = identity | dict(tag=tag, status="running")
    try:
        selected, cache = prepare_session(record, output)
        result["events"] = len(selected)
        if len(selected) < 5:
            result["status"] = "insufficient_candidates_for_five_folds"
            return result
        rows, labels, fit_records, paths = [], [], [], {}
        spatial = SpatialOperator(cache["centers"], imm=False)
        spatial_null = OccupancyMatchedNull.from_operator(spatial)
        (output / "spatial_null.json").write_text(json.dumps(spatial_null.diagnostics(), indent=2) + "\n")
        for e in selected.itertuples(index=False):
            for split in range(5):
                tr, half = cache[f"train_{split}"], cache[f"half_{split}"]
                label, fullpath, halfpath = classify(
                    cache[f"base_{e.event_id}"],
                    cache[f"durations_{e.event_id}"],
                    cache["rates"],
                    cache["centers"],
                    tr,
                    half,
                )
                labels.append(identity | dict(event_id=e.event_id, split=split) | label)
                paths[f"full_{e.event_id}_{split}"] = fullpath
                paths[f"half_{e.event_id}_{split}"] = halfpath
        label_table = pd.DataFrame(labels).set_index(["event_id", "split"])
        label_table.reset_index().to_csv(output / "labels.csv.gz", index=False)
        np.savez_compressed(output / "paths.npz", **paths)
        for fold, test, cal, excluded in folds(selected):
            sequences = [cache[f"counts_{eid}"] for eid in cal.event_id]
            fit = fit_learned_assembly(sequences, 50, seed(*identity.values(), "fit", fold))
            fit_meta = dict(
                fold=fold,
                test_ids=test.event_id.tolist(),
                calibration_ids=cal.event_id.tolist(),
                excluded_ids=excluded.event_id.tolist(),
                converged=bool(fit.converged),
                restart=int(fit.restart),
                restart_objectives=list(fit.restart_objectives),
                restart_converged=list(fit.restart_converged),
                initialization_with_replacement=bool(fit.initialization_with_replacement),
                n_calibration_bins=int(fit.n_calibration_bins),
            )
            fit_records.append(fit_meta)
            (output / "folds.json").write_text(json.dumps(fit_records, indent=2) + "\n")
            np.savez_compressed(
                output / f"fit_{fold}.npz",
                probabilities=fit.probabilities,
                initial=fit.initial,
                transition=fit.transition,
                occupancy=fit.occupancy,
                global_probability=fit.global_probability,
                objective_trace=fit.objective_trace,
            )
            if not fit.converged:
                raise ValueError("unconverged frozen K50 model")
            neural = NeuralOperator(fit.initial, fit.transition, fit.occupancy)
            matched = OccupancyMatchedNull.from_operator(neural)
            (output / f"neural_null_{fold}.json").write_text(json.dumps(matched.diagnostics(), indent=2) + "\n")
            for e in test.itertuples(index=False):
                eid = int(e.event_id)
                counts = cache[f"counts_{eid}"]
                for split in range(5):
                    held = cache[f"held_{split}"]
                    label = label_table.loc[(eid, split)].to_dict()
                    for level in ("full", "half"):
                        train = cache[f"train_{split}" if level == "full" else f"half_{split}"]
                        for model, emissions, operator, null in [
                            ("learned_hmm", fit.probabilities, neural, matched),
                            ("spatial_diffusion", cache["rates"], spatial, spatial_null),
                        ]:
                            forecasts = forecast_distributions(counts[:, train], emissions[train], operator, null)
                            scores = {
                                r["horizon"]: r
                                for r in score_predictions(
                                    forecasts,
                                    counts[:, held],
                                    emissions[held],
                                    fit.global_probability[held],
                                )
                            }
                            for h in HORIZONS:
                                common = (
                                    identity
                                    | dict(
                                        event_id=eid,
                                        split=split,
                                        fold=fold,
                                        level=level,
                                        model=model,
                                        horizon=h,
                                        duration_s=e.duration_s,
                                        n_full_bins=len(counts),
                                        n_train_cells=len(train),
                                        n_heldout_cells=len(held),
                                        n_train_spikes=int(counts[:, train].sum()),
                                        discarded_partial_spikes=int(cache[f"discarded_{eid}"]),
                                        forecast_uses_heldout=False,
                                        detector_uses_heldout=False,
                                        classification_uses_heldout=False,
                                        forecast_uses_future=False,
                                    )
                                    | {k: v for k, v in label.items() if k not in ID}
                                )
                                if h in scores:
                                    rows.append(common | scores[h] | dict(status="scored"))
                                else:
                                    rows.append(
                                        common
                                        | dict(
                                            status="insufficient_full_bins",
                                            n_target_bins=0,
                                            n_heldout_target_spikes=0,
                                            **{"score_" + k: np.nan for k in ("dynamic", *BASELINES)},
                                        )
                                    )
            pd.DataFrame(rows).to_csv(output / "scores.csv.gz", index=False)
            print(json.dumps(identity | dict(fold=fold, events=len(test), cumulative_rows=len(rows))), flush=True)
        expected = len(selected) * 5 * 2 * 2 * 3
        if len(rows) != expected:
            raise ValueError("incomplete score rows")
        result.update(status="complete", rows=len(rows))
    except Exception as exc:
        result.update(status="failed", error=repr(exc))
    finally:
        result["runtime_s"] = time.monotonic() - started
        result["output_sha256"] = {p.name: file_sha256(p) for p in output.iterdir() if p.name != "session_manifest.json"}
        (output / "session_manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def validate(rows, completed):
    expected = sum(x["events"] * 60 for x in completed if x["status"] == "complete")
    if rows.empty or len(rows) != expected or rows.duplicated(ID + ["event_id", "split", "level", "model", "horizon"]).any():
        raise ValueError("empty, incomplete or duplicate output")
    for field in ["forecast_uses_heldout", "detector_uses_heldout", "classification_uses_heldout", "forecast_uses_future"]:
        if rows[field].any():
            raise ValueError("leakage declaration")
    ok = rows.status.eq("scored")
    if not ok.eq(rows.n_full_bins > rows.horizon).all():
        raise ValueError("inconsistent temporal status")
    columns = ["score_" + k for k in ("dynamic", *BASELINES)]
    if not np.isfinite(rows.loc[ok, columns]).all().all() or (rows.loc[ok, columns] > 1e-8).any().any():
        raise ValueError("invalid proper predictive scores")
    if not rows.loc[~ok, columns].isna().all().all():
        raise ValueError("short event got scores")


def decisions(summary, completed):
    result = []
    for dataset, expected_animals, expected_sessions in [("pfeiffer_foster", 4, 8), ("tanni2022", 5, 25)]:
        sessions = [r for r in completed if r["dataset"] == dataset]
        technical = len(sessions) == expected_sessions and all(r["status"] == "complete" for r in sessions)
        primary = summary[
            summary.dataset.eq(dataset)
            & summary.level.eq("full")
            & summary.model.eq("learned_hmm")
            & summary.horizon.eq(2)
            & summary.group.eq("rejected_with_opportunity")
            & summary.metric.eq("delta_per_spike")
        ]
        required = ["dynamic_minus_matched_own", "dynamic_minus_global", "dynamic_minus_no_history"]
        tests = primary[primary.contrast.isin(required)]
        passed = bool(
            technical
            and len(tests) == 3
            and set(tests.contrast) == set(required)
            and tests.animals.eq(expected_animals).all()
            and tests.positive_animals.eq(expected_animals).all()
            and tests.ci_low.gt(0).all()
        )
        result.append(
            dict(
                dataset=dataset,
                technical_complete=technical,
                primary_and_adequacy_supported=passed,
                biological_replication=False,
                evidence_role="retrospective_independent_neuron_validation",
                uniform_speed_established=False,
                rejected_events_are_true_replay=False,
            )
        )
    return pd.DataFrame(result)


def report(root, completed):
    tables = []
    for r in completed:
        if r["status"] == "complete":
            tables.append(pd.read_csv(root / r["tag"] / "scores.csv.gz"))
    if not tables:
        raise ValueError("no scored sessions")
    rows = pd.concat(tables, ignore_index=True)
    validate(rows, completed)
    results = aggregate(rows)
    for name, table in zip(NAMES, results, strict=True):
        table.to_csv(root / f"rejected_forecast_{name}.csv.gz", index=False)
    decision = decisions(results[-2], completed)
    decision.to_csv(root / "rejected_forecast_decisions.csv", index=False)
    # Keep temporal and zero-target exclusions, not only successful primary rows.
    coverage = (
        rows[rows.level.eq("full") & rows.model.eq("learned_hmm")]
        .groupby(ID + ["horizon"], as_index=False)
        .agg(
            event_splits=("event_id", "size"),
            unique_events=("event_id", "nunique"),
            scored_splits=("status", lambda x: int(x.eq("scored").sum())),
            rejected_splits=("rejected_with_opportunity", "sum"),
            zero_target_splits=("n_heldout_target_spikes", lambda x: int(x.eq(0).sum())),
        )
    )
    coverage.to_csv(root / "rejected_forecast_coverage.csv", index=False)
    pd.DataFrame(completed).drop(columns=["output_sha256"], errors="ignore").to_csv(root / "rejected_forecast_session_status.csv", index=False)
    main = results[-2]
    main = main[main.level.eq("full") & main.model.eq("learned_hmm") & main.horizon.eq(2) & main.group.eq("rejected_with_opportunity") & main.metric.eq("delta_per_spike")]
    lines = [
        "# Independently detected, trajectory-rejected event forecasts",
        "",
        "Retrospective analysis on previously explored recordings. No biological replay ground truth.",
        "A reserved detector-cell pool supplies all event boundaries; evaluation-cell spikes never enter",
        "detection, geometry or test-event inference. The primary target is 40 ms ahead across a 20-ms gap.",
        "",
        "Primary: learned HMM minus independently filtered occupancy/dwell-matched null;",
        "per-event median split score per target spike, then equal sessions and animals.",
        "",
        "| Dataset | Contrast | Mean nats/spike | Animal-bootstrap 95% interval | Positive animals |",
        "| --- | --- | ---: | --- | ---: |",
    ]
    for r in main.itertuples(index=False):
        lines.append(f"| {r.dataset} | {r.contrast} | {r.mean:+.6f} | [{r.ci_low:+.6f}, {r.ci_high:+.6f}] | {r.positive_animals}/{r.animals} |")
    lines += ["", "## Decisions", ""]
    for r in decision.itertuples(index=False):
        lines.append(f"- {r.dataset}: technical complete={r.technical_complete}; primary plus adequacy controls supported={r.primary_and_adequacy_supported}.")
    lines += [
        "",
        "## Boundaries",
        "",
        "- A group-level positive forecast does not certify individual events or continuous spatial replay.",
        "- Four PF and five Tanni animals are the replication units; conditional bootstrap intervals are descriptive.",
        "- Geometry-pass is not the original full shuffle-validated criterion. Geometry-fail cannot pass its conjunction.",
        "- Detection now uses a subset of RUN-QC cells; this is a different candidate population from the all-cell archive.",
        "- All-cell calibration is confined to separate events; target-event held-out spikes are used only in scoring.",
        "- Lost-with-thinning is reported for BOTH full and smaller inference populations, using identical evaluation cells.",
        "- Unnormalized and per-bin results, shorter/longer horizons and spatial diffusion remain sensitivities.",
        "- Do not replace a failed primary with a favorable model, group, animal exclusion or normalization.",
        "- No physical speed, behavioral purpose, independent biological replication or novelty is established.",
        "- Independent reconstruction audit is required before scientific interpretation.",
        "",
    ]
    (root / "README.md").write_text("\n".join(lines))
    return decision


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, default=Path("/mnt/seagate10tb/florianpfaff/replay-coverage-real-inputs-all33-20260905"))
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--session", action="append", help="Explicit debugging subset; not a full-cohort result")
    a = p.parse_args()
    if a.workers < 1:
        p.error("workers must be positive")
    checked(a.input_dir / "coverage_input_manifest.json", SOURCE_MANIFEST_SHA)
    records = pd.read_csv(a.input_dir / "coverage_input_sessions.csv").sort_values(ID).to_dict("records")
    if len(records) != 33 or not all(r["status"] == "cached" for r in records):
        raise ValueError("wrong source cohort")
    if a.session:
        requested = set(a.session)
        records = [r for r in records if r["dataset"] + ":" + r["session"] in requested]
        if len(records) != len(requested):
            raise ValueError("requested sessions missing or ambiguous")
    inputs = {
        "source_manifest": a.input_dir / "coverage_input_manifest.json",
        "source_sessions": a.input_dir / "coverage_input_sessions.csv",
        "protocol": ROOT / "docs/independent_rejected_forecast_protocol.md",
        "producer": Path(__file__),
        "kernel": ROOT / "src/hipporeplayimm/independent_rejected_forecast.py",
    }
    for r in records:
        checked(r["artifact_path"], r["artifact_sha256"])
        inputs[r["artifact_path"]] = Path(r["artifact_path"])
    for name in [
        "training_continuity",
        "lagged_neural_prediction",
        "occupancy_matched_forecast",
        "learned_assembly_prediction",
        "conditional_spatial_prediction",
        "frozen_posterior_prediction",
    ]:
        inputs[name] = ROOT / f"src/hipporeplayimm/{name}.py"
    inputs["fold_logic"] = ROOT / "scripts/audit_2d_count_conditioned_prediction.py"
    m = build_script_provenance(input_paths=inputs, cwd=ROOT)
    if m["git_dirty"] or m["code_commit"] == "unavailable":
        raise ValueError("commit code before scoring")
    out = a.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    mp = out / "independent_rejected_forecast_manifest.json"
    m.update(
        status="running",
        primary_horizon=2,
        primary_group="rejected_with_opportunity",
        independent_event_detection=True,
        source_candidates_reused=False,
        expected_sessions=len(records),
        subset_debug=bool(a.session),
        workers=a.workers,
    )
    mp.write_text(json.dumps(m, indent=2) + "\n")
    tick, completed = time.monotonic(), []
    try:
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            futures = {pool.submit(process, r, out): r for r in records}
            for future in as_completed(futures):
                result = future.result()
                completed.append(result)
                print(json.dumps({k: v for k, v in result.items() if k != "output_sha256"}), flush=True)
                m["completed"] = completed
                mp.write_text(json.dumps(m, indent=2) + "\n")
        decision = report(out, completed)
        m["decisions"] = decision.to_dict("records")
        if len(completed) != len(records) or any(r["status"] != "complete" for r in completed):
            raise ValueError("incomplete sessions; inspect explicit statuses")
        for name, path in inputs.items():
            checked(path, m["input_file_sha256"][name])
        m["status"] = "complete"
    except BaseException as exc:
        m.update(status="failed", error=repr(exc))
        raise
    finally:
        m.update(completed=completed, runtime_s=time.monotonic() - tick)
        m["output_sha256"] = {str(q.relative_to(out)): file_sha256(q) for q in out.rglob("*") if q.is_file() and q != mp}
        mp.write_text(json.dumps(m, indent=2) + "\n")


if __name__ == "__main__":
    main()
