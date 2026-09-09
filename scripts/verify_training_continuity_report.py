#!/usr/bin/env python3
"""Check group medians, equal-animal estimates and intervals independently."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from _provenance import build_script_provenance, file_sha256
from audit_training_continuity_prediction import checked

ID = ["dataset", "animal", "session"]
KEY = ID + ["event_id", "split"]
SET = ["support", "bin_filter", "min_frames"]
AGG = SET + ["group", "contrast"]


def compare(actual, expected, keys, values):
    if actual.duplicated(keys).any() or expected.duplicated(keys).any():
        raise ValueError("duplicate aggregate keys")
    a, e = actual.set_index(keys).sort_index(), expected.set_index(keys).sort_index()
    if not a.index.equals(e.index) or not np.allclose(a[values].to_numpy(float), e[values].to_numpy(float), atol=1e-9, rtol=0, equal_nan=True):
        raise ValueError("independent aggregate mismatch: " + ",".join(values))


def reference_event_values(labels, predictions):
    contrasts = [f"{m}_{suffix}" for m in ("imm", "diffusion") for suffix in ("minus_iid", "minus_static", "minus_composition", "order_advantage", "order_map_interaction")]
    merged = labels.merge(predictions, on=KEY, validate="many_to_one", how="left", indicator=True)
    if not merged._merge.eq("both").all():
        raise ValueError("missing prediction in independent join")
    f, h = merged.full_training_pass, merged.nested_half_pass
    selections = {
        "all": np.ones(len(merged), bool),
        "geometric_pass": f,
        "geometric_fail": ~f,
        "rejected_with_opportunity": ~f & (merged.full_valid_frames >= merged.min_frames),
        "short_or_unsupported": ~f & (merged.full_valid_frames < merged.min_frames),
        "lost_with_thinning": f & ~h,
        "retained_with_thinning": f & h,
        "gained_with_thinning": ~f & h,
        "rejected_both": ~f & ~h,
    }
    result = []
    for group, keep in selections.items():
        for identity, event in merged[keep].groupby(ID + ["event_id", *SET]):
            scores = event[contrasts].to_numpy()
            counts = event.n_heldout_spikes.to_numpy()[:, None]
            normalized = np.divide(scores, counts, out=np.full_like(scores, np.nan), where=counts > 0)
            medians = np.median(scores, axis=0)
            per_spike = [np.median(c[np.isfinite(c)]) if np.isfinite(c).any() else np.nan for c in normalized.T]
            base = dict(zip(ID + ["event_id", *SET], identity, strict=True))
            for i, contrast in enumerate(contrasts):
                result.append(
                    base | {"group": group, "contrast": contrast, "delta": medians[i], "delta_per_heldout_spike": per_spike[i], "qualifying_splits": event.split.nunique()}
                )
    return pd.DataFrame(result)


def mean(values):
    values = np.asarray(values)
    n = np.isfinite(values).sum(axis=0)
    return np.divide(np.nansum(values, axis=0), n, out=np.full(2, np.nan), where=n > 0)


def reference_interval(events, draws=5000):
    values = ["delta", "delta_per_heldout_spike"]
    by_rat = {}
    for animal in sorted(events.animal.unique()):
        local = events[events.animal.eq(animal)]
        by_rat[animal] = [g.sort_values("event_id")[values].to_numpy() for _, g in local.groupby("session")]
    rats = list(by_rat)
    generator = np.random.default_rng(20260910)
    bootstrap = []
    for _ in range(draws):
        estimates = []
        for animal_index in generator.integers(0, len(rats), len(rats)):
            session_data = by_rat[rats[animal_index]]
            selected_sessions = generator.integers(0, len(session_data), len(session_data))
            session_means = []
            for j in selected_sessions:
                v = session_data[j]
                session_means.append(mean(v[generator.integers(0, len(v), len(v))]))
            estimates.append(mean(session_means))
        bootstrap.append(mean(estimates))
    bootstrap = np.asarray(bootstrap)
    ci = np.full((2, 2), np.nan)
    for i in range(2):
        measured = bootstrap[np.isfinite(bootstrap[:, i]), i]
        if len(measured):
            ci[:, i] = np.quantile(measured, [0.025, 0.975])
    return ci


def audit_setting(setting, run, report, records):
    labels, predictions = [], []
    for r in records:
        table = pd.read_csv(run / f"{r['tag']}_labels.csv.gz")
        mask = table.support.eq(setting[0]) & table.bin_filter.eq(setting[1]) & table.min_frames.eq(setting[2])
        labels.append(table[mask])
        predictions.append(pd.read_csv(run / f"{r['tag']}_predictions.csv.gz"))
    labels, predictions = pd.concat(labels), pd.concat(predictions)
    events = reference_event_values(labels, predictions)
    sessions = (
        events.groupby(ID + AGG)
        .agg(
            delta=("delta", "mean"), delta_per_heldout_spike=("delta_per_heldout_spike", "mean"), events=("event_id", "size"), mean_qualifying_splits=("qualifying_splits", "mean")
        )
        .reset_index()
    )
    animals = (
        sessions.groupby(["dataset", "animal", *AGG])
        .agg(
            delta=("delta", "mean"),
            delta_per_heldout_spike=("delta_per_heldout_spike", "mean"),
            events=("events", "sum"),
            sessions=("session", "size"),
            mean_qualifying_splits=("mean_qualifying_splits", "mean"),
        )
        .reset_index()
    )
    summary = (
        animals.groupby(["dataset", *AGG])
        .agg(
            mean=("delta", "mean"),
            mean_per_heldout_spike=("delta_per_heldout_spike", "mean"),
            animals=("animal", "size"),
            positive_animals=("delta", lambda a: (a > 0).sum()),
            events=("events", "sum"),
            sessions=("sessions", "sum"),
            mean_qualifying_splits=("mean_qualifying_splits", "mean"),
        )
        .reset_index()
    )
    for name, expected, keys in [("by_session", sessions, ID + AGG), ("by_animal", animals, ["dataset", "animal", *AGG]), ("sensitivity_summary", summary, ["dataset", *AGG])]:
        actual = pd.read_csv(report / f"training_continuity_{name}.csv")
        mask = actual.support.eq(setting[0]) & actual.bin_filter.eq(setting[1]) & actual.min_frames.eq(setting[2])
        columns = [c for c in expected if c not in keys]
        compare(actual[mask], expected, keys, columns)
    primary_intervals = []
    if setting == ("parent", "edge_only", 10):
        compare(
            pd.read_csv(report / "training_continuity_primary_event_contrasts.csv.gz"), events, ID + ["event_id", *AGG], ["delta", "delta_per_heldout_spike", "qualifying_splits"]
        )
        selected = events[events.group.isin(["rejected_with_opportunity", "lost_with_thinning"]) & events.contrast.str.startswith("imm_")]
        actual = pd.read_csv(report / "training_continuity_primary_summary.csv").set_index(["dataset", "group", "contrast"])
        for key, group in selected.groupby(["dataset", "group", "contrast"]):
            ci = reference_interval(group)
            values = [ci[0, 0], ci[1, 0], ci[0, 1], ci[1, 1]]
            if not np.allclose(actual.loc[key, ["ci_low", "ci_high", "per_spike_ci_low", "per_spike_ci_high"]].to_numpy(float), values, atol=1e-9, rtol=0, equal_nan=True):
                raise ValueError("hierarchical interval mismatch")
            primary_intervals.append({"dataset": key[0], "group": key[1], "contrast": key[2], "ci_low": values[0], "ci_high": values[1]})
        decisions = pd.read_csv(report / "training_continuity_decisions.csv").set_index(["dataset", "group"])
        for dataset, n in [("pfeiffer_foster", 4), ("tanni2022", 5)]:
            for group in ("rejected_with_opportunity", "lost_with_thinning"):
                panel = actual.reset_index()
                panel = panel[panel.dataset.eq(dataset) & panel.group.eq(group)]
                available = len(panel) == 5 and panel.animals.eq(n).all() and panel.events.gt(0).all()
                passed = available and panel.ci_low.gt(0).all() and panel.positive_animals.eq(n).all()
                row = decisions.loc[(dataset, group)]
                if (
                    bool(row.joint_predictive_support) != bool(passed)
                    or bool(row.all_source_animals_represented) != bool(available)
                    or row.biological_replay_truth_established
                    or row.parent_replication_gate_changed
                ):
                    raise ValueError("unsupported primary decision")
    return {
        "setting": list(setting),
        "event_medians_checked": len(events),
        "session_rows_checked": len(sessions),
        "animal_rows_checked": len(animals),
        "summary_rows_checked": len(summary),
        "intervals": primary_intervals,
    }


def run(args):
    root, report, out = args.run_dir.resolve(), args.report_dir.resolve(), args.output_dir.resolve()
    m = json.loads((root / "training_continuity_manifest.json").read_text())
    rm_path = report / "training_continuity_report_manifest.json"
    rm = json.loads(rm_path.read_text())
    if m.get("status") != "complete" or rm.get("status") != "complete" or rm["input_file_sha256"]["run_manifest"] != file_sha256(root / "training_continuity_manifest.json"):
        raise ValueError("report/source provenance mismatch")
    for directory, meta in ((root, m), (report, rm)):
        for name, digest in meta["output_sha256"].items():
            checked(directory / name, digest)
    for key, digest in rm["input_file_sha256"].items():
        checked(rm["input_file_paths"][key], digest)
    audit = json.loads(Path(rm["input_file_paths"]["classification_audit"]).read_text())
    if audit["status"] != "pass":
        raise ValueError("classification audit not passing")
    out.mkdir(parents=True, exist_ok=False)
    checked_settings, errors = [], []
    settings = [(support, filtering, frames) for support in ("parent", "arena_clipped") for filtering in ("edge_only", "bin_support") for frames in (10, 11)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(audit_setting, s, root, report, m["completed"]): s for s in settings}
        for job in as_completed(jobs):
            try:
                result = job.result()
                checked_settings.append(result)
                print(json.dumps(result), flush=True)
            except Exception as exc:  # noqa: BLE001 -- failed settings must remain visible.
                errors.append({"setting": jobs[job], "error": repr(exc)})
    expected_intervals = len(pd.read_csv(report / "training_continuity_primary_summary.csv"))
    complete = len(checked_settings) == 8 and sum(len(r["intervals"]) for r in checked_settings) == expected_intervals and not errors
    result = build_script_provenance(input_paths={"run_manifest": root / "training_continuity_manifest.json", "report_manifest": rm_path, "auditor": Path(__file__)}, cwd=ROOT)
    result.update(
        status="pass" if complete else "fail",
        settings=checked_settings,
        errors=errors,
        validation_scope="all-setting group medians/session/animal/summary reconstruction and all available primary hierarchical intervals; split0 and clipping-disagreement tables are descriptive, not independently reaggregated here",
    )
    (out / "training_continuity_report_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    if not complete:
        raise RuntimeError("report audit failed")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--report-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--workers", type=int, default=8)
    run(p.parse_args())
