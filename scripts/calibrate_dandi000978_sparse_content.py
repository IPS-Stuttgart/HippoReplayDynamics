#!/usr/bin/env python3
"""Calibrate RUN route readouts at frozen sleep-event spike counts; no replay."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from _provenance import file_sha256
from acquire_dandi000978 import destination, write_json
from audit_dandi000978_cohort import verify_acquisition_record
from dandi000978_coverage_content import (
    composition_scores,
    counts_in_interval,
    provenance,
    spike_lists,
    validate_frozen,
)
from dandi000978_verified_units import read_author_crosswalks, verified_unit_table
from preflight_dandi000978 import inspect
from validate_dandi000978_run_readouts import (
    PARAMETERS as RUN_PARAMETERS,
)
from validate_dandi000978_run_readouts import (
    aggregate_trials,
    epoch_masks,
    fit_routes,
    prepare_bins,
    seed_for,
)

PARAMETERS = {"seed": 20260923, "repeats": 10, "null_draws": 199, "minimum_match_fraction": 0.8}


def thin_exact(counts, target, rng):
    counts = np.asarray(counts)
    if counts.ndim != 1 or not np.isfinite(counts).all() or (counts < 0).any() or not np.equal(counts, counts.astype(np.int64)).all():
        raise ValueError("Nonnegative integer spike vector required")
    if not np.isfinite(target) or target != int(target) or target < 0:
        raise ValueError("Nonnegative integer target required")
    if target > counts.sum():
        return None
    if target == 0:
        return np.zeros(len(counts), dtype=np.int64)
    return rng.multivariate_hypergeometric(counts.astype(np.int64), int(target))


def fit_holdout(bins, counts, epoch):
    labels, totals, exposures = aggregate_trials(bins, counts)
    train, test = epoch_masks(labels, epoch)
    y = labels.route.to_numpy()
    if any(np.sum(y[train] == k) < RUN_PARAMETERS["min_training_trials_per_route"] for k in range(4)):
        raise ValueError("Insufficient training routes")
    if set(y[test]) != set(range(4)):
        raise ValueError("Missing test route")
    use = totals[train].sum(axis=0) >= RUN_PARAMETERS["min_training_spikes"]
    rates = fit_routes(totals[train][:, use], exposures[train], y[train])
    return rates, use, labels.loc[train, "trial_id"].to_numpy(), labels.loc[test, "trial_id"].to_numpy()


def sample_anchors(bins, counts, targets, key):
    """Fixed donor choices precede decoding; unsupported targets remain explicit."""
    if set(bins.route) != set(range(4)):
        raise ValueError("Four known test routes required")
    rng = np.random.default_rng(seed_for(PARAMETERS["seed"], *key, "anchors"))
    groups = bins.groupby("trial_id", sort=True).indices
    trial_routes = bins.groupby("trial_id").route.agg(["first", "nunique"])
    if (trial_routes["nunique"] != 1).any():
        raise ValueError("A trial has multiple routes")
    route_trials = {k: trial_routes[trial_routes["first"] == k].index.to_numpy() for k in range(4)}
    totals = {tid: counts[ix].sum(axis=0) for tid, ix in groups.items()}
    rows, native, sparse = [], [], []
    for target_id, n in enumerate(targets):
        for repeat in range(PARAMETERS["repeats"]):
            for route in range(4):
                tid = int(rng.choice(route_trials[route]))
                ix = int(rng.choice(groups[tid]))
                for scope, c in (("window_250ms", counts[ix]), ("whole_trial_optimistic", totals[tid])):
                    thinned = thin_exact(c, n, rng)
                    rows.append(
                        {
                            "target_index": target_id,
                            "repeat": repeat,
                            "route": route,
                            "trial_id": tid,
                            "window_index": ix,
                            "scope": scope,
                            "target_count": int(n),
                            "native_count": int(c.sum()),
                            "matched": thinned is not None,
                        }
                    )
                    native.append(c)
                    sparse.append(np.zeros_like(c) if thinned is None else thinned)
    return pd.DataFrame(rows), np.asarray(native), np.asarray(sparse)


def tie_credit(prob):
    ties = np.isclose(prob, prob.max(axis=1, keepdims=True), rtol=0, atol=1e-12)
    return ties / ties.sum(axis=1, keepdims=True)


def trial_null_labels(trial_ids, routes, key):
    ids, first = np.unique(trial_ids, return_index=True)
    truth = np.asarray(routes)[first]
    lookup = {tid: i for i, tid in enumerate(ids)}
    inverse = np.asarray([lookup[tid] for tid in trial_ids])
    if not np.array_equal(truth[inverse], routes):
        raise ValueError("Inconsistent donor-trial route")
    rng = np.random.default_rng(seed_for(PARAMETERS["seed"], *key, "trial_label_null"))
    shuffled = np.array([rng.permutation(truth) for _ in range(PARAMETERS["null_draws"])])
    return ids, shuffled, inverse


def evaluate_anchors(frame, native, sparse, rates, key):
    ids, null_labels, inverse = trial_null_labels(frame.trial_id.to_numpy(), frame.route.to_numpy(), key)
    rows, null_rows = [], []
    for scope, g in frame.groupby("scope", sort=True):
        ix = g.index[g.matched].to_numpy()
        if not len(ix):
            raise ValueError("No matched anchors")
        truth = frame.loc[ix, "route"].to_numpy(int)
        for regime, all_counts in (("native_matched_anchors", native), ("sleep_count_matched", sparse)):
            contrasts, prob = composition_scores(all_counts[ix], rates)
            accuracy = tie_credit(prob)
            arange = np.arange(len(ix))
            actual_score = contrasts[arange, truth].mean()
            actual_accuracy = accuracy[arange, truth].mean()
            null_score = np.array([contrasts[arange, draw[inverse[ix]]].mean() for draw in null_labels])
            null_accuracy = np.array([accuracy[arange, draw[inverse[ix]]].mean() for draw in null_labels])
            rows.append(
                {
                    "scope": scope,
                    "regime": regime,
                    "status": "scored",
                    "attempted_anchors": len(g),
                    "matched_anchors": len(ix),
                    "match_fraction": len(ix) / len(g),
                    "n_donor_trials": frame.loc[ix, "trial_id"].nunique(),
                    "accuracy": actual_accuracy,
                    "centered_score": actual_score,
                    "zero_fraction": float((all_counts[ix].sum(axis=1) == 0).mean()),
                    "median_spikes": float(np.median(all_counts[ix].sum(axis=1))),
                    "null_accuracy_p95": float(np.quantile(null_accuracy, 0.95)),
                    "null_score_p95": float(np.quantile(null_score, 0.95)),
                }
            )
            null_rows.extend(
                {"scope": scope, "regime": regime, "draw": j, "accuracy": a, "centered_score": s} for j, (a, s) in enumerate(zip(null_accuracy, null_score, strict=True))
            )
    return rows, null_rows, {"null_trial_ids": ids, "null_labels": null_labels, "donor_inverse": inverse}


def summarize(folds, nulls, expected_folds, out):
    keys = ["animal", "region", "scope", "regime"]
    metrics = ["accuracy", "centered_score", "match_fraction", "zero_fraction", "median_spikes"]
    means = folds.groupby(keys)[metrics].mean().reset_index()
    n = nulls.groupby(keys + ["draw"])[["accuracy", "centered_score"]].mean().reset_index()
    records = []
    for row in means.to_dict("records"):
        mask = np.ones(len(n), dtype=bool)
        for key in keys:
            mask &= n[key] == row[key]
        group = n[mask]
        for metric in ("accuracy", "centered_score"):
            values = group[metric].to_numpy()
            row[f"null_{metric}_mean"] = float(values.mean())
            row[f"null_{metric}_p95"] = float(np.quantile(values, 0.95))
            row[f"{metric}_excess"] = float(row[metric] - values.mean())
            row[f"{metric}_p"] = float((1 + (values >= row[metric]).sum()) / (1 + len(values)))
        records.append(row)
    summary = pd.DataFrame(records)
    summary.to_csv(out / "animal_summary.csv", index=False)
    n.to_csv(out / "animal_null_summary.csv", index=False)
    actual = set(zip(folds.file, folds.heldout_epoch, folds.region))
    complete = (
        bool(expected_folds)
        and actual == set(expected_folds)
        and len(folds) == 4 * len(expected_folds)
        and (folds.status == "scored").all()
        and not folds.duplicated(["file", "heldout_epoch", "region", "scope", "regime"]).any()
        and np.isfinite(folds[metrics].to_numpy()).all()
    )
    primary = summary[(summary.region == "PFC") & (summary.scope == "window_250ms") & (summary.regime == "sleep_count_matched")]
    calibration = bool(
        complete
        and set(primary.animal) == {"JS14", "ZT2"}
        and (primary.match_fraction >= PARAMETERS["minimum_match_fraction"]).all()
        and (primary.accuracy > primary.null_accuracy_p95).all()
        and (primary.centered_score > primary.null_centered_score_p95).all()
    )
    gates = [
        {"gate": "all_expected_folds_complete", "passed": bool(complete)},
        {"gate": "PFC_sparse_RUN_readout_detectable_both_animals", "passed": calibration},
        {"gate": "sleep_content_validated", "passed": False},
        {"gate": "paper_ready", "passed": False},
    ]
    pd.DataFrame(gates).to_csv(out / "gate_summary.csv", index=False)
    decision = "sparse_RUN_readout_detectable_sleep_transfer_unresolved" if calibration else "sparse_RUN_readout_not_established_both_animals"
    write_json(out / "decision.json", {"decision": decision, "technical_complete": bool(complete), "paper_ready": False})
    lines = [
        "# Sparse-count RUN calibration",
        "",
        f"Decision: `{decision}`.",
        "",
        "Two animals; whole RUN epochs held out; ZT2 files are not independent rats.",
        "No replay evidence rescored and no sleep trajectory labels changed.",
        "",
        "```csv",
        summary.to_csv(index=False).strip(),
        "```",
        "",
        "## Interpretation limits",
        "",
        "Count matching is without replacement; unmatched anchors are explicit, not capped or fabricated.",
        "Native and sparse scores use the same matched anchors. Short-window matching conditions on sufficient RUN spikes.",
        "Whole-trial matching is an optimistic route-composition control, not a brief replay model.",
        "Nulls permute routes at the donor-trial level, shared across repeated uses; repeats are not independent animals.",
        "Animal summaries equally weight RUN epochs; calibration includes other, possibly later, RUN training epochs.",
        "Even a positive result cannot establish RUN-to-sleep template validity, true sleep content or inter-area coordination.",
    ]
    (out / "calibration_summary.md").write_text("\n".join(lines) + "\n")
    return bool(complete)


def figure(summary, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, animal in zip(axes, ("JS14", "ZT2"), strict=True):
        group = summary[(summary.animal == animal) & (summary.region == "PFC")]
        x = np.arange(2)
        for offset, (regime, label, color) in zip(
            (-0.18, 0.18), (("native_matched_anchors", "Original anchor counts", "#17857c"), ("sleep_count_matched", "Sleep-matched counts", "#b55349")), strict=True
        ):
            rows = group[group.regime == regime].set_index("scope").loc[["window_250ms", "whole_trial_optimistic"]]
            ax.bar(x + offset, rows.accuracy, 0.34, label=label, color=color)
            ax.scatter(x + offset, rows.null_accuracy_p95, marker="_", s=130, color="black", zorder=3)
        ax.axhline(0.25, color="gray", linestyle=":", linewidth=1)
        ax.set(xticks=x, xticklabels=["250 ms RUN", "Whole trial\n(optimistic)"], title=animal, ylim=(0, 1))
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Known-route accuracy (four routes)")
    axes[1].legend(fontsize=8, loc="upper left")
    fig.suptitle("PFC readout calibration, not sleep replay validation")
    fig.text(0.5, 0.02, "Black ticks: trial-label null p95; not confidence intervals. Dotted line: 25% chance.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(out / "sparse_content_calibration.png", dpi=180)
    plt.close(fig)


def run(args):
    frozen_path = args.frozen_dir / "frozen_manifest.json"
    frozen = json.loads(frozen_path.read_text())
    validate_frozen(args.frozen_dir, frozen, frozen["code_commit"])
    asset_path = args.dataset_root / "metadata/asset_manifest.json"
    acquisition_path = args.dataset_root / "download_status.json"
    source = json.loads(asset_path.read_text())
    verify_acquisition_record(args.dataset_root, source, json.loads(acquisition_path.read_text()))
    if file_sha256(args.author_archive) != frozen["input_file_sha256"]["private_source"]:
        raise ValueError("Private source archive changed")
    events = pd.read_csv(args.frozen_dir / "selected_events.csv")
    if not len(events) or events.event_id.duplicated().any():
        raise ValueError("Empty or duplicate frozen candidate set")
    prov = provenance(
        {"frozen": frozen_path, "events": args.frozen_dir / "selected_events.csv", "private_source": args.author_archive, "assets": asset_path, "acquisition": acquisition_path}
    )
    args.output_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
    out = args.output_dir
    write_json(
        out / "manifest.json", {**prov, "parameters": PARAMETERS, "run_parameters": RUN_PARAMETERS, "scope": "RUN_count_calibration_no_replay", "private_inputs_required": True}
    )
    crosswalks = read_author_crosswalks(args.author_archive)
    rows, null_rows, status, distribution, expected = [], [], [], [], []
    for (filename, epoch), _ in events.groupby(["file", "training_run_epoch"]):
        expected.extend((filename, int(epoch), region) for region in ("CA1", "PFC"))
    for asset in source["assets"]:
        filename = Path(asset["path"]).name
        selected = events[events.file == filename]
        if selected.empty:
            continue
        animal = selected.animal.iloc[0]
        with h5py.File(destination(args.dataset_root, asset), "r") as f:
            _, _, trials, _, arrays = inspect(f)
            table = f[f["units/electrodes"].attrs["table"]]
            units = verified_unit_table(
                animal, filename, arrays["unit_ids"], f["units/electrodes"][()], table["location"].asstr()[()], table["group_name"].asstr()[()], crosswalks.get(filename)
            )
        bins, counts, _ = prepare_bins(arrays, trials)
        for epoch, targets in selected.groupby("training_run_epoch", sort=True):
            for region in ("CA1", "PFC"):
                base = {"animal": animal, "file": filename, "heldout_epoch": int(epoch), "region": region}
                key = (filename, int(epoch), region)
                try:
                    regional = counts[:, units.region.to_numpy() == region]
                    rates, use, training_ids, test_ids = fit_holdout(bins, regional, epoch)
                    if use.sum() < (20 if region == "CA1" else 5):
                        raise ValueError("Too few training-only regional units")
                    unit_ids = units.loc[units.region == region, "unit_id"].to_numpy()[use]
                    spikes = spike_lists(arrays, unit_ids)
                    target_counts = np.array([counts_in_interval(spikes, t.start_time_s, t.end_time_s).sum() for t in targets.itertuples(index=False)])
                    test = (bins.epoch.to_numpy() == epoch) & (bins.route.to_numpy() >= 0)
                    test_bins = bins.loc[test].reset_index(drop=True)
                    test_counts = regional[test][:, use]
                    frame, native, sparse = sample_anchors(test_bins, test_counts, target_counts, key)
                    metrics, null, audit = evaluate_anchors(frame, native, sparse, rates, key)
                    dest = out / asset["asset_id"] / region / f"epoch_{epoch}"
                    dest.mkdir(parents=True)
                    frame.to_csv(dest / "anchors.csv", index=False)
                    np.savez_compressed(
                        dest / "audit.npz",
                        native=native,
                        sparse=sparse,
                        rates=rates,
                        source_unit_ids=unit_ids,
                        target_counts=target_counts,
                        training_trial_ids=training_ids,
                        test_trial_ids=test_ids,
                        target_event_ids=targets.event_id.to_numpy(str),
                        **audit,
                    )
                    rows.extend({**base, **m, "encoding_units": int(use.sum())} for m in metrics)
                    null_rows.extend({**base, **m} for m in null)
                    for kind, c in (("sleep_targets", target_counts), ("RUN_windows", test_counts.sum(axis=1))):
                        distribution.append(
                            {
                                **base,
                                "kind": kind,
                                "n": len(c),
                                "p10": float(np.quantile(c, 0.1)),
                                "median": float(np.median(c)),
                                "p90": float(np.quantile(c, 0.9)),
                                "zero_fraction": float((c == 0).mean()),
                            }
                        )
                    status.append({**base, "status": "scored", "failure_reason": ""})
                except (ValueError, KeyError) as exc:
                    status.append({**base, "status": "failed", "failure_reason": str(exc)})
                pd.DataFrame(status).to_csv(out / "fold_status.csv", index=False)
                pd.DataFrame(rows).to_csv(out / "fold_metrics.csv", index=False)
                print(json.dumps(status[-1]), flush=True)
    if not rows:
        raise ValueError("No calibration folds completed")
    folds, nulls = pd.DataFrame(rows), pd.DataFrame(null_rows)
    nulls.to_csv(out / "fold_nulls.csv", index=False)
    pd.DataFrame(distribution).to_csv(out / "count_distributions.csv", index=False)
    complete = summarize(folds, nulls, expected, out)
    figure(pd.read_csv(out / "animal_summary.csv"), out)
    write_json(out / "terminal_status.json", {"status": "complete" if complete else "incomplete", "scoring_complete": complete})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--frozen-dir", type=Path, required=True)
    parser.add_argument("--author-archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args)
    except Exception as exc:
        if args.output_dir.exists() and not (args.output_dir / "terminal_status.json").exists():
            write_json(args.output_dir / "terminal_status.json", {"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        raise


if __name__ == "__main__":
    main()
