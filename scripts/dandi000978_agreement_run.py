#!/usr/bin/env python3
"""Read-only, provisional-group RUN route pilot. No replay or anatomical claims.

Units enter A/B only when both previously audited indexing interpretations agree.
Each sample is a complete trial. All models and nulls use identical whole-trial
and whole-epoch held-out folds. Confidence calibration uses training trials only
and a prespecified temperature grid.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

MODELS = ("conditional_identity", "poisson_population", "count_duration")
NULLS = ("within_epoch_permutation", "within_epoch_circular_shift")
ALPHA = 0.5
TEMPERATURES = np.array([1., 2., 4., 8., 16., 32., 64., 128., 256., 512., 1024.])


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, obj: dict | list) -> None:
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def agreement_groups(units: pd.DataFrame, unit_ids: np.ndarray) -> tuple[dict, pd.DataFrame]:
    """Use metadata only; unresolved/disputed units never enter either group."""
    if units.unit_id.duplicated().any() or len(set(unit_ids.tolist())) != len(unit_ids):
        raise ValueError("Non-unique file-local unit IDs")
    if set(units.unit_id) != set(unit_ids.tolist()):
        raise ValueError("Unit audit and spike arrays do not have identical IDs")
    u = units.set_index("unit_id").loc[unit_ids].reset_index()
    alt = u.alternative_tetrode_id_region_not_author_confirmed
    u["provisional_group"] = "excluded"
    for label, token in (("A", "CA1"), ("B", "PFC")):
        mask = u.nwb_row_region.eq(token) & alt.eq(token)
        u.loc[mask, "provisional_group"] = label
    groups = {k: np.flatnonzero(u.provisional_group.eq(k).to_numpy()) for k in ("A", "B")}
    if np.intersect1d(groups["A"], groups["B"]).size:
        raise ValueError("Validation groups overlap")
    if min(map(len, groups.values())) < 3:
        raise ValueError("Fewer than three metadata-agreement units in a group")
    u["anatomy_confirmed"] = False
    return groups, u


def prepare_trials(trials: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    t = trials.sort_values(["start_time", "id"]).reset_index(drop=True).copy()
    if t.id.duplicated().any():
        raise ValueError("Duplicate trial IDs")
    a, b = t.start_time.to_numpy(float), t.stop_time.to_numpy(float)
    if not np.isfinite(np.r_[a, b]).all() or (b <= a).any():
        raise ValueError("Invalid trial times")
    if len(t) > 1 and (a[1:] < b[:-1] - 1e-6).any():
        raise ValueError("Overlapping trials would duplicate test spikes")
    wells = t[["start_well", "end_well"]].to_numpy(float)
    valid = np.isfinite(wells).all(axis=1)
    valid &= (wells > 0).all(axis=1) & (wells == np.floor(wells)).all(axis=1)
    valid &= wells[:, 0] != wells[:, 1]
    t["included"] = valid
    t["exclusion_reason"] = np.where(valid, "", "missing_nonpositive_noninteger_or_same_well")
    q = t.loc[valid].copy().reset_index(drop=True)
    q["route"] = [f"{int(a)}->{int(b)}" for a, b in q[["start_well", "end_well"]].to_numpy()]
    q["duration_s"] = q.stop_time - q.start_time
    if len(q) < 30 or q.epoch_index.nunique() < 3 or q.route.nunique() < 2:
        raise ValueError("Insufficient trial/epoch/route support for the fixed pilot")
    # Correctness and trajectory_type are retained for auditing, not selection.
    return q, t


def trial_counts(arrays: dict, trials: pd.DataFrame) -> np.ndarray:
    starts = np.r_[0, arrays["spike_ends"][:-1]].astype(np.int64)
    ends = np.asarray(arrays["spike_ends"], dtype=np.int64)
    if len(starts) != len(arrays["unit_ids"]) or not len(ends):
        raise ValueError("Invalid ragged spike dimensions")
    if (np.diff(ends) < 0).any() or ends[-1] != len(arrays["spikes"]):
        raise ValueError("Invalid ragged spike offsets")
    x = np.zeros((len(trials), len(ends)), dtype=np.int64)
    for col, (a, b) in enumerate(zip(starts, ends, strict=True)):
        spikes = np.asarray(arrays["spikes"][a:b])
        if not np.isfinite(spikes).all() or (np.diff(spikes) < 0).any():
            raise ValueError("Nonfinite or unsorted unit spikes")
        # Half-open windows prevent double-counting shared trial boundaries.
        x[:, col] = np.searchsorted(spikes, trials.stop_time, side="left") - np.searchsorted(spikes, trials.start_time, side="left")
    return x


def make_folds(epoch: np.ndarray, scheme: str, n_folds: int = 5) -> list[tuple]:
    epoch = np.asarray(epoch)
    folds = []
    eps = np.unique(epoch)
    if scheme == "leave_one_epoch_out":
        for ep in eps:
            test = np.flatnonzero(epoch == ep)
            train = np.flatnonzero(epoch != ep)
            folds.append((f"epoch_{ep}", train, test, np.empty(0, dtype=int)))
    elif scheme == "blocked_trial":
        chunks = {ep: np.array_split(np.flatnonzero(epoch == ep), n_folds) for ep in eps}
        for fold in range(n_folds):
            test = np.concatenate([chunks[ep][fold] for ep in eps])
            purge = []
            for ep in eps:
                indices = np.flatnonzero(epoch == ep)
                held = chunks[ep][fold]
                if len(held):
                    loc = np.flatnonzero(np.isin(indices, held))
                    for p in (loc[0] - 1, loc[-1] + 1):
                        if 0 <= p < len(indices):
                            purge.append(indices[p])
            purge = np.setdiff1d(purge, test).astype(int)
            train = np.setdiff1d(np.arange(len(epoch)), np.r_[test, purge])
            if len(test):
                folds.append((f"block_{fold}", train, test, purge))
    else:
        raise ValueError(f"Unknown CV scheme: {scheme}")
    seen = np.zeros(len(epoch), dtype=int)
    for _, train, test, purge in folds:
        if not len(train) or np.intersect1d(train, np.r_[test, purge]).size:
            raise ValueError("Invalid or leaking fold")
        seen[test] += 1
    if not np.all(seen == 1):
        raise ValueError("Each trial must be evaluated exactly once per scheme")
    return folds


def softmax(scores: np.ndarray) -> np.ndarray:
    v = np.exp(scores - scores.max(axis=1, keepdims=True))
    v = np.maximum(v, 1e-300)
    return v / v.sum(axis=1, keepdims=True)


def raw_scores(x: np.ndarray, duration: np.ndarray, y: np.ndarray,
                train: np.ndarray, test: np.ndarray, k: int, model: str) -> np.ndarray:
    """Uniform class prior; estimates use training trials only, including null fits."""
    if model not in MODELS or not len(train):
        raise ValueError("Unknown model or empty training set")
    if not np.isfinite(x).all() or (x < 0).any() or not np.isfinite(duration).all() or (duration <= 0).any():
        raise ValueError("Counts/exposures invalid")
    scores = np.zeros((len(test), k))
    z = np.c_[np.log1p(x.sum(axis=1)), np.log(duration)]
    pooled_mean = z[train].mean(axis=0)
    pooled_var = np.maximum(z[train].var(axis=0), 1e-4)
    for c in range(k):
        selected = train[y[train] == c]
        counts = x[selected].sum(axis=0) + ALPHA
        if model == "conditional_identity":
            composition = counts / counts.sum()
            scores[:, c] = x[test] @ np.log(composition)
        elif model == "poisson_population":
            rate = counts / (duration[selected].sum() + 0.5)
            # Terms independent of class, including log(duration)*N and N!, cancel.
            scores[:, c] = x[test] @ np.log(rate) - duration[test] * rate.sum()
        else:
            mean = z[selected].mean(axis=0) if len(selected) else pooled_mean
            var = z[selected].var(axis=0) if len(selected) > 1 else pooled_var
            var = np.maximum(0.9 * var + 0.1 * pooled_var, 1e-4)
            scores[:, c] = -0.5 * (np.log(var) + (z[test] - mean)**2 / var).sum(axis=1)
    return scores


def fit_predict(x, duration, y, train, test, k, model):
    """Calibrate one temperature using two contiguous, training-only trial blocks.

    This calibrates predictive confidence, not the scientific null. The same
    nested procedure is repeated for every null fit. Outer test data are unused.
    """
    blocks = np.array_split(np.sort(train), 2)
    calibration_scores, calibration_y = [], []
    for held in blocks:
        inner_train = np.setdiff1d(train, held)
        if len(held) and len(inner_train):
            calibration_scores.append(raw_scores(x, duration, y, inner_train, held, k, model))
            calibration_y.append(y[held])
    temperature = 1.0
    if calibration_scores:
        score, truth = np.vstack(calibration_scores), np.concatenate(calibration_y)
        losses = []
        for temp in TEMPERATURES:
            scaled = score / temp
            maximum = scaled.max(axis=1)
            loss = maximum + np.log(np.exp(scaled - maximum[:, None]).sum(axis=1)) - scaled[np.arange(len(truth)), truth]
            losses.append(np.mean([loss[truth == c].mean() for c in np.unique(truth)]))
        temperature = float(TEMPERATURES[int(np.argmin(losses))])
    return softmax(raw_scores(x, duration, y, train, test, k, model) / temperature)


def metrics(y: np.ndarray, prob: np.ndarray) -> dict:
    pred = prob.argmax(axis=1)
    loss = -np.log(np.maximum(prob[np.arange(len(y)), y], 1e-300))
    present = np.unique(y)
    return {
        "n_trials": int(len(y)),
        "n_present_routes": int(len(present)),
        "balanced_accuracy": float(np.mean([(pred[y == c] == c).mean() for c in present])),
        "accuracy": float((pred == y).mean()),
        "macro_log_loss_nats": float(np.mean([loss[y == c].mean() for c in present])),
        "log_loss_nats": float(loss.mean()),
        "uniform_log_loss_nats": float(np.log(prob.shape[1])),
    }


def cross_predict(x, duration, y, k, folds, model):
    out = np.full((len(y), k), np.nan)
    for _, train, test, _ in folds:
        out[test] = fit_predict(x, duration, y, train, test, k, model)
    if not np.isfinite(out).all() or not np.allclose(out.sum(axis=1), 1):
        raise ValueError("Incomplete/nonfinite out-of-fold predictions")
    return out


def null_labels(y, epoch, rng, kind):
    yp = np.array(y, copy=True)
    for ep in np.unique(epoch):
        idx = np.flatnonzero(epoch == ep)
        if kind == "within_epoch_permutation":
            yp[idx] = rng.permutation(y[idx])
        elif kind == "within_epoch_circular_shift":
            if len(idx) < 2:
                continue
            guard = max(1, int(np.ceil(0.2 * len(idx))))
            guard = min(guard, len(idx) // 2)
            shift = int(rng.integers(guard, len(idx) - guard + 1))
            yp[idx] = np.roll(y[idx], shift)
        else:
            raise ValueError("Unknown null")
    return yp


def evaluate(trials, counts, groups, out: Path, n_null: int, seed: int):
    routes, y = np.unique(trials.route.to_numpy(), return_inverse=True)
    k = len(routes)
    epoch = trials.epoch_index.to_numpy()
    duration = trials.duration_s.to_numpy()
    summaries, fold_rows, pred_rows, null_rows, confusion_rows, plans = [], [], [], [], [], []
    for scheme in ("blocked_trial", "leave_one_epoch_out"):
        folds = make_folds(epoch, scheme)
        complete_support = all(np.bincount(y[tr], minlength=k).min() >= 2 for _, tr, _, _ in folds)
        for label, tr, te, purge in folds:
            plans.append({"scheme": scheme, "fold": label, "train_trial_ids": trials.id.iloc[tr].astype(int).tolist(),
                          "test_trial_ids": trials.id.iloc[te].astype(int).tolist(), "purged_trial_ids": trials.id.iloc[purge].astype(int).tolist()})
        for group, cols in groups.items():
            x = counts[:, cols]
            group_summaries = []
            for model in MODELS:
                p = cross_predict(x, duration, y, k, folds, model)
                base = {"group": group, "scheme": scheme, "model": model, "n_units": int(len(cols)),
                        "all_folds_have_two_training_trials_per_route": bool(complete_support)}
                row = {**base, **metrics(y, p)}
                summaries.append(row)
                group_summaries.append(row)
                pred = p.argmax(axis=1)
                for c in range(k):
                    for d in range(k):
                        confusion_rows.append({**base, "true_route": str(routes[c]), "predicted_route": str(routes[d]),
                                               "n_trials": int(((y == c) & (pred == d)).sum())})
                for fold, train, test, _ in folds:
                    fold_rows.append({**base, "fold": fold, "n_train": len(train),
                                      "train_class_counts": json.dumps(np.bincount(y[train], minlength=k).tolist()),
                                      **metrics(y[test], p[test])})
                    for i in test:
                        pred_rows.append({**base, "fold": fold, "trial_id": int(trials.id.iloc[i]),
                                          "epoch_index": int(epoch[i]), "route": str(routes[y[i]]),
                                          "predicted_route": str(routes[pred[i]]), "duration_s": float(duration[i]),
                                          "population_count": int(x[i].sum()),
                                          **{f"p_{route}": float(p[i, c]) for c, route in enumerate(routes)}})
                if model != "conditional_identity":
                    continue
                for null_idx, kind in enumerate(NULLS):
                    rng = np.random.default_rng(seed + 1009 * null_idx)
                    ba, ll = [], []
                    for j in range(n_null):
                        yp = null_labels(y, epoch, rng, kind)
                        pp = cross_predict(x, duration, yp, k, folds, model)
                        m = metrics(yp, pp)
                        ba.append(m["balanced_accuracy"])
                        ll.append(m["macro_log_loss_nats"])
                        null_rows.append({**base, "null_kind": kind, "null_index": j, **m})
                    row[f"{kind}_ba_p"] = float((1 + np.count_nonzero(np.asarray(ba) >= row["balanced_accuracy"])) / (n_null + 1))
                    row[f"{kind}_loss_p"] = float((1 + np.count_nonzero(np.asarray(ll) <= row["macro_log_loss_nats"])) / (n_null + 1))
                    row[f"{kind}_ba_mean"] = float(np.mean(ba))
                    row[f"{kind}_ba_q95"] = float(np.quantile(ba, 0.95))
                print(json.dumps(row, allow_nan=False), flush=True)
            identity, _, nuisance = group_summaries
            identity["macro_loss_gain_over_count_duration_nats"] = nuisance["macro_log_loss_nats"] - identity["macro_log_loss_nats"]
            identity["nominal_pilot_screen_pass"] = bool(
                complete_support and n_null >= 99
                and identity["macro_log_loss_nats"] < np.log(k)
                and identity["macro_loss_gain_over_count_duration_nats"] > 0
                and all(identity[f"{kind}_{metric}_p"] <= 0.05 for kind in NULLS for metric in ("ba", "loss")))
            # This is a conservative feasibility screen, not a corrected discovery test.
    tables = {"run_cv_summary": summaries, "run_fold_metrics": fold_rows,
              "run_predictions": pred_rows, "null_scores": null_rows, "route_confusion": confusion_rows}
    for name, rows in tables.items():
        pd.DataFrame(rows).to_csv(out / f"{name}.csv", index=False)
    write_json(out / "fold_plan.json", plans)
    return pd.DataFrame(summaries)


def analyze_asset(root, asset, out, n_null, seed):
    # Imports are delayed so numerical/leakage tests need no raw data or NWB reader.
    import h5py
    from acquire_dandi000978 import destination
    from preflight_dandi000978 import inspect
    from audit_dandi000978_cohort import metadata_fingerprint
    path = destination(root, asset)
    stat_before = path.stat()
    with h5py.File(path, "r") as f:
        report, epochs, raw_trials, units, arrays = inspect(f)
        fingerprint = metadata_fingerprint(f)
    groups, membership = agreement_groups(units, arrays["unit_ids"])
    trials, inclusion = prepare_trials(raw_trials)
    counts = trial_counts(arrays, trials)
    out.mkdir(parents=True, exist_ok=False)
    membership.to_csv(out / "unit_membership.csv", index=False)
    inclusion.to_csv(out / "trial_inclusion.csv", index=False)
    trials.to_csv(out / "run_trials.csv", index=False)
    epochs.to_csv(out / "epoch_inventory.csv", index=False)
    # Contains RUN counts only, not sleep spikes or raw NWB data.
    np.savez_compressed(out / "run_count_bank.npz", counts=counts, unit_ids=arrays["unit_ids"],
                        trial_ids=trials.id.to_numpy(), group_A=groups["A"], group_B=groups["B"])
    summary = evaluate(trials, counts, groups, out, n_null, seed)
    stat_after = path.stat()
    if (stat_before.st_size, stat_before.st_mtime_ns) != (stat_after.st_size, stat_after.st_mtime_ns):
        raise RuntimeError("Input file changed during analysis")
    info = {"subject": report["animal"], "file": path.name, "asset_id": asset.get("asset_id"),
            "expected_acquisition_sha256": asset["digest"]["dandi:sha2-256"],
            "full_raw_bytes_rehashed_this_run": False, "metadata_sha256": fingerprint,
            "local_size_bytes": stat_after.st_size, "local_mtime_ns": stat_after.st_mtime_ns,
            "group_sizes": {g: len(v) for g, v in groups.items()}, "n_trials": len(trials),
            "n_excluded_trials": len(inclusion) - len(trials), "n_epochs": int(trials.epoch_index.nunique()),
            "route_counts": {str(k): int(v) for k, v in trials.route.value_counts().items()},
            "anatomy_confirmed": False, "sorting_stability_confirmed": False, "replay_scored": False,
            "nominal_screen_is_not_multiplicity_corrected": True,
            "artifact_sha256": {p.name: digest(p) for p in sorted(out.iterdir()) if p.is_file()}}
    write_json(out / "asset_provenance.json", info)
    lines = [f"# {report['animal']}: provisional-group RUN pilot", "", "No confirmed anatomical labels; no sleep/replay scoring.", "",
             "Whole trials are the samples. Route is the ordered start-well/end-well pair; it is not a memory label.",
             "Group A/B membership uses metadata agreement only; a common export error could invalidate both anatomical interpretations.",
             "Full-trial decoding may use occupancy, movement and dwell differences. Success does not validate compressed replay or content independent of behavior.", "",
             "| Group | CV | Model | Balanced accuracy | Macro log loss (nats) |", "|---|---|---|---:|---:|"]
    for r in summary.itertuples():
        lines.append(f"| {r.group} | {r.scheme} | {r.model} | {r.balanced_accuracy:.4f} | {r.macro_log_loss_nats:.4f} |")
    lines += ["", "## Interpretation", "",
              "See run_cv_summary.csv for the 199-per-null default controls, training-class support and the nominal screen. A negative or unsupported screen is a scientific outcome, not a failed software run.",
              "Epoch folds and label shuffles are dependent within an animal, not biological replications. Two rats are a pilot, not a population-level claim.",
              "Independent-population agreement is not ground truth. A surviving fixed validation score after thinning is tautological; the downstream test must compare newly rejected sets against controls and test distributional selection bias separately."]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(info, allow_nan=False), flush=True)
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--subjects", nargs="+", default=["ER1", "JS34"])
    parser.add_argument("--n-null", type=int, default=199)
    parser.add_argument("--seed", type=int, default=20260921)
    args = parser.parse_args()
    if args.n_null < 1 or args.n_null > 999:
        parser.error("n-null must be between 1 and 999")
    allowed = {"ER1", "JS34", "JS14", "JS15", "JS17", "JS21", "KL8"}
    if len(set(args.subjects)) != len(args.subjects) or not set(args.subjects) <= allowed:
        parser.error("Unsupported/duplicate subject; ZT2 cross-file joining is forbidden")
    root, out = args.dataset_root.resolve(), args.output_dir.resolve()
    if out == root or root in out.parents or out in root.parents:
        parser.error("Output must be separate from the raw dataset tree")
    from audit_dandi000978_cohort import verify_acquisition_record
    manifest_path, status_path = root / "metadata/asset_manifest.json", root / "download_status.json"
    manifest, status = json.loads(manifest_path.read_text()), json.loads(status_path.read_text())
    verify_acquisition_record(root, manifest, status)
    out.mkdir(parents=True, exist_ok=False)
    provenance = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "host": socket.gethostname(),
                  "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                  "argv": sys.argv, "seed": args.seed, "n_null": args.n_null,
                  "manifest_sha256": digest(manifest_path), "acquisition_record_sha256": digest(status_path),
                  "checksum_basis": "prior_completed_acquisition_hashes_plus_current_sizes_not_raw_rehash",
                  "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  "github_run_id": os.environ.get("GITHUB_RUN_ID"), "github_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
                  "source_sha256": {p.name: digest(p) for p in [Path(__file__), Path(__file__).with_name("preflight_dandi000978.py"),
                                    Path(__file__).with_name("audit_dandi000978_cohort.py"), Path(__file__).with_name("acquire_dandi000978.py")]},
                  "temperature_grid": TEMPERATURES.tolist(), "temperature_fit": "two_contiguous_training_only_trial_blocks_macro_log_loss",
                  "protocol": "docs/dandi000978_agreement_run_protocol.md", "anatomy_confirmed": False, "replay_scored": False,
                  "subjects": [], "errors": []}
    for subject in args.subjects:
        assets = [a for a in manifest["assets"] if f"sub-JDS-SingleDay-{subject}/" in a["path"]
                  or Path(a["path"]).name == f"sub-JDS-SingleDay-{subject}_behavior+ecephys.nwb"]
        try:
            if len(assets) != 1:
                raise ValueError(f"Expected exactly one asset for {subject}, found {len(assets)}")
            provenance["subjects"].append(analyze_asset(root, assets[0], out / subject, args.n_null, args.seed))
        except Exception as exc:
            import traceback
            traceback.print_exc()
            provenance["errors"].append({"subject": subject, "error": str(exc), "type": type(exc).__name__})
        write_json(out / "campaign_provenance.json", provenance)
    print(json.dumps(provenance, indent=2, allow_nan=False), flush=True)
    if provenance["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
