"""Fixed-map synchronized CA1/PFC RUN calibration, without sleep rescoring."""

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
from calibrate_dandi000978_joint_readout import PARAMETERS as STUDY_PARAMETERS
from calibrate_dandi000978_joint_readout import group_simulation, score_matrix, validate_pair
from calibrate_dandi000978_sparse_content import thin_exact
from dandi000978_coverage_content import composition_scores, provenance
from preflight_dandi000978 import inspect
from validate_dandi000978_run_readouts import prepare_bins, seed_for

CONDITIONS = ("synchronous", "position_matched", "independent")
ANCHOR_SEED = 20260925


def nearest_other_trial(bins, indices):
    """Behavior-only matching; spike counts and decoded routes are not inputs."""
    xy = bins[["x_cm", "y_cm"]].to_numpy()
    if not np.isfinite(xy).all():
        raise ValueError("Nonfinite matching positions")
    routes, trials = bins.route.to_numpy(), bins.trial_id.to_numpy()
    cache = {}
    for ix in np.unique(indices):
        allowed = np.flatnonzero((routes == routes[ix]) & (trials != trials[ix]))
        if not len(allowed):
            raise ValueError("No same-route other-trial donor")
        distance = np.linalg.norm(xy[allowed] - xy[ix], axis=1)
        j = int(np.argmin(distance))
        cache[int(ix)] = (int(allowed[j]), float(distance[j]))
    return np.array([cache[int(i)][0] for i in indices]), np.array([cache[int(i)][1] for i in indices])


def thin_rows(counts, targets, key):
    rng = np.random.default_rng(seed_for(ANCHOR_SEED, *key))
    sparse = np.zeros_like(counts)
    matched = np.zeros(len(counts), dtype=bool)
    for i, (row, target) in enumerate(zip(counts, targets, strict=True)):
        value = thin_exact(row, int(target), rng)
        if value is not None:
            sparse[i] = value
            matched[i] = True
    return sparse, matched


def regional_counts(counts, all_ids, selected_ids):
    if len(set(all_ids)) != len(all_ids) or len(set(selected_ids)) != len(selected_ids):
        raise ValueError("Duplicate unit identity")
    lookup = {uid: j for j, uid in enumerate(all_ids)}
    return counts[:, [lookup[uid] for uid in selected_ids]]


def make_fold(bins, counts, all_ids, ca_frame, pf_frame, ca, pf, key):
    validate_pair(ca_frame, pf_frame, ca, pf)
    mask = ca_frame.scope.to_numpy() == "window_250ms"
    source_rows = np.flatnonzero(mask)
    frame = ca_frame.loc[mask].reset_index(drop=True).copy()
    pframe = pf_frame.loc[mask].reset_index(drop=True)
    if not set(bins.trial_id).issubset(ca["test_trial_ids"]):
        raise ValueError("Source windows are not held out")
    ca_counts = regional_counts(counts, all_ids, ca["source_unit_ids"])
    pf_counts = regional_counts(counts, all_ids, pf["source_unit_ids"])
    ca_ix, pf_ix = frame.window_index.to_numpy(int), pframe.window_index.to_numpy(int)
    if not np.array_equal(ca_counts[ca_ix], ca["native"][source_rows]):
        raise ValueError("CA1 archived donor does not match source spikes")
    if not np.array_equal(pf_counts[pf_ix], pf["native"][source_rows]):
        raise ValueError("PFC archived donor does not match source spikes")
    if not np.array_equal(bins.route.to_numpy()[ca_ix], frame.route):
        raise ValueError("Donor route mismatch")
    near, distance = nearest_other_trial(bins, ca_ix)
    ca_native, ca_sparse = ca_counts[ca_ix], ca["sparse"][source_rows]
    native = {"synchronous": pf_counts[ca_ix], "position_matched": pf_counts[near], "independent": pf_counts[pf_ix]}
    sparse = {"independent": pf["sparse"][source_rows]}
    matched = {"independent": pframe.matched.to_numpy(bool)}
    for condition in CONDITIONS[:2]:
        sparse[condition], matched[condition] = thin_rows(native[condition], pframe.target_count.to_numpy(), (*key, condition))
    frame["position_matched_index"] = near
    frame["position_match_distance_cm"] = distance
    frame["independent_index"] = pf_ix
    frame["independent_distance_cm"] = np.linalg.norm(bins[["x_cm", "y_cm"]].to_numpy()[pf_ix] - bins[["x_cm", "y_cm"]].to_numpy()[ca_ix], axis=1)
    frame["ca1_matched"] = frame.matched
    for condition in CONDITIONS:
        frame[condition + "_matched"] = matched[condition]
    grouped = list(frame.groupby(["target_index", "repeat"], sort=True))
    groups = []
    blocks = []
    for (target, repeat), g in grouped:
        if len(g) != 4 or set(g.route) != set(range(4)):
            raise ValueError("Incomplete route block")
        ix = g.sort_values("route").index.to_numpy()
        groups.append(ix)
        row = {"target_index": target, "repeat": repeat, "event_id": ca["target_event_ids"][target]}
        for condition in CONDITIONS:
            row[condition + "_matched"] = bool((frame.ca1_matched.to_numpy()[ix] & matched[condition][ix]).all())
        row["common_matched"] = all(row[c + "_matched"] for c in CONDITIONS)
        blocks.append(row)
    blocks = pd.DataFrame(blocks)
    ix = np.array(groups).reshape(-1)
    bank = {}
    audit = {"ca1_native": ca_native, "ca1_sparse": ca_sparse, "ca1_rates": ca["rates"], "pfc_rates": pf["rates"], "block_indices": np.array(groups)}
    for regime, ca_c, pf_c in (("native", ca_native, native), ("sleep_matched", ca_sparse, sparse)):
        _, prob = composition_scores(ca_c[ix], ca["rates"])
        references = {"decoded": prob.argmax(axis=1).reshape(-1, 4), "oracle": np.broadcast_to(np.arange(4), (len(blocks), 4))}
        audit[regime + "_ca1_zero"] = (ca_c[ix].sum(axis=1) == 0).reshape(-1, 4)
        for condition in CONDITIONS:
            vectors, _ = composition_scores(pf_c[condition][ix], pf["rates"])
            audit[regime + "_" + condition + "_pfc_counts"] = pf_c[condition]
            for ref, values in references.items():
                bank[regime + "_" + condition + "_" + ref] = score_matrix(vectors.reshape(-1, 4, 4), values)
    return frame, blocks, bank, audit


def summarize(blocks, bank, primary, out):
    coverage, signal, power = [], [], []
    for animal, animal_blocks in blocks.groupby("animal"):
        desired = primary.loc[primary.animal == animal, "event_id"].to_numpy()
        for condition in (*CONDITIONS, "common"):
            supported = animal_blocks[animal_blocks[condition + "_matched"]]
            coverage.append(
                {
                    "animal": animal,
                    "condition": condition,
                    "blocks": len(animal_blocks),
                    "matched_blocks": len(supported),
                    "targets": animal_blocks.event_id.nunique(),
                    "supported_targets": supported.event_id.nunique(),
                    "primary_targets": len(desired),
                    "missing_primary_targets": len(set(desired) - set(supported.event_id)),
                }
            )
        for regime in ("native", "sleep_matched"):
            for support in ("all", "common") if regime == "native" else ("common",):
                group = animal_blocks if support == "all" else animal_blocks[animal_blocks.common_matched]
                indices = group.index.to_numpy()
                for condition in CONDITIONS:
                    for ref in ("decoded", "oracle"):
                        values = bank[regime + "_" + condition + "_" + ref][indices]
                        diag = np.diagonal(values, axis1=1, axis2=2).mean(axis=1)
                        stats = group[["file", "heldout_epoch", "event_id"]].copy()
                        stats["score"] = diag
                        stats["excess"] = diag - values.mean(axis=(1, 2))
                        per_target = stats.groupby(["file", "heldout_epoch", "event_id"])[["score", "excess"]].mean()
                        per_epoch = per_target.groupby(["file", "heldout_epoch"]).mean()
                        base = {"animal": animal, "regime": regime, "support": support, "condition": condition, "reference": ref}
                        signal.append(
                            {**base, "n_epochs": len(per_epoch), "n_targets": len(per_target), "mean_score": per_epoch.score.mean(), "mean_excess": per_epoch.excess.mean()}
                        )
                        if regime != "sleep_matched":
                            continue
                        for design, requested in [("original_primary", desired), *[(f"sensitivity_n{n}", n) for n in STUDY_PARAMETERS["sizes"]]]:
                            # Same draws for every donor condition and reference.
                            result, sim = group_simulation(values, group.event_id.to_numpy(), requested, (animal, "synchronized_common", design))
                            power.append({**base, "design": design, **result})
                            if sim is not None:
                                np.savez_compressed(out / "simulations" / f"{animal}_{condition}_{ref}_{design}.npz", **sim)
    pd.DataFrame(coverage).to_csv(out / "coverage.csv", index=False)
    pd.DataFrame(signal).to_csv(out / "signal_summary.csv", index=False)
    pd.DataFrame(power).to_csv(out / "conditional_sensitivity.csv", index=False)


def run(args):
    root, prior = args.dataset_root, args.calibration_dir
    inputs = {
        "prior_manifest": prior / "manifest.json",
        "prior_verification": prior / "independent_verification.json",
        "joint_verification": args.joint_dir / "independent_verification.json",
        "joint_manifest": args.joint_dir / "manifest.json",
        "assets": root / "metadata/asset_manifest.json",
        "acquisition": root / "download_status.json",
        "sleep_events": args.sleep_events,
    }
    old_manifest = json.loads(inputs["prior_manifest"].read_text())
    for label in ("prior_verification", "joint_verification"):
        if not json.loads(inputs[label].read_text())["verified"]:
            raise ValueError("Unverified predecessor")
    for key in ("assets", "acquisition"):
        if file_sha256(inputs[key]) != old_manifest["input_file_sha256"][key]:
            raise ValueError("Acquisition provenance changed")
    source = json.loads(inputs["assets"].read_text())
    verify_acquisition_record(root, source, json.loads(inputs["acquisition"].read_text()))
    sources = sorted(prior.glob("*/CA1/epoch_*/audit.npz"))
    for j, path in enumerate(sources):
        for region in ("CA1", "PFC"):
            parent = path.parent.parent.parent / region / path.parent.name
            for name in ("audit.npz", "anchors.csv"):
                inputs[f"fold{j}_{region}_{name}"] = parent / name
    inputs["fold_metrics"] = prior / "fold_metrics.csv"
    joint_manifest = json.loads(inputs["joint_manifest"].read_text())
    for name, path in inputs.items():
        previous_name = {"prior_manifest": "calibration_manifest", "prior_verification": "calibration_verification"}.get(name, name)
        if previous_name in joint_manifest["input_file_sha256"] and file_sha256(path) != joint_manifest["input_file_sha256"][previous_name]:
            raise ValueError("Verified joint predecessor inputs changed: " + name)
    prov = provenance(inputs)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False, mode=0o700)
    (out / "simulations").mkdir()
    write_json(
        out / "manifest.json",
        {**prov, "anchor_seed": ANCHOR_SEED, "study_parameters": STUDY_PARAMETERS, "scope": "synchronized_RUN_calibration_no_sleep_rescore", "conditions": CONDITIONS},
    )
    folds = pd.read_csv(inputs["fold_metrics"])
    primary = pd.read_csv(args.sleep_events)
    primary = primary[primary.primary_lost_half]
    block_parts, bank_parts, statuses, distance_rows = [], {}, [], []
    for asset in source["assets"]:
        paths = [p for p in sources if p.parent.parent.parent.name == asset["asset_id"]]
        if not paths:
            continue
        filename = Path(asset["path"]).name
        animal = folds.loc[folds.file == filename, "animal"].iloc[0]
        with h5py.File(destination(root, asset), "r") as f:
            _, _, trials, _, arrays = inspect(f)
        bins, counts, _ = prepare_bins(arrays, trials)
        for path in paths:
            epoch = int(path.parent.name.split("_")[1])
            selected = (bins.epoch == epoch) & (bins.route >= 0)
            tb = bins.loc[selected].reset_index(drop=True)
            pf_path = path.parent.parent.parent / "PFC" / path.parent.name
            ca, pf = dict(np.load(path)), dict(np.load(pf_path / "audit.npz"))
            cf, pff = pd.read_csv(path.parent / "anchors.csv"), pd.read_csv(pf_path / "anchors.csv")
            frame, block, bank, audit = make_fold(tb, counts[selected], arrays["unit_ids"], cf, pff, ca, pf, (filename, epoch))
            base = {"animal": animal, "file": filename, "heldout_epoch": epoch}
            dest = out / asset["asset_id"] / path.parent.name
            dest.mkdir(parents=True)
            frame.to_csv(dest / "anchors.csv", index=False)
            tb.to_csv(dest / "source_bins.csv", index=False)
            np.savez_compressed(dest / "audit.npz", **audit, **bank, ca1_source_unit_ids=ca["source_unit_ids"], pfc_source_unit_ids=pf["source_unit_ids"])
            for k, v in base.items():
                block[k] = v
            block_parts.append(block)
            for name, value in bank.items():
                bank_parts.setdefault(name, []).append(value)
            for column in ("position_match_distance_cm", "independent_distance_cm"):
                distance_rows.append({**base, "metric": column, "median": float(frame[column].median()), "p95": float(frame[column].quantile(0.95))})
            statuses.append(
                {
                    **base,
                    "status": "complete",
                    "blocks": len(block),
                    "common_blocks": int(block.common_matched.sum()),
                    "native_ca1_zero_fraction": float(audit["native_ca1_zero"].mean()),
                }
            )
            pd.DataFrame(statuses).to_csv(out / "fold_status.csv", index=False)
            print(json.dumps(statuses[-1]), flush=True)
    if len(statuses) != len(sources) or len(sources) != 14:
        raise ValueError("Expected exactly 14 prior joint folds")
    blocks = pd.concat(block_parts, ignore_index=True)
    bank = {k: np.concatenate(v) for k, v in bank_parts.items()}
    blocks.to_csv(out / "blocks.csv", index=False)
    np.savez_compressed(out / "joint_bank.npz", **bank)
    pd.DataFrame(distance_rows).to_csv(out / "donor_distance_summary.csv", index=False)
    summarize(blocks, bank, primary, out)
    write_json(out / "decision.json", {"technical_complete": True, "sleep_content_validated": False, "paper_ready": False, "decision": "synchronized_RUN_diagnostic_only"})
    write_json(out / "terminal_status.json", {"status": "complete"})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--calibration-dir", type=Path, required=True)
    p.add_argument("--joint-dir", type=Path, required=True)
    p.add_argument("--sleep-events", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    try:
        run(args)
    except Exception as exc:
        if args.output_dir.exists():
            write_json(args.output_dir / "terminal_status.json", {"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        raise


if __name__ == "__main__":
    main()
