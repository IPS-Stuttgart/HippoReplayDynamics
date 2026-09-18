"""Real-spike, behavior-anchored RUN control with fold-internal maps and cell QC."""

import argparse
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from hipporeplayimm.tirole_two_track import blocked_training_mask, file_sha256, fit_maps, load_session
from hipporeplayimm.two_track_content import classify_sequence, event_bin_counts, nested_subsets, split_populations, stable_seed
from scripts.audit_tirole_evaluation_identity import context_odds, matched_permutations, null_summary


def select_windows(session, cap=20):
    n = int(np.floor(session.times[-1] - session.times[0]))
    ids = np.arange(n)
    start = session.times[0] + ids
    end = start + 1.0
    left = np.searchsorted(session.times, start)
    right = np.searchsorted(session.times, end)
    rows = []
    for track, x in enumerate(session.positions, 1):
        valid = session.run_mask() & np.isfinite(x)
        bad = np.r_[0, np.cumsum(~valid)]
        good = (right > left) & (bad[right] == bad[left])
        for index in ids[good]:
            rows.append(
                {
                    "window_id": int(index),
                    "start_s": float(start[index]),
                    "end_s": float(end[index]),
                    "duration_s": 1.0,
                    "truth_track": track,
                    "fold": int(index // 10 % 5),
                    "time_block": int(index // 10),
                }
            )
    all_windows = pd.DataFrame(rows)
    if all_windows.empty:
        raise ValueError("no behavior-only RUN windows")
    selected = []
    availability = []
    for fold in range(5):
        groups = {t: all_windows[all_windows.fold.eq(fold) & all_windows.truth_track.eq(t)] for t in [1, 2]}
        take = min(cap, *(len(g) for g in groups.values()))
        for t, g in groups.items():
            ordered = sorted(g.window_id, key=lambda eid: stable_seed(20260918, session.name, fold, t, eid, "RUN-known-context-window"))
            selected.extend(ordered[:take])
            availability.append({"fold": fold, "truth_track": t, "available_windows": len(g), "selected_windows": take, "target_windows": cap})
    chosen = all_windows[all_windows.window_id.isin(selected)].sort_values("window_id").reset_index(drop=True)
    if chosen.empty or chosen.window_id.duplicated().any():
        raise ValueError("no unique balanced RUN windows")
    return chosen, pd.DataFrame(availability)


def exposure_rates(rates):
    return 5.0 * np.maximum(np.asarray(rates, float), 1e-4)


def behavior_bins(session, window):
    x = session.positions[int(window["truth_track"]) - 1]
    means = []
    edges = float(window["start_s"]) + np.arange(11) * 0.1
    for start, end in pairwise(edges):
        a, b = np.searchsorted(session.times, [start, end])
        values = x[a:b]
        if not len(values) or not np.isfinite(values).all():
            raise ValueError("missing known RUN position")
        means.append(float(values.mean()))
    return np.asarray(means)


def run(dataset, preflight, session_name, output):
    if output.exists():
        raise ValueError("new immutable RUN control required")
    git = lambda *a: subprocess.check_output(["git", *a], cwd=ROOT, text=True).strip()
    commit = git("rev-parse", "HEAD")
    if git("status", "--porcelain"):
        raise ValueError("freeze protocol before reading outcomes")
    pm = json.loads((preflight / "two_track_preflight_manifest.json").read_text())
    if session_name not in ["RAT3_SESS2", "RAT5_SESS2"] or session_name not in pm["eligible_sessions"]:
        raise ValueError("frozen primary sessions only")
    sources = []
    for src in pm["input_files"]:
        if src["session"] != session_name:
            continue
        p = dataset / Path(src["path"]).name
        if file_sha256(p) != src["sha256"] or p.stat().st_size != src["size"]:
            raise ValueError("changed source recording")
        sources.append({**src, "path": str(p.resolve())})
    if len(sources) != 2:
        raise ValueError("both spike and position inputs required")
    session = load_session(dataset, session_name)
    windows, availability = select_windows(session)
    output.mkdir(parents=True)
    (output / "fold_maps").mkdir()
    availability.to_csv(output / "window_availability.csv", index=False)
    folds = {}
    partitions = []
    qc = []
    for fold in range(5):
        train = blocked_training_mask(session.times, session.times[0], fold)
        maps = fit_maps(session, train)
        if len(maps["common_units"]) < 20 or not maps["valid_bins"].any(axis=1).all():
            raise ValueError("insufficient fold-internal maps; do not drop fold silently")
        reserve, splits = split_populations(maps["common_units"], f"{session_name}:RUNfold{fold}")
        np.savez_compressed(output / "fold_maps" / f"{fold}.npz", **maps, training_samples=train, unit_ids=session.unit_ids)
        folds[fold] = (maps, splits)
        partitions.append({"fold": fold, "reserved_detector_units": reserve.tolist(), "splits": [{"inference": a.tolist(), "evaluation": b.tolist()} for a, b in splits]})
        qc.append(
            {
                "fold": fold,
                "training_samples": int(train.sum()),
                "excluded_training_samples": int((~train).sum()),
                "common_units": len(maps["common_units"]),
                "track1_valid_bins": int(maps["valid_bins"][0].sum()),
                "track2_valid_bins": int(maps["valid_bins"][1].sum()),
            }
        )
        for row in windows[windows.fold.eq(fold)].to_dict("records"):
            a, b = np.searchsorted(session.times, [row["start_s"], row["end_s"]])
            if train[a:b].any():
                raise ValueError("test window enters map training")
    (output / "partitions.json").write_text(json.dumps(partitions, indent=2) + "\n")
    pd.DataFrame(qc).to_csv(output / "fold_training_qc.csv", index=False)
    counts = []
    truth = []
    for window in windows.to_dict("records"):
        c = event_bin_counts(session, window["start_s"], window["end_s"], 0.1)
        if len(c) != 10:
            raise ValueError("RUN exposure must contain ten full bins")
        counts.append(c)
        truth.append(behavior_bins(session, window))
    counts = np.asarray(counts)
    truth = np.asarray(truth)
    windows["behavior_start_cm"] = truth[:, 0]
    windows["behavior_end_cm"] = truth[:, -1]
    windows["behavior_span_cm"] = truth.max(axis=1) - truth.min(axis=1)
    windows["behavior_net_displacement_cm"] = np.abs(truth[:, -1] - truth[:, 0])
    windows.to_csv(output / "RUN_windows.csv", index=False)
    np.savez_compressed(output / "RUN_counts.npz", counts=counts, window_ids=windows.window_id.to_numpy(), truth_bin_position_cm=truth, unit_ids=session.unit_ids)
    rows = []
    readouts = []
    started = time.monotonic()
    for fold, (maps, splits) in folds.items():
        fitted = exposure_rates(maps["rates"])
        occupancy = maps["occupancy_s"]
        run_rate = (maps["rates"] * (occupancy / occupancy.sum(axis=1, keepdims=True))[:, None]).sum(axis=2).mean(axis=0)
        for split, (inference, evaluation) in enumerate(splits):
            permutations, _ = matched_permutations(run_rate[evaluation], seed=stable_seed(20260918, session_name, fold, split, "RUN-evaluation-identity"))
            halves = [nested_subsets(inference, f"{session_name}:RUNfold{fold}", split, repeat)[0.5] for repeat in range(5)]
            for index, window in windows[windows.fold.eq(fold)].iterrows():
                eid = int(window.window_id)
                c = counts[index]
                sign = 1 if window.truth_track == 1 else -1
                values = context_odds(c[:, evaluation], fitted[:, evaluation], maps["valid_bins"], permutations, True)
                b = null_summary(values)
                meta = {**window.to_dict(), "session": session_name, "animal": session_name.split("_")[0], "split": split}
                readout = {
                    **meta,
                    "n_evaluation_spikes": int(c[:, evaluation].sum()),
                    "n_evaluation_active_units": int((c[:, evaluation].sum(axis=0) > 0).sum()),
                    "evaluation_true_probability": float(expit(sign * b["observed_log_odds"])),
                    "evaluation_true_z": float(sign * b["identity_z_log_odds"]),
                    "evaluation_correct": float(sign * b["observed_log_odds"] > 0) if np.isfinite(b["observed_log_odds"]) else np.nan,
                }
                readouts.append(readout)
                shuffled = np.random.default_rng(stable_seed(20260918, session_name, eid, fold, split, "RUN-order-negative-control")).permutation(len(c))
                for order, observed in [("original", c), ("whole_bin_shuffled", c[shuffled])]:
                    rng = np.random.default_rng(stable_seed(20260918, session_name, eid, fold, split, order, "RUN-selection-nulls"))
                    shifts = rng.integers(0, len(maps["bin_centers_cm"]), (499, 2, c.shape[1]))
                    time_permutations = np.argsort(rng.random((499, len(c))), axis=1)
                    for likelihood, conditional in [("poisson", False), ("conditional_count", True)]:
                        for repeat, ids in [(-1, inference), *enumerate(halves)]:
                            result = classify_sequence(
                                observed[:, ids], fitted[:, ids], maps["valid_bins"], maps["bin_centers_cm"], shifts[:, :, ids], time_permutations, conditional_count=conditional
                            )
                            rows.append(
                                {
                                    **meta,
                                    "order": order,
                                    "likelihood": likelihood,
                                    "repeat": repeat,
                                    "fraction": 1.0 if repeat == -1 else 0.5,
                                    "n_inference_cells": len(ids),
                                    "n_inference_spikes": int(observed[:, ids].sum()),
                                    **result,
                                }
                            )
        print(
            json.dumps(
                {"session": session_name, "fold_completed": fold, "windows": len(windows[windows.fold.eq(fold)]), "score_rows": len(rows), "elapsed_s": time.monotonic() - started}
            ),
            flush=True,
        )
    scores = pd.DataFrame(rows)
    content = pd.DataFrame(readouts)
    if len(scores) != len(windows) * 5 * 2 * 2 * 6 or len(content) != len(windows) * 5 or scores.duplicated(["window_id", "split", "order", "likelihood", "repeat"]).any():
        raise ValueError("incomplete real RUN calibration")
    if git("status", "--porcelain") or git("rev-parse", "HEAD") != commit:
        raise ValueError("code changed during calibration")
    scores.to_csv(output / "RUN_sequence_scores.csv", index=False)
    content.to_csv(output / "RUN_independent_content.csv", index=False)
    manifest = {
        "status": "complete",
        "session": session_name,
        "code_commit": commit,
        "git_dirty": False,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "command_line": sys.argv,
        "dataset_root": str(dataset.resolve()),
        "source_files": sources,
        "preflight_manifest_sha256": file_sha256(preflight / "two_track_preflight_manifest.json"),
        "n_windows": len(windows),
        "n_score_rows": len(scores),
        "n_readout_rows": len(content),
        "all_cells_and_maps_training_fold_only": True,
        "windows_behavior_only": True,
        "behavioral_context_labels_not_replay_truth": True,
        "actual_window_duration_s": 1.0,
        "actual_bin_s": 0.1,
        "classifier_bin_s": 0.02,
        "rate_exposure_multiplier": 5.0,
        "new_simulated_spikes": False,
        "sleep_replay_rescored": False,
        "biological_replay_bias_confirmed": False,
        "elapsed_s": time.monotonic() - started,
        "output_sha256": {str(p.relative_to(output)): file_sha256(p) for p in output.rglob("*") if p.is_file()},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--preflight-dir", type=Path, required=True)
    p.add_argument("--session", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    run(a.dataset_root, a.preflight_dir, a.session, a.output_dir)
