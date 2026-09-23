#!/usr/bin/env python3
"""Frozen task-only arm readout feasibility; no rest or surprise-effect scoring."""

from __future__ import annotations

import argparse
import hashlib
import json
import socket
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from _provenance import build_script_provenance, file_sha256
from preflight_roscow2025 import ANIMALS, read_single, session_number, timestamps
from validate_dandi000978_run_readouts import posterior

PARAMETERS = {
    "window_start_s": -2.0, "window_stop_s": -0.25,
    "minimum_ca1_units": 5, "minimum_trials_per_arm": 3,
    "prior_seconds": 1.0, "rate_floor_hz": 1e-10,
    "aggregate_null_draws": 10000, "seed": 20260923,
}
READOUTS = ("poisson", "composition", "count_only")


def stable_seed(*parts):
    return int.from_bytes(hashlib.sha256(json.dumps(parts).encode()).digest()[:8], "little")


def fit_rates(counts, labels, exposure):
    counts, labels = np.asarray(counts), np.asarray(labels)
    if counts.ndim != 2 or len(counts) != len(labels) or counts.shape[1] == 0:
        raise ValueError("counts/label dimensions invalid")
    if exposure <= 0 or not np.isfinite(counts).all() or (counts < 0).any():
        raise ValueError("invalid counts/exposure")
    if set(labels) != {0, 1, 2}:
        raise ValueError("all three arms required in training")
    global_rate = counts.sum(axis=0) / (len(counts) * exposure)
    rates = [(counts[labels == arm].sum(axis=0) + PARAMETERS["prior_seconds"] * global_rate)
             / ((labels == arm).sum() * exposure + PARAMETERS["prior_seconds"]) for arm in range(3)]
    return np.maximum(np.asarray(rates), PARAMETERS["rate_floor_hz"])


def leave_one_out(counts, labels, exposure):
    labels = np.asarray(labels, dtype=int)
    result = {kind: np.empty((len(labels), 3)) for kind in READOUTS}
    for i in range(len(labels)):
        train = np.arange(len(labels)) != i
        rates = fit_rates(counts[train], labels[train], exposure)
        for kind in READOUTS:
            result[kind][i] = posterior(counts[i:i + 1], rates, np.array([exposure]), kind)[0]
    return result


def scores(probabilities, labels):
    labels = np.asarray(labels, dtype=int)
    maxima = probabilities.max(axis=1, keepdims=True)
    ties = np.isclose(probabilities, maxima, rtol=1e-12, atol=1e-15)
    correct = ties[np.arange(len(labels)), labels] / ties.sum(axis=1)
    balanced = np.mean([correct[labels == arm].mean() for arm in range(3)])
    logp = np.log(np.maximum(probabilities[np.arange(len(labels)), labels], np.finfo(float).tiny))
    return {"balanced_accuracy": float(balanced), "mean_log_score_above_chance": float(logp.mean() + np.log(3))}


def score_session(counts, labels, exposure):
    metrics, original = [], None
    for shift in range(len(labels)):
        shifted = np.roll(labels, shift)
        predictions = leave_one_out(counts, shifted, exposure)
        if shift == 0:
            original = predictions
        for kind, prob in predictions.items():
            metrics.append({"shift": shift, "readout": kind, **scores(prob, shifted)})
    return pd.DataFrame(metrics), original


def count_windows(cells, arrivals, task_bounds):
    a = np.asarray(arrivals) + PARAMETERS["window_start_s"]
    b = np.asarray(arrivals) + PARAMETERS["window_stop_s"]
    if (a < task_bounds[0]).any() or (b > task_bounds[1]).any() or (a[1:] < b[:-1]).any():
        raise ValueError("overlapping or out-of-task counting windows")
    return np.column_stack([np.searchsorted(timestamps(x), b, side="left")
                            - np.searchsorted(timestamps(x), a, side="left") for x in cells])


def summarize_animals(nulls):
    rows = []
    for animal in ANIMALS.values():
        for kind in READOUTS:
            group = nulls[(nulls.animal == animal) & (nulls.readout == kind)]
            sessions = [x.sort_values("shift") for _, x in group.groupby("session")]
            if not sessions:
                rows.append({"animal": animal, "readout": kind, "n_sessions": 0, "screen_pass": False})
                continue
            rng = np.random.default_rng(stable_seed(PARAMETERS["seed"], animal, kind))
            sampled, observed, log_scores = [], [], []
            for session in sessions:
                observed.append(float(session.loc[session["shift"] == 0, "balanced_accuracy"].iloc[0]))
                log_scores.append(float(session.loc[session["shift"] == 0, "mean_log_score_above_chance"].iloc[0]))
                sampled.append(rng.choice(session.balanced_accuracy.to_numpy(), PARAMETERS["aggregate_null_draws"]))
            null = np.mean(sampled, axis=0)
            actual, p95 = float(np.mean(observed)), float(np.quantile(null, 0.95))
            rows.append({
                "animal": animal, "readout": kind, "n_sessions": len(sessions),
                "mean_session_balanced_accuracy": actual, "null_median": float(np.median(null)),
                "null_p95": p95, "empirical_p_value": float((1 + (null >= actual).sum()) / (1 + len(null))),
                "mean_session_log_score_above_chance": float(np.mean(log_scores)),
                "screen_pass": bool(len(sessions) >= 2 and actual > max(p95, 1 / 3)),
            })
    return pd.DataFrame(rows)


def write_figure(animals, target):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), sharey=True)
    for ax, kind in zip(axes, READOUTS, strict=True):
        sub = animals[animals.readout == kind]
        for j, row in enumerate(sub.itertuples()):
            if row.n_sessions:
                ax.plot([j, j], [row.null_median, row.null_p95], color="0.55", linewidth=3)
                ax.scatter(j, row.mean_session_balanced_accuracy, color="#176d96", zorder=3)
        ax.axhline(1 / 3, color="0.6", linestyle=":")
        ax.set_xticks(range(3), [f"{r.animal}\nn={r.n_sessions} sessions" for r in sub.itertuples()], fontsize=8)
        ax.set_title(kind.replace("_", " "))
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("Session-balanced held-out arm accuracy")
    fig.suptitle("Task-only feasibility: dots = observed; gray = null median to p95", fontsize=11)
    fig.tight_layout()
    fig.savefig(target, dpi=160)
    plt.close(fig)


def run(args):
    preflight = args.preflight_dir.resolve()
    root, output = args.dataset_root.resolve(), args.output_dir.resolve()
    m = json.loads((preflight / "roscow_preflight_manifest.json").read_text())
    for name, sha in m["outputs"].items():
        if file_sha256(preflight / name) != sha:
            raise ValueError("preflight payload changed")
    if file_sha256(root / "download_status.json") != m["download_status_sha256"]:
        raise ValueError("download provenance changed")
    download = json.loads((root / "download_status.json").read_text())
    verified = {x["path"]: x["sha256"] for x in download["verified"]}
    provenance = build_script_provenance(cwd=ROOT, input_paths={
        "preflight_manifest": preflight / "roscow_preflight_manifest.json",
        "download_record": root / "download_status.json",
        "producer": Path(__file__), "protocol": ROOT / "docs/roscow_task_arm_protocol.md",
        "posterior_helper": ROOT / "scripts/validate_dandi000978_run_readouts.py",
        "preflight_helper": ROOT / "scripts/preflight_roscow2025.py",
        "provenance_helper": ROOT / "scripts/_provenance.py",
    })
    if provenance["git_dirty"] or provenance["code_commit"] == "unavailable":
        raise ValueError("Clean committed producer required")
    session_metadata = pd.read_csv(preflight / "roscow_preflight_sessions.csv")
    trial_metadata = pd.read_csv(preflight / "roscow_preflight_trials.csv")
    output.mkdir(parents=True, exist_ok=False)
    session_rows, predictions, null_frames, arrays, raw_hashes = [], [], [], {}, {}
    for row in session_metadata.itertuples():
        key = {"animal": row.animal, "session": int(row.session)}
        decision = {**key, "status": "excluded", "reason": "", "n_ca1_units": row.n_ca1_units}
        trials = trial_metadata[(trial_metadata.animal == row.animal) & (trial_metadata.session == row.session)
                                & trial_metadata.eligible_probabilistic_outcome].sort_values("trial_index_1based")
        support = trials.chosen_arm_1based.value_counts().reindex([1, 2, 3], fill_value=0)
        decision.update(n_trials=len(trials), min_trials_per_arm=int(support.min()))
        if row.status != "metadata_pass":
            decision["reason"] = "preflight_metadata_failure"
        elif row.n_ca1_units < PARAMETERS["minimum_ca1_units"]:
            decision["reason"] = "too_few_ca1_units"
        elif support.min() < PARAMETERS["minimum_trials_per_arm"]:
            decision["reason"] = "too_few_trials_per_arm"
        if decision["reason"]:
            session_rows.append(decision)
            continue
        code = next(k for k, name in ANIMALS.items() if name == row.animal)
        folder = next(p for p in (root / "raw/data/ephys_data" / f"Rat_{code}").iterdir()
                      if p.is_dir() and session_number(p) == row.session)
        for name in ("spiketimes.mat", "nNAcUnits.mat", "startEndTimes.mat"):
            p = folder / name
            rel = str(p.relative_to(root / "raw"))
            if file_sha256(p) != verified[rel]:
                raise ValueError(f"raw input changed: {rel}")
            raw_hashes[rel] = verified[rel]
        spikes = loadmat(folder / "spiketimes.mat")["spiketimes"].reshape(-1)
        n_striatal = int(read_single(folder / "nNAcUnits.mat"))
        cells = spikes[n_striatal:]
        if len(cells) != row.n_ca1_units:
            raise ValueError("CA1 boundary differs from preflight")
        epochs = np.asarray(read_single(folder / "startEndTimes.mat"))
        counts = count_windows(cells, trials.reward_arrival_time_s.to_numpy(), epochs[1])
        labels = trials.chosen_arm_1based.to_numpy(dtype=int) - 1
        exposure = PARAMETERS["window_stop_s"] - PARAMETERS["window_start_s"]
        nulls, probs = score_session(counts, labels, exposure)
        for k, v in key.items():
            nulls[k] = v
        null_frames.append(nulls)
        prefix = f"{row.animal}_session{row.session}"
        arrays[prefix + "_counts"] = counts
        arrays[prefix + "_labels"] = labels
        arrays[prefix + "_trials"] = trials.trial_index_1based.to_numpy(dtype=int)
        for kind, prob in probs.items():
            for i, trial in enumerate(trials.itertuples()):
                predictions.append({
                    **key, "readout": kind, "trial_index_1based": trial.trial_index_1based,
                    "chosen_arm_1based": int(labels[i] + 1), "n_spikes": int(counts[i].sum()),
                    "n_active_units": int((counts[i] > 0).sum()),
                    **{f"p_arm{j + 1}": float(prob[i, j]) for j in range(3)},
                })
            decision[kind + "_balanced_accuracy"] = scores(prob, labels)["balanced_accuracy"]
        decision.update(status="scored", n_label_shifts=len(labels))
        session_rows.append(decision)
        print(json.dumps(decision), flush=True)
    nulls = pd.concat(null_frames, ignore_index=True) if null_frames else pd.DataFrame(columns=["animal", "session", "readout"])
    animals = summarize_animals(nulls)
    gates = [{"gate": "at_least_two_sessions_each_animal", "passed": bool((animals.n_sessions >= 2).all())}]
    for kind in ("poisson", "composition"):
        gates.append({"gate": kind + "_above_shift_null_each_animal", "passed": bool(animals.loc[animals.readout == kind, "screen_pass"].all())})
    gates.append({"gate": "task_arm_feasibility", "passed": all(g["passed"] for g in gates)})
    for name, data in (("sessions", pd.DataFrame(session_rows)), ("trial_predictions", pd.DataFrame(predictions)),
                       ("shift_controls", nulls), ("by_animal", animals), ("gates", pd.DataFrame(gates))):
        data.to_csv(output / f"roscow_task_arm_{name}.csv", index=False)
    np.savez_compressed(output / "task_arm_count_matrices.npz", **arrays)
    write_figure(animals, output / "roscow_task_arm_readout.png")
    summary = ["# Task-arm readout feasibility", "", f"Feasibility screen passed: {gates[-1]['passed']}.",
               "No rest replay or reward/surprise effect was tested.",
               "Readout is categorical arm identity, not position, direction or sequence content.",
               "A passing screen does not remove behavioral-state/expectation confounds.",
               "Circular shifts preserve repetitive action structure and can be conservative.", ""]
    for r in animals.itertuples():
        if r.n_sessions:
            summary.append(f"- {r.animal}, {r.readout}: {r.n_sessions} sessions, accuracy {r.mean_session_balanced_accuracy:.3f}; null p95 {r.null_p95:.3f}.")
    (output / "roscow_task_arm_summary.md").write_text("\n".join(summary) + "\n")
    manifest = {**provenance, "parameters": PARAMETERS, "source_release": download["source_commit"],
                "created_at_utc": datetime.now(UTC).isoformat(), "host": socket.gethostname(),
                "dataset_root": str(root), "raw_input_sha256": raw_hashes,
                "task_arm_feasibility_passed": gates[-1]["passed"],
                "biological_hypothesis_scored": False, "rest_replay_scored": False,
                "outputs": {p.name: file_sha256(p) for p in output.iterdir() if p.is_file()}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--preflight-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
