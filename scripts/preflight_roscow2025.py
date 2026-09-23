#!/usr/bin/env python3
"""Audit public Roscow trial/neural alignment; no replay or outcome hypothesis scoring."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import socket
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

ANIMALS = {"Q": "Quirinius", "S": "Severus", "T": "Trevor"}


def vector(x):
    return np.asarray(x, dtype=float).reshape(-1)


def timestamps(x):
    x = vector(x)
    if not np.isfinite(x).all() or (x < 0).any() or (np.diff(x) < 0).any():
        raise ValueError("nonfinite, negative, or unsorted timestamps")
    return x


def session_field(behavior, field, session):
    n_sessions = np.size(behavior["n_trials"])
    if not 1 <= session <= n_sessions:
        raise ValueError("session number outside behavioral table")
    x = behavior[field]
    if n_sessions == 1:
        return x
    if len(x) != n_sessions:
        raise ValueError(f"session axis mismatch: {field}")
    return x[session - 1]


def trial_table(behavior, session, arrival, task_bounds):
    """Use the authors' SessionN -> behavior{N} mapping, not folder order."""
    n = int(vector(behavior["n_trials"])[session - 1])
    actions = vector(session_field(behavior, "actions", session))
    rewards = vector(session_field(behavior, "rewarded", session))
    probs = np.asarray(session_field(behavior, "reward_probs", session), dtype=str).reshape(-1)
    values = session_field(behavior, "arm_values", session)
    if n == 1:
        values = [values]
    arm_values = np.asarray([np.asarray(x, dtype=str).reshape(-1) for x in values])
    arrival = timestamps(arrival)
    if n < 1 or any(len(x) != n for x in (actions, rewards, arrival)):
        raise ValueError("trial count/action/reward/arrival mismatch")
    if probs.shape != (3,) or set(probs) != {"low", "medium", "high"}:
        raise ValueError("unexpected arm-probability labels")
    if arm_values.shape != (n, 3):
        raise ValueError("arm legitimacy table shape mismatch")
    if not set(arm_values.reshape(-1)).issubset({"Legitimate", "Illegitimate", "Optimal"}):
        raise ValueError("unknown arm legitimacy label")
    if not np.isin(actions, [1, 2, 3]).all() or not np.isin(rewards, [0, 1]).all():
        raise ValueError("invalid action/reward labels")
    if (np.diff(arrival) <= 0).any() or not ((arrival >= task_bounds[0]) & (arrival <= task_bounds[1])).all():
        raise ValueError("arrival timestamps not unique or outside task epoch")
    chosen = actions.astype(int) - 1
    labels = arm_values[np.arange(n), chosen]
    if ((labels == "Illegitimate") & (rewards == 1)).any():
        raise ValueError("rewarded illegitimate trial")
    initial = np.asarray(session_field(behavior, "reward_probs", 1), dtype=str).reshape(-1)
    return pd.DataFrame({
        "trial_index_1based": np.arange(1, n + 1), "reward_arrival_time_s": arrival,
        "chosen_arm_1based": actions.astype(int), "rewarded": rewards.astype(int),
        "chosen_probability_label": probs[chosen], "chosen_legitimacy": labels,
        "eligible_probabilistic_outcome": labels != "Illegitimate",
        "probability_assignment_changed_from_session1": not np.array_equal(probs, initial),
        "alignment_rule": "authors_SessionN_to_behavior_cell_N_then_trial_order",
    })


def unit_table(spikes, n_striatal, epochs):
    if np.shape(epochs) != (3, 2) or not np.isfinite(epochs).all():
        raise ValueError("expected finite PRE/TASK/POST epoch bounds")
    if (epochs[:, 1] <= epochs[:, 0]).any() or (epochs[1:, 0] < epochs[:-1, 1]).any():
        raise ValueError("invalid or overlapping epochs")
    cells = np.asarray(spikes, dtype=object).reshape(-1)
    if not float(n_striatal).is_integer() or not 0 <= n_striatal < len(cells):
        raise ValueError("invalid striatal/CA1 unit boundary")
    rows = []
    for i, cell in enumerate(cells):
        t = timestamps(cell)
        row = {"unit_index_1based": i + 1, "region": "vStr" if i < n_striatal else "CA1", "n_spikes": len(t)}
        for (start, stop), phase in zip(epochs, ("pre", "task", "post"), strict=True):
            row[f"{phase}_spikes"] = int(((t >= start) & (t < stop)).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def read_single(path):
    data = {k: v for k, v in loadmat(path, simplify_cells=True).items() if not k.startswith("__")}
    if len(data) != 1:
        raise ValueError(f"expected exactly one MATLAB variable: {path.name}")
    return next(iter(data.values()))


def session_number(folder):
    match = re.fullmatch(r"session(\d+)", folder.name, re.IGNORECASE)
    if match is None:
        raise ValueError(f"Unrecognized session directory: {folder}")
    return int(match[1])


def audit_session(folder, behavior):
    animal = ANIMALS[folder.parent.name.removeprefix("Rat_")]
    session = session_number(folder)
    identity = {"animal": animal, "session": session}
    row = dict(identity)
    issues = []
    units, trials = pd.DataFrame(), pd.DataFrame()
    epochs = np.asarray(read_single(folder / "startEndTimes.mat"), dtype=float)
    try:
        # Unsqueezed cell array retains the unit axis even for a single unit.
        spikes = loadmat(folder / "spiketimes.mat")["spiketimes"]
        units = unit_table(spikes, float(read_single(folder / "nNAcUnits.mat")), epochs)
        for key, val in identity.items():
            units[key] = val
        row.update(n_units=len(units), n_ca1_units=int((units.region == "CA1").sum()),
                   n_striatal_units=int((units.region == "vStr").sum()), neural_timestamps_valid=True)
        for i, phase in enumerate(("pre", "task", "post")):
            row[f"{phase}_duration_s"] = float(epochs[i, 1] - epochs[i, 0])
            row[f"{phase}_ca1_spikes"] = int(units.loc[units.region == "CA1", f"{phase}_spikes"].sum())
    except (ValueError, KeyError, TypeError) as exc:
        issues.append("neural: " + str(exc))
        row["neural_timestamps_valid"] = False
    try:
        trials = trial_table(behavior, session, read_single(folder / "RarrivalTimes.mat"), epochs[1])
        for key, val in identity.items():
            trials[key] = val
        eligible = trials[trials.eligible_probabilistic_outcome]
        row.update(trial_alignment_passed=True, n_trials=len(trials),
                   n_eligible_trials=len(eligible), n_illegitimate_trials=len(trials) - len(eligible),
                   n_rewarded_eligible=int(eligible.rewarded.sum()),
                   n_unrewarded_eligible=int((eligible.rewarded == 0).sum()),
                   probability_assignment_changed_from_session1=bool(trials.probability_assignment_changed_from_session1.iloc[0]))
        for label in ("low", "medium", "high"):
            subset = eligible[eligible.chosen_probability_label == label]
            row[f"{label}_rewarded"] = int(subset.rewarded.sum())
            row[f"{label}_unrewarded"] = int((subset.rewarded == 0).sum())
    except (ValueError, KeyError, TypeError, IndexError) as exc:
        issues.append("trials: " + str(exc))
        row["trial_alignment_passed"] = False
    try:
        ripples = loadmat(folder / "ripples3std.mat", simplify_cells=True)
        start, peak, end = (timestamps(ripples[x]) for x in ("rippleStart", "ripplePeak", "rippleStop"))
        if not (len(start) == len(peak) == len(end)) or not ((start <= peak) & (peak <= end) & (end > start)).all():
            raise ValueError("invalid ripple interval/peak alignment")
        row.update(ripple_timestamps_valid=True, n_native_ripples=len(peak),
                   median_ripple_duration_ms=float(np.median(end - start) * 1000))
        for (a, b), phase in zip(epochs, ("pre", "task", "post"), strict=True):
            row[f"{phase}_native_ripples"] = int(((start >= a) & (end <= b)).sum())
    except (ValueError, KeyError, TypeError) as exc:
        issues.append("ripples: " + str(exc))
        row["ripple_timestamps_valid"] = False
    for filename, prefix in (("CPentryTimes.mat", "cp_entry"), ("CPexitTimes.mat", "cp_exit")):
        try:
            ts = timestamps(read_single(folder / filename))
            row[f"n_{prefix}_timestamps"] = len(ts)
        except (ValueError, KeyError, TypeError) as exc:
            issues.append(prefix + ": " + str(exc))
    row.update(spatial_xy_available=False, sequence_content_validated=False,
               cp_trial_cycle_alignment_validated=False,
               status="metadata_pass" if not issues else "needs_review", issues="; ".join(issues))
    return row, trials, units


def run(root, output):
    status_path = root / "download_status.json"
    status = json.loads(status_path.read_text())
    if status["status"] != "complete_verified" or status["assets_verified"] != status["assets_expected"]:
        raise ValueError("Acquisition must be fully verified before cohort audit")
    for item in status["verified"]:
        path = root / "raw" / item["path"]
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != item["sha256"]:
                raise ValueError(f"Input changed since download: {path}")
    rows, trial_frames, unit_frames = [], [], []
    for code, name in ANIMALS.items():
        behavior = read_single(root / "raw/data/behavioural_data" / f"behaviour_{name.lower()}.mat")
        folders = sorted((p for p in (root / "raw/data/ephys_data" / f"Rat_{code}").iterdir()
                          if p.is_dir()), key=session_number)
        for folder in folders:
            row, trials, units = audit_session(folder, behavior)
            rows.append(row)
            if not trials.empty:
                trial_frames.append(trials)
            if not units.empty:
                unit_frames.append(units)
            print(json.dumps(row), flush=True)
    if not rows:
        raise ValueError("No sessions found")
    sessions = pd.DataFrame(rows)
    trials = pd.concat(trial_frames, ignore_index=True) if trial_frames else pd.DataFrame()
    units = pd.concat(unit_frames, ignore_index=True) if unit_frames else pd.DataFrame()
    outcome = pd.DataFrame(columns=["animal", "chosen_probability_label", "rewarded", "n_trials"])
    if not trials.empty:
        trials = trials.merge(sessions[["animal", "session", "status"]], on=["animal", "session"],
                              how="left", validate="many_to_one")
        eligible = trials[trials.eligible_probabilistic_outcome & (trials.status == "metadata_pass")]
        outcome = eligible.groupby(["animal", "chosen_probability_label", "rewarded"]).size().reset_index(name="n_trials")
    animal = sessions.groupby("animal").agg(
        sessions=("session", "size"), trial_aligned_sessions=("trial_alignment_passed", "sum"),
        neural_valid_sessions=("neural_timestamps_valid", "sum"),
        native_ripple_valid_sessions=("ripple_timestamps_valid", "sum"),
    ).reset_index()
    output.mkdir(parents=True, exist_ok=False)
    for name, table in (("sessions", sessions), ("trials", trials), ("units", units), ("by_animal", animal), ("by_outcome", outcome)):
        table.to_csv(output / f"roscow_preflight_{name}.csv", index=False)
    try:
        repo = Path(__file__).resolve().parents[1]
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = "unavailable", None
    manifest = {
        "created_at_utc": datetime.now(UTC).isoformat(), "host": socket.gethostname(),
        "dataset_root": str(root), "source_commit": status["source_commit"], "code_commit": commit,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "download_status_sha256": hashlib.sha256(status_path.read_bytes()).hexdigest(),
        "command_line": sys.argv, "working_directory": str(Path.cwd()), "git_dirty": dirty,
        "n_sessions": len(sessions), "n_animals": len(animal),
        "trial_aligned_sessions": int(sessions.trial_alignment_passed.sum()),
        "n_aligned_trials": len(trials), "n_metadata_pass_sessions": int((sessions.status == "metadata_pass").sum()),
        "n_eligible_trials_in_metadata_pass_sessions": int(outcome.n_trials.sum()), "hypothesis_scored": False,
        "sequence_content_ready": False,
        "outputs": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.glob("*.csv")},
    }
    (output / "roscow_preflight_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    text = ["# Roscow public-data feasibility audit", "",
            f"Neural animals: {len(animal)}. Sessions: {len(sessions)}.",
            f"Trial-aligned sessions: {manifest['trial_aligned_sessions']}; aligned trials: {len(trials)}.",
            f"Metadata passes: {int((sessions.status == 'metadata_pass').sum())}.", "",
            "This is an input audit, not evidence for surprise-driven replay.",
            "Reward probability is a task label, not a fitted subjective expectation/RPE.",
            "Illegitimate arm entries are excluded from probabilistic outcome contrasts.",
            "Unit counts are session-unit records, not necessarily unique neurons across days.",
            "Reward-arrival rows follow the authors' explicit session/trial indexing convention.",
            "Central-platform timestamps have not been validated as complete trial cycles.",
            "No continuous x/y positions were found in the acquired electrophysiology files.",
            "A task-state sequence decoder still requires independent validation.",
            "PRE/POST home-cage ripple counts are not trial-triggered replay outcomes.", "",
            "## Sessions requiring review", ""]
    for row in rows:
        if row["status"] != "metadata_pass":
            text.append(f"- {row['animal']} Session{row['session']}: {row['issues']}")
    (output / "roscow_preflight_summary.md").write_text("\n".join(text) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.dataset_root.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    main()
