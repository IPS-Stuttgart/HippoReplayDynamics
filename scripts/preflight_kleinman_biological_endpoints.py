"""Inventory intervention labels and neural payloads without decoding outcomes.

Reward schedules are planned entries, not validated delivered outcomes. Missing
novelty labels (Experiment 2) remain missing rather than implying familiarity.
"""

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.io import loadmat, whosmat


def binary_label(value, name):
    values = np.asarray(value, dtype=float)
    if values.size != 1 or float(values.reshape(-1)[0]) not in (0.0, 1.0):
        raise ValueError(f"invalid {name} label")
    return int(values.reshape(-1)[0])


def visit_count(info, key):
    values = np.asarray(info.get(key, np.empty((0, 2))), dtype=float)
    if values.size == 0:
        return 0
    values = np.atleast_2d(values)
    if values.shape[1] != 2 or not np.isfinite(values).all():
        raise ValueError(f"invalid {key} intervals")
    return len(values)


def inspect_session(path):
    path = Path(path)
    folder = path.parent
    row = {
        "experiment": folder.parent.parent.name,
        "subject": folder.parent.name,
        "session": folder.name,
        "session_info_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "spike_data_present": (folder / "spike_data.mat").is_file(),
        "sdes_present": (folder / "sdes.mat").is_file(),
        "ripple_events_present": (folder / "ripple_events.mat").is_file(),
        "schedule_alignment_validated": False,
        "unit_identity_validated": False,
    }
    try:
        info = loadmat(path, simplify_cells=True)["session_info"]
        row.update(
            drug=binary_label(info["drug"], "drug"),
            novel_label_present="novel" in info,
            novel=binary_label(info["novel"], "novel") if "novel" in info else "",
            info_fields=";".join(sorted(info)),
            left_visit_rows=visit_count(info, "left_visit"),
            right_visit_rows=visit_count(info, "right_visit"),
            epoch_change_present="epoch_change" in info,
            reward_end_label_present="incr_end" in info or "volatile_end" in info,
        )
        schedule = np.asarray(info.get("reward_schedule", np.empty((0, 2))), float)
        if schedule.size:
            schedule = np.atleast_2d(schedule)
            if (schedule.ndim != 2 or schedule.shape[1] != 2
                    or not np.isfinite(schedule).all() or (schedule < 0).any()):
                raise ValueError("invalid reward schedule")
        row.update(
            reward_schedule_present=bool(schedule.size),
            reward_schedule_rows=len(schedule),
            zero_reward_schedule_entries=int((schedule == 0).sum()),
            planned_reward_values=";".join(str(float(x)) for x in np.unique(schedule)),
        )
        for name in ("spike_data", "sdes", "ripple_events"):
            file = folder / f"{name}.mat"
            entries = whosmat(file) if file.is_file() else []
            matching = [shape for key, shape, _ in entries if key == name]
            row[name + "_shape"] = json.dumps(matching[0]) if matching else "missing"
        row["status"] = "metadata_readable"
    except (ValueError, KeyError, TypeError, OSError, NotImplementedError) as exc:
        row.update(status="failed", error=repr(exc))
    return row


def summarize_inventory(rows):
    summary = []
    for experiment in sorted({r["experiment"] for r in rows}):
        group = [r for r in rows if r["experiment"] == experiment]
        summary.append({
            "experiment": experiment,
            "sessions": len(group),
            "subjects": len({r["subject"] for r in group}),
            "sessions_with_sorted_spikes": sum(r["spike_data_present"] for r in group),
            "sessions_with_sdes": sum(r["sdes_present"] for r in group),
            "sessions_with_reward_schedule": sum(
                r.get("reward_schedule_present", False) for r in group),
            "sessions_with_omission_schedule_and_spikes": sum(
                r.get("zero_reward_schedule_entries", 0) > 0 and r["spike_data_present"]
                for r in group),
            "sessions_with_novelty_label": sum(r.get("novel_label_present", False) for r in group),
            "sessions_with_epoch_change": sum(r.get("epoch_change_present", False) for r in group),
            "metadata_failures": sum(r["status"] == "failed" for r in group),
        })
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    rows = [inspect_session(path) for path in sorted(
        args.dataset_root.glob("Experiment_*/*/*/session_info.mat"))]
    if not rows:
        raise ValueError("no session_info files found")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    columns = list(dict.fromkeys(k for row in rows for k in row))
    destination = output / "kleinman_intervention_inventory.csv"
    with destination.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize_inventory(rows)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),  # noqa: UP017 (4090 has Python 3.10)
        "dataset_root": str(args.dataset_root.resolve()),
        "real_events_scored": False,
        "schedule_alignment_validated": False,
        "interpretation": "Payload inventory only; schedule entries are not confirmed delivered outcomes.",
        "summary": summary,
        "source_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "output_sha256": {destination.name: hashlib.sha256(destination.read_bytes()).hexdigest()},
    }
    (output / "inventory_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

