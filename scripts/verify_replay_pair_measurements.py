"""Independent raw-spike reconstruction of the saved candidate pair-order table.

Verifies all measured originals, candidate accounting and input/output hashes.
It does not validate the unimplemented coordination endpoint or theta control.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hipporeplayimm.data import load_replay_session  # noqa: E402

try:
    from scripts._provenance import build_script_provenance, file_sha256
except ModuleNotFoundError:
    from _provenance import build_script_provenance, file_sha256


def reference_order(a_bins: np.ndarray, b_bins: np.ndarray, width: float, lower: float, upper: float) -> float:
    if not len(a_bins) or not len(b_bins):
        return 0.0
    delta = (b_bins[None, :] - a_bins[:, None]) * width
    positive = np.count_nonzero((delta >= lower - 1e-12) & (delta <= upper + 1e-12))
    negative = np.count_nonzero((-delta >= lower - 1e-12) & (-delta <= upper + 1e-12))
    return float((positive - negative) / (len(a_bins) * len(b_bins)))


def verify_inputs(manifest: dict, dataset_root: Path) -> None:
    expected: dict[str, dict[str, str]] = {s: {} for s in manifest["protocol"]["sessions"]}
    checked = {}
    for row in manifest["input_files"]:
        path = Path(row["path"])
        session_id = "/".join(path.parent.parts[-2:])
        if session_id not in expected or path.name in expected[session_id]:
            raise ValueError("Unrecognized or duplicate input identity")
        digest = file_sha256(path)
        if digest != row["sha256"]:
            raise ValueError(f"Input hash mismatch: {path}")
        checked[path.resolve()] = digest
        expected[session_id][path.name] = digest
    for session_id, files in expected.items():
        folder = dataset_root / session_id
        actual = {f.name for f in folder.iterdir() if f.is_file()}
        if not files or actual != set(files):
            raise ValueError(f"Dataset file inventory mismatch: {session_id}")
        for name, digest in files.items():
            path = folder / name
            loaded_digest = checked.get(path.resolve()) or file_sha256(path)
            if loaded_digest != digest:
                raise ValueError(f"Loaded dataset hash mismatch: {path}")


def verify(root: Path, dataset_root: Path) -> dict:
    manifest = json.loads((root / "manifest.json").read_text())
    p = manifest["protocol"]
    verify_inputs(manifest, dataset_root)
    for name, digest in manifest["outputs"].items():
        if file_sha256(root / name) != digest:
            raise ValueError(f"Output hash mismatch: {name}")
    events = pd.read_csv(root / "candidate_event_inventory.csv", float_precision="round_trip")
    pauses = pd.read_csv(root / "pause_inventory.csv", float_precision="round_trip")
    pairs = pd.read_csv(root / "candidate_pair_order.csv", float_precision="round_trip")
    summaries = pd.read_csv(root / "session_summary.csv")
    if pairs.empty or events.empty or pauses.empty:
        raise ValueError("No measured pairs; empty measurements cannot pass verification")
    if summaries["session"].duplicated().any() or set(summaries["session"]) != set(p["sessions"]):
        raise ValueError("Session inventory mismatch")
    for frame in (events, pauses, pairs):
        if not set(frame["session"]).issubset(set(p["sessions"])):
            raise ValueError("Unrecognized table session")
    for frame, names in ((events, ("order_measured", "eligible_candidate")),
                         (pauses, ("matched_run_supported", "nonoverlapping_measurement_subset"))):
        if any(frame[name].dtype != bool for name in names):
            raise ValueError("Invalid or missing boolean flags")
    if events["event_id"].duplicated().any() or pauses["pause_id"].duplicated().any():
        raise ValueError("Duplicate event/pause identities")
    if pairs[["event_id", "cell_a", "cell_b"]].duplicated().any():
        raise ValueError("Duplicate pair rows")
    event_map = events.set_index("event_id")
    pause_map = pauses.set_index("pause_id")
    measured = set(events.loc[events["order_measured"], "event_id"])
    if set(pairs["event_id"]) != measured:
        raise ValueError("Measured event/pair accounting mismatch")
    verified_rows, max_error, max_shuffle_error = 0, 0.0, 0.0
    for session_id in p["sessions"]:
        session = load_replay_session(dataset_root / session_id)
        raw = session.excitatory_spikes()
        frame = pairs[pairs["session"] == session_id]
        session_events = events[events["session"] == session_id]
        session_pauses = pauses[pauses["session"] == session_id]
        for pause in session_pauses.itertuples():
            contained = np.flatnonzero((session.ripple_events[:, 0] >= pause.start_s) &
                                      (session.ripple_events[:, 1] <= pause.end_s))
            saved = session_events[session_events["pause_id"] == pause.pause_id]
            if set(saved["event_index"]) != set(contained) or len(saved) != len(contained):
                raise ValueError("Contained native candidate accounting mismatch")
            if len(contained) != pause.native_contained_events:
                raise ValueError("Pause native-event count mismatch")
            if int(saved["eligible_candidate"].sum()) != pause.eligible_candidate_events:
                raise ValueError("Pause eligible-event count mismatch")
            if len(frame[frame["pause_id"] == pause.pause_id]) != pause.candidate_pair_rows:
                raise ValueError("Pause pair accounting mismatch")
        for record in session_events.itertuples():
            if record.pause_id not in pause_map.index:
                raise ValueError("Orphan candidate event")
            pause = pause_map.loc[record.pause_id]
            native = session.ripple(int(record.event_index))
            if pause["session"] != session_id or native.start != record.start_s or native.end != record.end_s:
                raise ValueError("Native event identity/timing mismatch")
            if not (pause["start_s"] <= native.start < native.end <= pause["end_s"]):
                raise ValueError("Candidate crosses its pause")
            if record.order_measured != (record.eligible_candidate and pause["nonoverlapping_measurement_subset"]):
                raise ValueError("Measured candidate eligibility mismatch")
        for identity, group in frame.groupby("event_id"):
            row = event_map.loc[identity]
            pause = pause_map.loc[row["pause_id"]]
            index = int(row["event_index"])
            native = session.ripple(index)
            if native.start != row["start_s"] or native.end != row["end_s"]:
                raise ValueError("Native event identity/timing mismatch")
            if not (pause["start_s"] <= native.start < native.end <= pause["end_s"]):
                raise ValueError("Candidate crosses its pause")
            n_bins = int(np.ceil((native.end - native.start) / p["order_target_bin_s"]))
            width = (native.end - native.start) / n_bins
            hit = raw[(raw[:, 0] >= native.start) & (raw[:, 0] < native.end)]
            bins = np.minimum(n_bins - 1, np.floor((hit[:, 0] - native.start) / width).astype(int))
            by_cell = {int(c): bins[hit[:, 1] == c] for c in np.unique(hit[:, 1])}
            seed = int.from_bytes(hashlib.sha256(f"{p['seed']}:{identity}".encode()).digest()[:8], "little")
            rng = np.random.default_rng(seed)
            inverse_permutations = [np.argsort(rng.permutation(n_bins)) for _ in range(p["order_shuffles"])]
            cells = set(group["cell_a"]) | set(group["cell_b"])
            if len(cells) != row["n_active_eligible_units"] or (group["cell_a"] >= group["cell_b"]).any():
                raise ValueError("Active-cell pair identity mismatch")
            if len(group) != len(cells) * (len(cells) - 1) // 2:
                raise ValueError("Incomplete active-cell pair enumeration")
            if sum(len(by_cell[int(c)]) for c in cells) != row["n_eligible_spikes"]:
                raise ValueError("Measured cell spike-count mismatch")
            for record in group.itertuples():
                a, b = by_cell[record.cell_a], by_cell[record.cell_b]
                if len(a) != record.a_spikes or len(b) != record.b_spikes:
                    raise ValueError("Pair cell-count mismatch")
                value = reference_order(a, b, width, p["order_min_lag_s"], p["order_max_lag_s"])
                error = abs(value - record.a_before_b_asymmetry)
                max_error = max(max_error, error)
                if not np.isfinite(error) or error > 1e-12 or record.n_shuffles != p["order_shuffles"]:
                    raise ValueError("Original order reconstruction/shuffle-count mismatch")
                if abs(record.bin_width_s - width) > 1e-12:
                    raise ValueError("Event bin width mismatch")
                nulls = [reference_order(inv[a], inv[b], width, p["order_min_lag_s"], p["order_max_lag_s"])
                         for inv in inverse_permutations]
                shuffle_error = max(abs(np.mean(nulls) - record.shuffle_mean_asymmetry),
                                    abs(np.std(nulls, ddof=1) - record.shuffle_sd_asymmetry))
                if not np.isfinite(shuffle_error) or shuffle_error > 1e-12:
                    raise ValueError("Shuffle summary reconstruction mismatch")
                max_shuffle_error = max(max_shuffle_error, shuffle_error)
                verified_rows += 1
    for row in summaries.itertuples():
        if len(pairs[pairs["session"] == row.session]) != row.candidate_pair_rows:
            raise ValueError("Session pair accounting mismatch")
        if len(events[(events["session"] == row.session) & events["order_measured"]]) != row.order_measured_events:
            raise ValueError("Session event accounting mismatch")
        saved_pauses = pauses[pauses["session"] == row.session]
        if len(saved_pauses) != row.immobile_pauses:
            raise ValueError("Session pause accounting mismatch")
        if int(saved_pauses["matched_run_supported"].sum()) != row.matched_run_supported_pauses:
            raise ValueError("Session supported-pause accounting mismatch")
        if int(saved_pauses["nonoverlapping_measurement_subset"].sum()) != row.nonoverlapping_supported_pauses:
            raise ValueError("Session nonoverlapping-pause accounting mismatch")
    return {"created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "verified_measurement_subset",
            "input_hashes_verified": len(manifest["input_files"]),
            "output_hashes_verified": len(manifest["outputs"]),
            "sessions_reconciled": len(p["sessions"]), "pauses_reconciled": len(pauses),
            "contained_candidates_reconciled": len(events),
            "original_pair_scores_independently_reconstructed": verified_rows,
            "shuffle_pair_summaries_independently_reconstructed": verified_rows,
            "measured_events": len(measured), "max_absolute_original_order_error": max_error,
            "max_absolute_shuffle_summary_error": max_shuffle_error,
            "theta_control_verified": False, "future_coordination_endpoint_verified": False,
            "biological_association_verified": False, "full_goal_complete": False,
            "scope": "all measured original pair scores, deterministic whole-bin shuffle means/SDs, native candidate accounting, and hashes; RUN exposure matching and phase synchronization are not independently verified here"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurement-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Preserve earlier verifications; choose a new output")
    result = verify(args.measurement_dir, args.dataset_root)
    result["provenance"] = build_script_provenance(input_paths={"measurement_manifest": args.measurement_dir / "manifest.json"})
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
