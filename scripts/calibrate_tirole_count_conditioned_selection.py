"""Paired likelihood sensitivity on unchanged known-label simulated observations."""

import argparse
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from hipporeplayimm.tirole_two_track import file_sha256
from hipporeplayimm.two_track_content import classify_sequence
from scripts.calibrate_tirole_selection_transport import simulate_anchor
from scripts.score_tirole_content_coverage import validate_bank

KEYS = ["anchor_event", "split", "draw", "truth_track", "generator", "repeat"]
UNCHANGED = [
    "session",
    "animal",
    "epoch",
    "ripple_positive",
    "true_path_span_cm",
    "fraction",
    "n_inference_cells",
    "n_inference_spikes",
    "n_time_bins",
    "n_active_inference",
    "n_nonempty_bins",
    "sequence_eligible",
]


def compare_unchanged(new_scores, old_scores, new_content, old_content):
    if new_scores.duplicated(KEYS).any() or old_scores.duplicated(KEYS).any():
        raise ValueError("duplicate paired score keys")
    pd.testing.assert_frame_equal(new_scores.set_index(KEYS)[UNCHANGED].sort_index(), old_scores.set_index(KEYS)[UNCHANGED].sort_index(), check_exact=False, rtol=1e-12, atol=1e-12)
    keys = KEYS[:4]
    pd.testing.assert_frame_equal(new_content.set_index(keys).sort_index(), old_content.set_index(keys).sort_index(), check_exact=False, rtol=1e-12, atol=1e-12)


def run(source, output):
    if output.exists():
        raise ValueError("new immutable likelihood-sensitivity run required")
    git = lambda *args: subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    commit = git("rev-parse", "HEAD")
    if git("status", "--porcelain"):
        raise ValueError("freeze protocol before sensitivity scoring")
    m = json.loads((source / "manifest.json").read_text())
    if (
        m["status"] != "complete"
        or m["git_dirty"]
        or not m["counts_conditioned_on_recorded_totals"]
        or m["n_anchors"] != 40
        or m["n_draws"] != 20
        or m["n_coverage_repeats"] != 5
        or m["n_sequence_nulls"] != 499
    ):
        raise ValueError("frozen original transport required")
    for name, h in m["output_sha256"].items():
        if file_sha256(source / name) != h:
            raise ValueError("changed source transport")
    bank = Path(m["bank_dir"])
    info, parts, events, counts, offsets, maps = validate_bank(bank)
    if file_sha256(bank / "manifest.json") != m["bank_manifest_sha256"]:
        raise ValueError("candidate bank changed")
    anchors = pd.read_csv(source / "count_anchors.csv").sort_values("event_id")
    pd.testing.assert_frame_equal(events.set_index("event_id").loc[anchors.event_id].reset_index(), anchors.reset_index(drop=True), check_exact=False)
    output.mkdir(parents=True)
    (output / "sequence_shards").mkdir()
    (output / "count_anchors.csv").write_bytes((source / "count_anchors.csv").read_bytes())
    hashes = {"count_anchors.csv": file_sha256(output / "count_anchors.csv")}
    started = time.monotonic()
    n_scores = 0
    for rank, event in enumerate(anchors.to_dict("records"), 1):
        eid = int(event["event_id"])
        original = counts[offsets[eid] : offsets[eid + 1]]
        scores, content = simulate_anchor(info, parts, event, original, maps, sequence_classifier=partial(classify_sequence, conditional_count=True))
        old_scores = pd.read_csv(source / "sequence_shards" / f"{eid}.csv")
        old_content = pd.read_csv(source / "content_shards" / f"{eid}.csv")
        compare_unchanged(scores, old_scores, content, old_content)
        if len(scores) != 2400 or len(content) != 200:
            raise ValueError("incomplete anchor")
        scores["decoder_likelihood"] = "conditional_population_identity_given_count"
        path = output / "sequence_shards" / f"{eid}.csv"
        scores.to_csv(path, index=False)
        hashes[str(path.relative_to(output))] = file_sha256(path)
        n_scores += len(scores)
        print(json.dumps({"session": info["session"], "anchors_completed": rank, "anchors_total": 40, "elapsed_s": time.monotonic() - started}), flush=True)
    if git("rev-parse", "HEAD") != commit or git("status", "--porcelain"):
        raise ValueError("code changed during sensitivity")
    manifest = {
        "status": "complete",
        "session": info["session"],
        "code_commit": commit,
        "git_dirty": False,
        "command_line": sys.argv,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_transport_dir": str(source.resolve()),
        "source_transport_manifest_sha256": file_sha256(source / "manifest.json"),
        "bank_dir": str(bank.resolve()),
        "bank_manifest_sha256": file_sha256(bank / "manifest.json"),
        "n_anchors": 40,
        "n_draws": 20,
        "n_splits": 5,
        "n_coverage_repeats": 5,
        "n_sequence_nulls": 499,
        "sequence_rows": n_scores,
        "content_rows_reconstructed_unchanged": 8000,
        "same_maps_paths_counts_subsets_and_null_seeds": True,
        "opportunities_and_evaluation_readouts_unchanged": True,
        "decoder_likelihood": "conditional_population_identity_given_count",
        "real_data_rescored": False,
        "real_fractions_corrected": False,
        "thresholds_changed": False,
        "elapsed_s": time.monotonic() - started,
        "output_sha256": hashes,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-transport-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    run(a.source_transport_dir, a.output_dir)
