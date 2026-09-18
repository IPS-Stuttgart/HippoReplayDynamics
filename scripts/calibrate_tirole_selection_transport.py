"""Known-label selection transport with fixed paths/count anchors, never real rescoring."""

import argparse
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from hipporeplayimm.tirole_two_track import file_sha256
from hipporeplayimm.two_track_content import classify_sequence, content_readout, nested_subsets, stable_seed
from scripts.calibrate_tirole_content_measurement import generate_track_counts, ordered_path
from scripts.score_tirole_content_coverage import validate_bank

N_DRAWS = 20
N_COVERAGE_REPEATS = 5
N_NULLS = 499


def simulate_anchor(info, parts, event, original, maps, n_draws=N_DRAWS, n_repeats=N_COVERAGE_REPEATS):
    """Full scores are reused across coverage repeats; evaluation never selects an event."""
    session = info["session"]
    eid = int(event["event_id"])
    rates, valid, centers = maps["rates"], maps["valid_bins"], maps["bin_centers_cm"]
    scores, contents = [], []
    metadata = {"session": session, "animal": info["animal"], "anchor_event": eid, "epoch": event["epoch"], "ripple_positive": bool(event["primary_ripple_candidate"])}
    for split, part in enumerate(parts["splits"]):
        train, held = np.asarray(part["inference"]), np.asarray(part["evaluation"])
        if set(train) & set(held):
            raise ValueError("inference/evaluation cell overlap")
        path = ordered_path(valid, len(original), np.random.default_rng(stable_seed(20260918, session, eid, split, "known-path")))
        span = float(abs(centers[path[-1]] - centers[path[0]]))
        halves = [nested_subsets(train, session, split, r)[0.5] for r in range(n_repeats)]
        for draw in range(n_draws):
            for track in range(2):
                base = {**metadata, "split": split, "draw": draw, "truth_track": track + 1, "true_path_span_cm": span}
                rng = np.random.default_rng(stable_seed(20260918, session, eid, split, draw, track, "selection-transport-spikes-v1"))
                generated = generate_track_counts(original, rates, train, held, path, track, rng)
                swaps = rng.integers(0, 2, (199, len(held))).astype(bool)
                values = {}
                for name, conditional in [("poisson", False), ("conditional_count", True)]:
                    c = content_readout(generated[:, held], rates[:, held], valid, swaps, conditional)
                    values[name + "_track2_probability"] = c["track2_probability"]
                    values[name + "_true_signed_z"] = (1 - 2 * track) * c["z_log_odds"]
                contents.append({**base, "n_evaluation_spikes": int(generated[:, held].sum()), **values})
                permutation = rng.permutation(len(generated))
                for generator, counts in [("ordered", generated), ("whole_bin_shuffled", generated[permutation])]:
                    nrng = np.random.default_rng(stable_seed(20260918, session, eid, split, draw, track, generator, "selection-transport-nulls-v1"))
                    shifts = nrng.integers(0, len(centers), (N_NULLS, 2, original.shape[1]))
                    null_permutations = np.argsort(nrng.random((N_NULLS, len(counts))), axis=1)
                    for repeat, ids in [(-1, train), *enumerate(halves)]:
                        result = classify_sequence(counts[:, ids], rates[:, ids], valid, centers, shifts[:, :, ids], null_permutations)
                        scores.append(
                            {
                                **base,
                                "generator": generator,
                                "repeat": repeat,
                                "fraction": 1.0 if repeat == -1 else 0.5,
                                "n_inference_cells": len(ids),
                                "n_inference_spikes": int(counts[:, ids].sum()),
                                "n_time_bins": len(counts),
                                **result,
                            }
                        )
    return pd.DataFrame(scores), pd.DataFrame(contents)


def run(bank, calibration, output):
    if output.exists():
        raise ValueError("new immutable transport experiment required")
    git = lambda *args: subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    commit = git("rev-parse", "HEAD")
    if git("status", "--porcelain"):
        raise ValueError("freeze simulation protocol before execution")
    info, parts, events, counts, offsets, maps = validate_bank(bank)
    if info["cohort_stratum"] != "strict_RUN_pass":
        raise ValueError("frozen primary sessions only")
    m = json.loads((calibration / "manifest.json").read_text())
    if m["status"] != "complete" or m["git_dirty"] or m["bank_manifest_sha256"] != file_sha256(bank / "manifest.json"):
        raise ValueError("mismatched source calibration")
    for name, digest in m["output_sha256"].items():
        if file_sha256(calibration / name) != digest:
            raise ValueError("changed original calibration")
    anchors = pd.read_csv(calibration / "count_anchors.csv").sort_values("event_id")
    if len(anchors) != 40 or anchors.event_id.duplicated().any():
        raise ValueError("unchanged forty-anchor test bank required")
    pd.testing.assert_frame_equal(events.set_index("event_id").loc[anchors.event_id].reset_index(), anchors.reset_index(drop=True), check_exact=False)
    output.mkdir(parents=True)
    (output / "sequence_shards").mkdir()
    (output / "content_shards").mkdir()
    anchors.to_csv(output / "count_anchors.csv", index=False)
    start = time.monotonic()
    n_scores, n_contents = 0, 0
    hashes = {"count_anchors.csv": file_sha256(output / "count_anchors.csv")}
    for rank, event in enumerate(anchors.to_dict("records")):
        eid = int(event["event_id"])
        original = counts[offsets[eid] : offsets[eid + 1]]
        scores, content = simulate_anchor(info, parts, event, original, maps)
        if len(scores) != 5 * N_DRAWS * 2 * 2 * (1 + N_COVERAGE_REPEATS) or len(content) != 5 * N_DRAWS * 2:
            raise ValueError("incomplete simulation outputs")
        if scores.duplicated(["split", "draw", "truth_track", "generator", "repeat"]).any():
            raise ValueError("duplicate sequence outputs")
        for name, data in [("sequence_shards", scores), ("content_shards", content)]:
            path = output / name / f"{eid}.csv"
            data.to_csv(path, index=False)
            hashes[str(path.relative_to(output))] = file_sha256(path)
        n_scores += len(scores)
        n_contents += len(content)
        print(json.dumps({"session": info["session"], "completed_anchors": rank + 1, "expected_anchors": len(anchors), "elapsed_s": time.monotonic() - start}), flush=True)
    if git("status", "--porcelain") or git("rev-parse", "HEAD") != commit:
        raise ValueError("code changed during experiment")
    manifest = {
        "status": "complete",
        "session": info["session"],
        "code_commit": commit,
        "git_dirty": False,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "command_line": sys.argv,
        "bank_dir": str(bank.resolve()),
        "bank_manifest_sha256": file_sha256(bank / "manifest.json"),
        "calibration_dir": str(calibration.resolve()),
        "calibration_manifest_sha256": file_sha256(calibration / "manifest.json"),
        "n_anchors": len(anchors),
        "n_draws": N_DRAWS,
        "n_splits": 5,
        "n_coverage_repeats": N_COVERAGE_REPEATS,
        "n_sequence_nulls": N_NULLS,
        "sequence_rows": n_scores,
        "content_rows": n_contents,
        "paired_tracks_share_path_and_counts": True,
        "paths_unchanged_from_original_calibration": True,
        "coverage_subsets_fixed_across_anchors": True,
        "real_data_rescored": False,
        "real_experience_fractions_corrected": False,
        "biological_replication": False,
        "known_map_not_known_trajectory": True,
        "counts_conditioned_on_recorded_totals": True,
        "elapsed_s": time.monotonic() - start,
        "output_sha256": hashes,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bank-dir", type=Path, required=True)
    p.add_argument("--calibration-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    run(a.bank_dir, a.calibration_dir, a.output_dir)
