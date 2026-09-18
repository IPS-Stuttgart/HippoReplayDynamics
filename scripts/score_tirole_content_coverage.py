"""Score a frozen split/repeat without redetecting events or changing evaluation cells."""

import argparse
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hipporeplayimm.tirole_two_track import file_sha256
from hipporeplayimm.two_track_content import classify_sequence, content_readout, nested_subsets, stable_seed


def validate_bank(bank):
    manifest = json.loads((bank / "manifest.json").read_text())
    for name, digest in manifest["outputs_sha256"].items():
        if file_sha256(bank / name) != digest:
            raise ValueError("bank checksum mismatch: " + name)
    parts = json.loads((bank / "partitions.json").read_text())
    detector = set(parts["detector"])
    for split in parts["splits"]:
        a, b = set(split["inference"]), set(split["evaluation"])
        if not a or not b or a & b or a & detector or b & detector or len(a) != len(split["inference"]) or len(b) != len(split["evaluation"]):
            raise ValueError("cell-role overlap, duplicate identity or empty population")
    data = np.load(bank / "event_counts.npz")
    counts, offsets = data["counts"], data["offsets"]
    if counts.ndim != 2 or (counts < 0).any() or not np.equal(counts, np.floor(counts)).all():
        raise ValueError("invalid event count array")
    if offsets.ndim != 1 or offsets[0] != 0 or offsets[-1] != len(counts) or (np.diff(offsets) <= 0).any():
        raise ValueError("invalid event offsets")
    events = pd.read_csv(bank / "candidate_events.csv")
    if len(events) + 1 != len(offsets) or not np.array_equal(events.event_id, np.arange(len(events))):
        raise ValueError("event identity/offset mismatch")
    maps = np.load(bank / "RUN_maps.npz")
    if not np.array_equal(data["unit_ids"], maps["unit_ids"]):
        raise ValueError("map/count unit mismatch")
    if maps["rates"].shape[1] != counts.shape[1]:
        raise ValueError("map/count dimension mismatch")
    n = counts.shape[1]
    if any(i < 0 or i >= n for i in detector):
        raise ValueError("invalid detector identity")
    for split in parts["splits"]:
        if any(i < 0 or i >= n for i in split["inference"] + split["evaluation"]):
            raise ValueError("invalid inference/evaluation identity")
        if set(split["inference"]) | set(split["evaluation"]) | detector != set(maps["common_units"]):
            raise ValueError("partition differs from common RUN units")
    return manifest, parts, events, counts, offsets, maps


def chosen_events(events, session, stratum, limit):
    eligible = events[events.primary_ripple_candidate] if stratum == "ripple" else events[~events.primary_ripple_candidate & events.ripple_supported]
    if limit is None:
        return eligible
    if limit < 1:
        raise ValueError("positive cap required")
    selected = []
    for _, group in eligible.groupby("epoch"):
        order = sorted(group.event_id.tolist(), key=lambda e: stable_seed(20260918, session, e, "technical-cap"))
        selected.extend(order[:limit])
    return eligible[eligible.event_id.isin(selected)]


def run(bank, output, split_index, repeat, stratum="ripple", limit=None):
    if output.exists() and any(output.iterdir()):
        raise ValueError("new output directory required")
    info, parts, events, counts, offsets, maps = validate_bank(bank)
    session = info["session"]
    config_path = Path(__file__).resolve().parents[1] / "docs/two_track_content_configuration.json"
    config = json.loads(config_path.read_text())
    if not 0 <= split_index < config["cell_splits"] or not 0 <= repeat < config["subsample_repeats"]:
        raise ValueError("split/repeat outside frozen design")
    if parts["seed"] != config["seed"]:
        raise ValueError("seed differs from frozen protocol")
    chosen = chosen_events(events, session, stratum, limit)
    if chosen.empty:
        raise ValueError("zero selected events; no vacuous technical pass")
    selected = parts["splits"][split_index]
    inference = np.array(selected["inference"], int)
    evaluation = np.array(selected["evaluation"], int)
    subsets = nested_subsets(inference, session, split_index, repeat, config["seed"])
    rates, valid, centers = maps["rates"], maps["valid_bins"], maps["bin_centers_cm"]
    k = config["n_sequence_shuffles"]
    rows = []
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    for count, event in enumerate(chosen.to_dict("records"), 1):
        eid = int(event["event_id"])
        c = counts[offsets[eid] : offsets[eid + 1]]
        rng = np.random.default_rng(stable_seed(config["seed"], session, eid, split_index, "sequence-nulls"))
        shifts = rng.integers(0, len(centers), (k, 2, counts.shape[1]))
        permutations = np.argsort(rng.random((k, len(c))), axis=1)
        identity_permutation = rng.permutation(inference)
        evalrng = np.random.default_rng(stable_seed(config["seed"], session, eid, split_index, "evaluation-context"))
        swaps = evalrng.integers(0, 2, (config["content_null_count"], len(evaluation))).astype(bool)
        context = content_readout(c[:, evaluation], rates[:, evaluation], valid, swaps)
        conditional = content_readout(c[:, evaluation], rates[:, evaluation], valid, swaps, True)
        for fraction, units in subsets.items():
            for arm in ["real"] + (["cell_identity_randomized"] if fraction in {1.0, 0.5} else []):
                cc = c[:, units] if arm == "real" else c[:, identity_permutation[np.searchsorted(inference, units)]]
                before = time.monotonic()
                result = classify_sequence(cc, rates[:, units], valid, centers, shifts[:, :, units], permutations, config["alpha_per_track_per_null"])
                rows.append(
                    {
                        **event,
                        "session": session,
                        "animal": info["animal"],
                        "cohort_stratum": info["cohort_stratum"],
                        "split": split_index,
                        "repeat": repeat,
                        "fraction": fraction,
                        "arm": arm,
                        "n_inference_units": len(units),
                        "n_evaluation_units": len(evaluation),
                        "n_inference_spikes": int(cc.sum()),
                        "n_evaluation_active_units": int((c[:, evaluation].sum(axis=0) > 0).sum()),
                        **result,
                        **{"evaluation_" + key: value for key, value in context.items()},
                        **{"conditional_evaluation_" + key: value for key, value in conditional.items()},
                        "scoring_runtime_s": time.monotonic() - before,
                        "status": "complete",
                    }
                )
        if count % 10 == 0 or count == len(chosen):
            state = {"status": "running", "completed_events": count, "selected_events": len(chosen), "elapsed_s": time.monotonic() - started}
            (output / "progress.json").write_text(json.dumps(state) + "\n")
            print(json.dumps(state), flush=True)
    table = pd.DataFrame(rows)
    expected = len(chosen) * 6
    if len(table) != expected or table.duplicated(["event_id", "fraction", "arm"]).any():
        raise ValueError("incomplete or duplicate score rows")
    table.to_csv(output / "event_content_coverage_scores.csv", index=False)
    repo = Path(__file__).resolve().parents[1]
    git = lambda *args: subprocess.check_output(["git", *args], cwd=repo, text=True).strip()
    manifest = {
        "status": "complete",
        "session": session,
        "split": split_index,
        "repeat": repeat,
        "candidate_stratum": stratum,
        "technical_pilot_only": limit is not None,
        "selected_events": len(chosen),
        "score_rows": len(table),
        "expected_score_rows": expected,
        "selected_event_ids": chosen.event_id.tolist(),
        "bank_manifest_sha256": file_sha256(bank / "manifest.json"),
        "configuration_sha256": file_sha256(config_path),
        "code_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "command_line": sys.argv,
        "elapsed_s": time.monotonic() - started,
        "output_sha256": file_sha256(output / "event_content_coverage_scores.csv"),
        "biological_claim_gate": False,
        "static_context_readout_not_temporal_prediction": True,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "progress.json").write_text(json.dumps({"status": "complete", "selected_events": len(chosen)}) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bank-dir", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--split", required=True, type=int)
    p.add_argument("--repeat", required=True, type=int)
    p.add_argument("--candidate-stratum", choices=["ripple", "mua_only"], default="ripple")
    p.add_argument("--max-events-per-epoch", type=int)
    a = p.parse_args()
    run(a.bank_dir, a.output_dir, a.split, a.repeat, a.candidate_stratum, a.max_events_per_epoch)
