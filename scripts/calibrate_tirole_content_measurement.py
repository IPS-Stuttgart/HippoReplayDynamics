"""Known-map/known-generator calibration at observed counts; no replay truth is asserted."""

import argparse
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from hipporeplayimm.lagged_neural_prediction import NeuralOperator
from hipporeplayimm.occupancy_matched_forecast import OccupancyMatchedNull
from hipporeplayimm.tirole_two_track import file_sha256
from hipporeplayimm.two_track_content import classify_sequence, content_readout, nested_subsets, stable_seed
from scripts.forecast_tirole_content_coverage import forecast_event, relative_ids
from scripts.score_tirole_content_coverage import validate_bank

ANCHORS_PER_EPOCH = 20
PREDICTION_REPEATS = 5


def anchors(events, session):
    choices = []
    for _, g in events[events.ripple_supported].groupby("epoch"):
        order = sorted(g.event_id.tolist(), key=lambda eid: stable_seed(20260918, session, eid, "measurement-calibration-anchor"))
        choices.extend(order[:ANCHORS_PER_EPOCH])
    if not choices:
        raise ValueError("no supported count anchors")
    return events[events.event_id.isin(choices)].sort_values("event_id")


def draw_counts(totals, weights, rng):
    totals = np.asarray(totals)
    weights = np.asarray(weights, float)
    if (
        totals.ndim != 1
        or weights.ndim != 2
        or len(totals) != len(weights)
        or (totals < 0).any()
        or not np.equal(totals, np.floor(totals)).all()
        or not np.isfinite(weights).all()
        or (weights <= 0).any()
    ):
        raise ValueError("nonnegative integer totals and positive time-by-cell weights required")
    return np.array([rng.multinomial(int(n), w / w.sum()) for n, w in zip(totals, weights, strict=True)])


def ordered_path(valid, n, rng):
    common = np.asarray(valid, bool).all(axis=0)
    edges = np.diff(np.r_[False, common, False].astype(int))
    intervals = list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True))
    if not intervals:
        raise ValueError("no common supported track interval")
    lo, hi = max(intervals, key=lambda ab: ab[1] - ab[0])
    if hi - lo < 4 or n < 5:
        raise ValueError("insufficient known-path support")
    span = int(rng.integers(3, hi - lo))
    start = int(rng.integers(lo, hi - span))
    path = np.rint(np.linspace(start, start + span, n)).astype(int)
    return path if rng.random() < 0.5 else path[::-1]


def generate_track_counts(original, rates, train, held, path, track, rng):
    out = np.zeros_like(original)
    for ids in [train, held]:
        weights = np.maximum(rates[track][:, path].T[:, ids], 1e-4)
        out[:, ids] = draw_counts(original[:, ids].sum(axis=1), weights, rng)
    return out


def generate_state_counts(original, emissions, train, held, initial, transition, rng):
    states = []
    state = int(rng.choice(len(initial), p=initial))
    for t in range(len(original)):
        if t:
            state = int(rng.choice(len(initial), p=transition[state]))
        states.append(state)
    out = np.zeros_like(original)
    for ids in [train, held]:
        out[:, ids] = draw_counts(original[:, ids].sum(axis=1), emissions[ids][:, states].T, rng)
    return out, np.array(states)


def calibrate_content(info, parts, selected, counts, offsets, maps):
    records = []
    session = info["session"]
    rates = maps["rates"]
    valid = maps["valid_bins"]
    centers = maps["bin_centers_cm"]
    for rank, event in enumerate(selected.to_dict("records")):
        eid = int(event["event_id"])
        original = counts[offsets[eid] : offsets[eid + 1]]
        for split, part in enumerate(parts["splits"]):
            train = np.array(part["inference"])
            held = np.array(part["evaluation"])
            repeat = (rank + split) % 5
            subsets = nested_subsets(train, session, split, repeat)
            path = ordered_path(valid, len(original), np.random.default_rng(stable_seed(20260918, session, eid, split, "known-path")))
            for track in range(2):
                rng = np.random.default_rng(stable_seed(20260918, session, eid, split, track, "known-track-spikes"))
                generated = generate_track_counts(original, rates, train, held, path, track, rng)
                swaps = rng.integers(0, 2, (199, len(held))).astype(bool)
                context = content_readout(generated[:, held], rates[:, held], valid, swaps)
                conditional = content_readout(generated[:, held], rates[:, held], valid, swaps, True)
                shuffled = generated[rng.permutation(len(generated))]
                for generator, c in [("ordered", generated), ("whole_bin_shuffled", shuffled)]:
                    null_rng = np.random.default_rng(stable_seed(20260918, session, eid, split, track, generator, "sequence-nulls"))
                    shifts = null_rng.integers(0, len(centers), (499, 2, counts.shape[1]))
                    permutations = np.argsort(null_rng.random((499, len(c))), axis=1)
                    for fraction in [1.0, 0.5]:
                        ids = subsets[fraction]
                        result = classify_sequence(c[:, ids], rates[:, ids], valid, centers, shifts[:, :, ids], permutations)
                        sign = 1 if track == 0 else -1
                        records.append(
                            {
                                "session": session,
                                "animal": info["animal"],
                                "cohort_stratum": info["cohort_stratum"],
                                "anchor_event": eid,
                                "epoch": event["epoch"],
                                "anchor_ripple_positive": event["primary_ripple_candidate"],
                                "split": split,
                                "repeat": repeat,
                                "truth_track": track + 1,
                                "generator": generator,
                                "fraction": fraction,
                                "n_time_bins": len(c),
                                "n_inference_spikes": int(c[:, ids].sum()),
                                "n_evaluation_spikes": int(c[:, held].sum()),
                                **result,
                                "correct_inferred_track": result["inferred_track"] == track + 1,
                                "evaluation_true_signed_log_odds": sign * context["log_odds"],
                                "evaluation_true_signed_z": sign * context["z_log_odds"],
                                "conditional_true_signed_log_odds": sign * conditional["log_odds"],
                                "conditional_true_signed_z": sign * conditional["z_log_odds"],
                                "evaluation_track2_probability": context["track2_probability"],
                                "true_path_span_cm": float(abs(centers[path[-1]] - centers[path[0]])),
                            }
                        )
        print(json.dumps({"stage": "content", "session": session, "completed_anchors": rank + 1, "expected_anchors": len(selected)}), flush=True)
    result = pd.DataFrame(records)
    expected = len(selected) * 5 * 2 * 2 * 2
    if len(result) != expected or result.duplicated(["anchor_event", "split", "truth_track", "generator", "fraction"]).any():
        raise ValueError("incomplete content calibration")
    return result


def calibrate_prediction(info, parts, selected, counts, offsets, source):
    rows = []
    session = info["session"]
    fold_records = json.loads((source / "folds.json").read_text())
    selected_by_id = selected.set_index("event_id")
    for record in fold_records:
        fold = record["fold"]
        ids = sorted(set(record["test_ids"]) & set(selected.event_id))
        if not ids:
            continue
        z = np.load(source / f"fit_{fold}.npz")
        fit = SimpleNamespace(**{k: z[k] for k in ["probabilities", "initial", "transition", "occupancy", "global_probability"]})
        population = z["population"]
        op = NeuralOperator(fit.initial, fit.transition, fit.occupancy)
        matched = OccupancyMatchedNull.from_operator(op)
        null_matrix = matched.step(np.eye(len(fit.initial)))
        for eid in ids:
            original = counts[offsets[eid] : offsets[eid + 1]][:, population]
            event = selected_by_id.loc[eid]
            for split, part in enumerate(parts["splits"]):
                held = relative_ids(population, part["evaluation"])
                for repeat in range(PREDICTION_REPEATS):
                    half = nested_subsets(np.asarray(part["inference"]), session, split, repeat)[0.5]
                    train = relative_ids(population, half)
                    for generator, transition in [("dynamic", fit.transition), ("matched_null", null_matrix)]:
                        rng = np.random.default_rng(stable_seed(20260918, session, eid, split, repeat, "known-temporal-generator"))
                        c, _ = generate_state_counts(original, fit.probabilities, train, held, fit.initial, transition, rng)
                        _, scores = forecast_event(c, train, held, fit, op, matched)
                        for score in scores:
                            rows.append(
                                {
                                    "session": session,
                                    "animal": info["animal"],
                                    "cohort_stratum": info["cohort_stratum"],
                                    "anchor_event": eid,
                                    "epoch": event["epoch"],
                                    "anchor_ripple_positive": event["primary_ripple_candidate"],
                                    "fold": fold,
                                    "split": split,
                                    "repeat": repeat,
                                    "generator": generator,
                                    "horizon_ms": 20 * score["horizon"],
                                    **score,
                                }
                            )
        print(json.dumps({"stage": "prediction", "session": session, "completed_fold": fold}), flush=True)
    result = pd.DataFrame(rows)
    if len(result) != len(selected) * 5 * PREDICTION_REPEATS * 2 * 3:
        raise ValueError("incomplete temporal calibration")
    return result


def run(bank, source, output):
    if output.exists():
        raise ValueError("new calibration directory required")
    git = lambda *a: subprocess.check_output(["git", *a], cwd=ROOT, text=True).strip()
    commit = git("rev-parse", "HEAD")
    if git("status", "--porcelain"):
        raise ValueError("freeze calibration before execution")
    info, parts, events, counts, offsets, maps = validate_bank(bank)
    m = json.loads((source / "manifest.json").read_text())
    if m["status"] != "complete" or m["git_dirty"] or m["bank_manifest_sha256"] != file_sha256(bank / "manifest.json"):
        raise ValueError("invalid source forecast")
    for name, digest in m["output_sha256"].items():
        if file_sha256(source / name) != digest:
            raise ValueError("changed source fit")
    selected = anchors(events, info["session"])
    start = time.monotonic()
    output.mkdir(parents=True)
    selected.to_csv(output / "count_anchors.csv", index=False)
    content = calibrate_content(info, parts, selected, counts, offsets, maps)
    content.to_csv(output / "known_track_calibration.csv", index=False)
    prediction = calibrate_prediction(info, parts, selected, counts, offsets, source)
    prediction.to_csv(output / "known_temporal_calibration.csv", index=False)
    if git("status", "--porcelain") or git("rev-parse", "HEAD") != commit:
        raise ValueError("calibration code changed during run")
    manifest = {
        "status": "complete",
        "code_commit": commit,
        "git_dirty": False,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "command_line": sys.argv,
        "session": info["session"],
        "bank_manifest_sha256": file_sha256(bank / "manifest.json"),
        "source_forecast_manifest_sha256": file_sha256(source / "manifest.json"),
        "anchors": len(selected),
        "content_rows": len(content),
        "prediction_rows": len(prediction),
        "elapsed_s": time.monotonic() - start,
        "known_map_not_known_trajectory": True,
        "generating_parameters_given_to_decoder": True,
        "latent_paths_or_track_labels_given_to_decoder": False,
        "counts_conditioned_on_recorded_totals": True,
        "model_matched_best_case_not_replay_ground_truth": True,
        "output_sha256": {p.name: file_sha256(p) for p in output.iterdir()},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bank-dir", type=Path, required=True)
    p.add_argument("--forecast-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    run(a.bank_dir, a.forecast_dir, a.output_dir)
