#!/usr/bin/env python3
"""Independent reconstruction of the frozen sampling-remedy experiment."""
import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def same(a, b, label):
    if not np.allclose(np.asarray(a, float), np.asarray(b, float), atol=1e-8, rtol=1e-8, equal_nan=True):
        raise AssertionError(label)


def seed(*parts):
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "little")


def tiles(xy, grid):
    tile = np.clip(((xy-grid.min(0))/np.maximum(grid.max(0)-grid.min(0), 1)*3).astype(int), 0, 2)
    return tile[:, 0]*3+tile[:, 1]


def posterior(counts, rates, grid):
    score = np.zeros((len(counts), len(grid)))
    for j in range(len(rates)):
        score += counts[:, j, None]*np.log(rates[j])[None, :]-.02*rates[j][None, :]
    p = np.exp(score-logsumexp(score, axis=1)[:, None])
    mean = np.einsum("ij,jk->ik", p, grid)
    width = np.sqrt(np.maximum(np.sum(p*np.sum(grid**2, axis=1), axis=1)-np.sum(mean**2, axis=1), 0))
    entropy = -np.sum(p*np.log(np.maximum(p, 1e-300)), axis=1)/np.log(len(grid))
    region = np.stack([p[:, tiles(grid, grid) == k].sum(axis=1) for k in range(9)], axis=1)
    return dict(mean=mean, width=width, entropy=entropy, region=region,
                map_region=tiles(grid, grid)[p.argmax(axis=1)])


def reconstruct_selection(rates, grid, counts, truth, random_seed):
    order = np.random.default_rng(random_seed).permutation(len(rates))
    half = len(rates)//2
    universe = sorted(order[:2*half].tolist())
    expected_random = dict(a=sorted(order[:half].tolist()), b=sorted(order[half:2*half].tolist()))
    normed = rates[universe]/np.linalg.norm(rates[universe], axis=1)[:, None]
    peak = grid[rates[universe].argmax(axis=1)]
    distance = np.square(normed[:, None]-normed[None, :]).sum(axis=2)
    distance += np.square((peak[:, None]-peak[None, :])/40).sum(axis=2)
    rng = np.random.default_rng(random_seed)
    remaining, pairs = list(range(len(universe))), []
    while remaining:
        first = int(rng.choice(remaining))
        remaining.remove(first)
        second = min(remaining, key=lambda j: (distance[first, j], j))
        remaining.remove(second)
        pairs.append((universe[first], universe[second]))
    proposals = []
    tile = tiles(grid, grid)
    for i in range(128):
        swaps = rng.integers(0, 2, len(pairs))
        a = sorted(pair[swap] for pair, swap in zip(pairs, swaps, strict=True))
        b = sorted(pair[1-swap] for pair, swap in zip(pairs, swaps, strict=True))
        profiles = []
        for indices in (a, b):
            profile = np.array([rates[indices][:, tile == k].mean() if (tile == k).any() else 0 for k in range(9)])
            profiles.append(profile/profile.sum())
        proposals.append(dict(a=a, b=b, candidate_id=i, profile_loss=float(np.square(profiles[0]-profiles[1]).sum())))
    shortlist = sorted(proposals, key=lambda p: (p["profile_loss"], p["candidate_id"]))[:8]
    region = tiles(truth, grid)
    for candidate in shortlist:
        errors = []
        for side in ("a", "b"):
            ix = candidate[side]
            errors.append(np.linalg.norm(posterior(counts[:, ix], rates[ix], grid)["mean"]-truth, axis=1))
        local = [abs(errors[0][region == k].mean()-errors[1][region == k].mean()) for k in range(9) if (region == k).sum() >= 10]
        candidate["run_loss"] = float(max(e.mean() for e in errors)+np.mean(local))
    best = min(shortlist, key=lambda p: (p["run_loss"], p["profile_loss"], p["candidate_id"]))
    return expected_random, best, universe


def reconstruct_generator(actual, rates, drift, universe, grid, random_seed, source):
    rng = np.random.default_rng(random_seed)
    states = rng.integers(len(grid), size=len(actual))
    generator = rates.copy() if source == "sim_matched" else drift*rng.lognormal(0, .4, (len(rates), 1))
    output = np.zeros_like(actual)
    total = actual[:, universe].sum(axis=1)
    for i, state in enumerate(states):
        weights = generator[universe, state]
        output[i, universe] = rng.multinomial(int(total[i]), weights/weights.sum())
    assert np.array_equal(output.sum(axis=1), total)
    return output, grid[states]


def check_session(record, development_time):
    folder = Path(record["artifact_dir"])
    frozen = json.loads((folder/"selected_before_test.json").read_text())
    assert sha(folder/"selected_before_test.json") == record["frozen_sha256"]
    assert sha(frozen["cache_path"]) == frozen["cache_sha256"]
    if record["dataset"] == "blackstad_moser":
        assert development_time < frozen["created_at_utc"]
    with np.load(frozen["cache_path"]) as source:
        raw = {k: source[k] for k in source.files}
    with np.load(folder/"audit_arrays.npz") as source:
        data = {k: source[k] for k in source.files}
    mask = raw["unit_qc_mask"].astype(bool)
    support = raw["valid_spatial_bins"] & (raw["occupancy_first_half_s"] >= .05)
    ids, grid, rates = data["cell_ids"], data["grid_cm"], data["rates_hz"]
    same(ids, raw["cell_ids"][mask], "unit alignment")
    same(grid, raw["bin_centers_cm"][support], "grid alignment")
    same(rates, np.maximum(raw["rates_first_half_hz"][mask][:, support], 1e-4), "first-half rates")
    same(data["drift_hz"], np.maximum(raw["rates_second_half_hz"][mask][:, support], 1e-4), "drift rates")
    quarter = np.linspace(raw["supported_run_intervals"][:, 0].min(), raw["supported_run_intervals"][:, 1].max(), 5)
    for block, lower, upper in (("calibration", quarter[2], quarter[3]), ("test", quarter[3], quarter[4])):
        windows = data[f"{block}_windows"]
        assert len(windows) >= 100 and (windows[:, 0] >= lower).all() and (windows[:, 1] <= upper).all()
        same(windows[:, 1]-windows[:, 0], .02, "20 ms RUN duration")
        expected_xy = np.stack([np.interp(windows[:, 0]+.01, raw["position"][:, 0], raw["position"][:, axis]) for axis in (1, 2)], axis=1)
        same(windows[:, 2:], expected_xy, "RUN known position")
    raw_events = {int(value): i for i, value in enumerate(raw["candidate_event_indices"])}
    assert len(data["event_indices"]) == record["candidates"]
    for i, event in enumerate(data["event_indices"]):
        index = raw_events[int(event)]
        left, right = raw["candidate_offsets"][index:index+2]
        full = np.flatnonzero(np.isclose(raw["candidate_base_durations_s"][left:right], .005, atol=1e-9, rtol=0))
        assert len(full) >= 4 and np.array_equal(full, np.arange(len(full)))
        bins = left+full[-4:]
        same(data["endpoint_counts"][i], raw["candidate_base_counts"][bins][:, mask].sum(axis=0), "endpoint base counts")
        same(data["endpoint_start_s"][i], raw["candidate_base_starts_s"][bins[0]], "endpoint start")
        same(data["endpoint_end_s"][i], raw["candidate_base_starts_s"][bins[-1]]+.005, "endpoint end")
    excluded = json.loads((folder/"excluded_endpoints.json").read_text())
    assert set(data["event_indices"]) | {e["event_index"] for e in excluded} == set(raw_events)
    for j, cell in enumerate(ids):
        spikes = np.sort(raw["spikes"][raw["spikes"][:, 1] == cell, 0])
        for block in ("calibration", "test", "endpoint"):
            if block == "endpoint":
                start, end = data["endpoint_start_s"], data["endpoint_end_s"]
            else:
                start, end = data[f"{block}_windows"][:, :2].T
            expected = np.searchsorted(spikes, end)-np.searchsorted(spikes, start)
            same(data[f"{block}_counts"][:, j], expected, "raw spike reconstruction")
    assert sha(folder/"event_readouts.csv.gz") == record["readouts_sha256"]
    frame = pd.read_csv(folder/"event_readouts.csv.gz")
    assert not frame.duplicated(["source", "split", "condition", "draw", "event_index"]).any()
    checked = 0
    for part in frozen["parts"]:
        reconstructed = reconstruct_selection(rates, grid, data["calibration_counts"], data["calibration_windows"][:, 2:],
            seed(frozen["seed"], record["dataset"], record["animal"], record["session"], part["split"]))
        for name, expected in zip(("random", "balanced"), reconstructed[:2], strict=True):
            assert part[name]["a"] == expected["a"] and part[name]["b"] == expected["b"]
        assert part["universe"] == reconstructed[2]
        cases = [("real", -1, data["endpoint_counts"], None, data["event_indices"]),
                 ("run_test", -1, data["test_counts"], data["test_windows"][:, 2:], np.arange(len(data["test_counts"])))]
        for source in ("sim_matched", "sim_drift"):
            for draw in range(2):
                counts, truth = reconstruct_generator(data["endpoint_counts"], rates, data["drift_hz"], part["universe"], grid,
                    seed(frozen["seed"], record["dataset"], record["animal"], record["session"], part["split"], source, draw), source)
                cases.append((source, draw, counts, truth, data["event_indices"]))
        for condition in ("random", "balanced"):
            a, b = part[condition]["a"], part[condition]["b"]
            assert len(a) == len(b) >= 5 and not set(a) & set(b)
            for source, draw, counts, truth, events in cases:
                rows = frame.loc[frame.source.eq(source) & frame.draw.eq(draw) & frame.split.eq(part["split"]) & frame.condition.eq(condition)]
                assert len(rows) == len(events)
                rows = rows.set_index("event_index").loc[events]
                outputs = []
                for side, indices in (("a", a), ("b", b)):
                    result = posterior(counts[:, indices], rates[indices], grid)
                    outputs.append(result)
                    same(result["mean"], rows[[f"{side}_x_cm", f"{side}_y_cm"]], "posterior mean")
                    same(result["width"], rows[f"{side}_width_cm"], "width")
                    same(result["entropy"], rows[f"{side}_entropy"], "entropy")
                    same(result["region"], rows[[f"{side}_region_{j}" for j in range(9)]], "regional posterior")
                    same(counts[:, indices].sum(1), rows[f"{side}_spikes"], "spikes")
                    same((counts[:, indices] > 0).sum(1), rows[f"{side}_active"], "active units")
                    if truth is not None:
                        same(np.linalg.norm(result["mean"]-truth, axis=1), rows[f"{side}_truth_error_cm"], "known-truth error")
                    else:
                        assert rows[f"{side}_truth_error_cm"].isna().all()
                same(np.linalg.norm(outputs[0]["mean"]-outputs[1]["mean"], axis=1), rows.endpoint_separation_cm, "separation")
                same(np.abs(outputs[0]["region"]-outputs[1]["region"]).sum(1)/2, rows.regional_tv, "regional TV")
                same(outputs[0]["map_region"] == outputs[1]["map_region"], rows.map_region_agreement, "MAP tile")
                same((outputs[0]["entropy"]+outputs[1]["entropy"])/2, rows.pair_entropy, "paired entropy")
                same((rows.a_truth_error_cm+rows.b_truth_error_cm)/2, rows.pair_truth_error_cm, "paired truth error")
                checked += len(rows)
    return frame, dict(dataset=record["dataset"], animal=record["animal"], session=record["session"],
                       candidate_endpoints=len(data["event_indices"]), readouts_reconstructed=checked,
                       run_windows=len(data["calibration_windows"])+len(data["test_windows"]))


def check_aggregate_tables(data, root):
    for name, extra in (("summary", []), ("by_animal", ["animal"]), ("by_session", ["animal", "session"])):
        table = pd.read_csv(root/f"{name}.csv", dtype={"animal": str, "session": str})
        keys = ["dataset", "source", "split", "condition"]+extra
        metrics = [c for c in table if c not in keys]
        assert not table.duplicated(keys).any()
        for group, expected in table.set_index(keys).iterrows():
            mask = np.logical_and.reduce([data[key].astype(str).eq(str(value)) for key, value in zip(keys, group, strict=True)])
            selected = data.loc[mask]
            assert not selected.empty
            animal_means = []
            for _, animal in selected.groupby("animal"):
                sessions = []
                for _, session in animal.groupby("session"):
                    draws = [draw[metrics].mean().to_numpy() for _, draw in session.groupby("draw")]
                    sessions.append(np.mean(draws, axis=0))
                animal_means.append(np.mean(sessions, axis=0))
            same(np.mean(animal_means, axis=0), expected, f"{name}: equal-draw/session/animal aggregation")


def check_gates(root, manifest):
    table = pd.read_csv(root/"summary.csv")
    animal = pd.read_csv(root/"by_animal.csv")
    external = table.loc[table.dataset.eq("blackstad_moser") & table.split.eq(0)]
    records = [r for r in manifest["results"] if r["dataset"] == "blackstad_moser"]
    retained = [r for r in records if r["status"] == "complete"]
    total = sum(r["source_candidates"] for r in records)
    expected = dict(inputs_unchanged=manifest["inputs_unchanged"], external_coverage=(
        len({r["animal"] for r in retained}) >= 4 and total > 0 and sum(r["candidates"] for r in retained)/total >= .8))
    if external.empty:
        expected["external_results_present"] = False
    else:
        real = external.loc[external.source.eq("real")].set_index("condition")
        for metric in ("regional_tv", "endpoint_separation_cm"):
            r, b = real.loc["random", metric], real.loc["balanced", metric]
            expected[f"{metric}_reduction"] = r > 0 and (r-b)/r >= .1
            x = animal.loc[animal.dataset.eq("blackstad_moser") & animal.source.eq("real") & animal.split.eq(0)]
            x = x.pivot(index="animal", columns="condition", values=metric)
            delta = (x.random-x.balanced).to_numpy()
            expected[f"{metric}_animals_positive"] = np.count_nonzero(delta > 0) >= 4
            resamples = np.random.default_rng(20260914).choice(delta, (5000, len(delta)), replace=True).mean(axis=1)
            expected[f"{metric}_bootstrap"] = np.quantile(resamples, .025) > 0
        for metric, slack in (("pair_entropy", 0), ("a_entropy", .01), ("b_entropy", .01)):
            expected[f"{metric}_not_worse"] = real.loc["balanced", metric]-real.loc["random", metric] <= slack
        for source in ("run_test", "sim_matched", "sim_drift"):
            truth = external.loc[external.source.eq(source)].set_index("condition")
            for metric, slack in (("pair_truth_error_cm", 0), ("a_truth_error_cm", 2), ("b_truth_error_cm", 2)):
                expected[f"{source}_{metric}"] = truth.loc["balanced", metric]-truth.loc["random", metric] <= slack
    expected["statistical_validation"] = all(expected.values())
    gates = pd.read_csv(root/"gates.csv").set_index("gate")
    assert gates.passed.dtype == bool and set(gates.index) == set(expected)
    for name, value in expected.items():
        assert gates.loc[name, "passed"] == value, name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.result_dir
    manifest = json.loads((root/"manifest.json").read_text())
    assert manifest["status"] == "complete" and manifest["inputs_unchanged"]
    for key, path in manifest["input_file_paths"].items():
        assert sha(path) == manifest["input_file_sha256"][key]
    development = json.loads((root/"development_finished_before_external.json").read_text())
    assert development["frozen_code"] == manifest["input_file_sha256"]["code"]
    assert development["frozen_protocol"] == manifest["input_file_sha256"]["protocol"]
    frames, records = [], []
    for rec in manifest["results"]:
        if rec["status"] != "complete":
            print("retained failure:", rec, flush=True)
            continue
        frame, checked = check_session(rec, development["created_at_utc"])
        frames.append(frame)
        records.append(checked)
        print(json.dumps(checked), flush=True)
    data = pd.concat(frames, ignore_index=True)
    check_aggregate_tables(data, root)
    check_gates(root, manifest)
    pd.DataFrame(records).to_csv(root/"independent_reconstruction.csv", index=False)
    report = dict(passed=True, sessions=len(records), source_endpoints=sum(r["candidate_endpoints"] for r in records),
                  reconstructed_readouts=sum(r["readouts_reconstructed"] for r in records),
                  reconstructs="all assignments, counts, simulation draws, posteriors and aggregate outcomes",
                  limitations="native spike sorting and biological replay truth unavailable; source encoding cache shared",
                  verified_table_sha256={name: sha(root/f"{name}.csv") for name in ("summary", "by_animal", "by_session", "gates")},
                  script_sha256=sha(__file__), producer_manifest_sha256=sha(root/"manifest.json"),
                  created_at_utc=datetime.now(UTC).isoformat())
    (root/"independent_audit.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"AUDIT FAILED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise
