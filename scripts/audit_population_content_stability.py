#!/usr/bin/env python3
"""Reconstruct counts, selected readouts and aggregate policy results independently."""
import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def same(a, b, name):
    if not np.allclose(np.asarray(a, float), np.asarray(b, float), atol=1e-8, rtol=1e-8, equal_nan=True):
        raise AssertionError(name)


def independent_posterior(counts, rates, grid):
    likelihood = np.zeros(len(grid))
    for count, rate in zip(counts, rates, strict=True):
        likelihood += count*np.log(rate)-.02*rate
    p = np.exp(likelihood-logsumexp(likelihood))
    mean = np.sum(p[:, None]*grid, axis=0)
    width = np.sqrt(np.sum(p*np.sum((grid-mean)**2, axis=1)))
    ent = -np.sum(p*np.log(np.maximum(p, 1e-300)))/np.log(len(p))
    ix = np.minimum(2, ((grid-grid.min(axis=0))/np.maximum(np.ptp(grid, axis=0), 1)*3).astype(int))
    tile = ix[:, 0]*3+ix[:, 1]
    region = np.bincount(tile, weights=p, minlength=9)
    return mean, width, ent, region


def check_session(record):
    folder = Path(record["artifact_dir"])
    frozen = json.loads((folder/"frozen_populations.json").read_text())
    if digest(frozen["cache_path"]) != frozen["cache_sha256"]:
        raise AssertionError("source cache changed")
    with np.load(frozen["cache_path"], allow_pickle=False) as raw:
        raw = {key: raw[key] for key in raw.files}
        mask = raw["unit_qc_mask"].astype(bool)
        support = raw["valid_spatial_bins"] & (raw["occupancy_first_half_s"] >= .05)
        with np.load(folder/"audit_arrays.npz", allow_pickle=False) as archive:
            data = {key: archive[key] for key in archive.files}
        same(data["rates_hz"], np.maximum(raw["rates_first_half_hz"][mask][:, support], 1e-4), "encoding source")
        same(data["grid_cm"], raw["bin_centers_cm"][support], "spatial support")
        assert np.array_equal(data["cell_ids"], raw["cell_ids"][mask])
        raw_ids = {int(value): i for i, value in enumerate(raw["candidate_event_indices"])}
        for e, event in enumerate(data["event_indices"]):
            index = raw_ids[int(event)]
            left, right = raw["candidate_offsets"][index:index+2]
            full = np.flatnonzero(np.isclose(raw["candidate_base_durations_s"][left:right], .005, atol=1e-9, rtol=0))
            bins = full[-4:]+left
            same(data["endpoint_counts"][e], np.sum(raw["candidate_base_counts"][bins][:, mask], axis=0), "all endpoint counts")
            same(data["endpoint_start_s"][e], raw["candidate_base_starts_s"][bins[0]], "endpoint start")
            same(data["endpoint_end_s"][e], raw["candidate_base_starts_s"][bins[-1]]+.005, "endpoint end")
        for block in ("calibration", "test"):
            indices = np.linspace(0, len(data[f"{block}_windows"])-1, 16, dtype=int)
            for cell_index, cell in enumerate(data["cell_ids"]):
                spikes = raw["spikes"][raw["spikes"][:, 1] == cell, 0]
                for i in indices:
                    start, end = data[f"{block}_windows"][i, :2]
                    n = np.count_nonzero((spikes >= start) & (spikes < end))
                    assert n == data[f"{block}_counts"][i, cell_index]
    frame = pd.read_csv(folder/"event_readouts.csv.gz")
    assert digest(folder/"event_readouts.csv.gz") == record["readouts_sha256"]
    checked = 0
    for part in frozen["parts"]:
        a, b = part["a"], part["b"]
        assert len(a) == len(b) and not set(a) & set(b)
        assert part["a_ids"] == data["cell_ids"][a].tolist()
        assert part["b_ids"] == data["cell_ids"][b].tolist()
        for source, key in (("real", "endpoint"), ("run_test", "test")):
            rows = frame.loc[frame.source.eq(source) & frame.split.eq(part["split"])].set_index("event_index")
            n = len(data[f"{key}_counts"])
            for i in np.linspace(0, n-1, min(24, n), dtype=int):
                event = int(data["event_indices"][i]) if source == "real" else i
                row = rows.loc[event]
                outputs = []
                for side, units in (("a", a), ("b", b)):
                    counts = data[f"{key}_counts"][i, units]
                    out = independent_posterior(counts, data["rates_hz"][units], data["grid_cm"])
                    outputs.append(out)
                    same(out[0], row[[f"{side}_x_cm", f"{side}_y_cm"]], "posterior mean")
                    same(out[1], row[f"{side}_width_cm"], "posterior width")
                    same(out[2], row[f"{side}_entropy"], "posterior entropy")
                    same(out[3], row[[f"{side}_region_{j}" for j in range(9)]], "regional probability")
                    same(counts.sum(), row[f"{side}_spikes"], "spike support")
                    if source == "run_test":
                        same(np.linalg.norm(out[0]-data["test_windows"][i, 2:]), row[f"{side}_truth_error_cm"], "RUN true error")
                same(np.linalg.norm(outputs[0][0]-outputs[1][0]), row.endpoint_separation_cm, "separation")
                same(.5*np.abs(outputs[0][3]-outputs[1][3]).sum(), row.regional_tv, "regional TV")
                checked += 1
    return dict(dataset=record["dataset"], animal=record["animal"], session=record["session"],
                all_endpoint_counts=len(data["event_indices"]), sampled_readouts=checked,
                sampled_run_windows=32, checked=True)


def check_selection(root):
    selections = pd.read_csv(root/"validation/selections.csv.gz")
    if selections.selected.dtype != bool:
        raise AssertionError("invalid boolean selection")
    for _, group in selections.groupby(["dataset", "animal", "session", "source", "split", "draw"]):
        expected = group.sort_values(["risk", "event_index"]).head(int(np.ceil(len(group)/2)))
        assert set(group.loc[group.selected, "event_index"]) == set(expected.event_index)
    sessions = pd.read_csv(root/"validation/policy_by_session.csv")
    total = pd.read_csv(root/"validation/policy_summary.csv")
    metrics = ["endpoint_separation_cm", "regional_tv", "a_entropy", "b_entropy", "a_truth_error_cm", "b_truth_error_cm"]
    for row in total.itertuples(index=False):
        local = sessions.loc[sessions.source.eq(row.source) & sessions.split.eq(row.split) &
                             sessions.policy.eq(row.policy) & sessions.retention.eq(row.retention)]
        for metric in metrics:
            per_animal = []
            for _, animal in local.groupby("animal"):
                per_session = [s[metric].mean() for _, s in animal.groupby("session")]
                per_animal.append(np.mean(per_session))
            same(np.mean(per_animal), getattr(row, metric), "equal-animal summary")
    return len(selections)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.result_dir
    statuses = json.loads((root/"status.json").read_text())
    assert statuses["status"] == "complete" and all(s["exit_code"] == 0 for s in statuses["stages"])
    model = json.loads((root/"frozen/frozen_model.json").read_text())
    pf = json.loads((root/"pf/manifest.json").read_text())
    external = json.loads((root/"tanni/manifest.json").read_text())
    assert model["training_animals"] == ["Rat1", "Rat2", "Rat3", "Rat4"]
    assert model["created_at_utc"] < external["created_at_utc"]
    assert external["input_file_sha256"]["frozen_model"] == digest(root/"frozen/frozen_model.json")
    assert pf["inputs_unchanged"] and external["inputs_unchanged"]
    records = []
    for manifest in (pf, external):
        assert manifest["status"] == "complete"
        for row in manifest["results"]:
            records.append(check_session(row))
            print("audited", row["dataset"], row["session"], flush=True)
    n_selections = check_selection(root)
    pd.DataFrame(records).to_csv(root/"independent_reconstruction.csv", index=False)
    result = dict(passed=True, sessions=len(records), all_endpoint_counts=sum(r["all_endpoint_counts"] for r in records),
                  sampled_readouts=sum(r["sampled_readouts"] for r in records), policy_rows=n_selections,
                  limitations="simulation truths and Ridge fitting not independently reconstructed; tested in unit suite",
                  created_at_utc=datetime.now(UTC).isoformat(), script_sha256=digest(__file__),
                  frozen_model_sha256=digest(root/"frozen/frozen_model.json"))
    (root/"independent_audit.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
