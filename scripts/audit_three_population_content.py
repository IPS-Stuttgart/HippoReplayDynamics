#!/usr/bin/env python3
"""Independently reconstruct three-population counts, posteriors and predictions."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/"src")]

import numpy as np
import pandas as pd
from scipy.special import expit, logsumexp

from scripts._provenance import build_script_provenance, file_sha256


KEYS = ["dataset", "animal", "session", "split", "source", "draw", "event_index"]


def independent_seed(*values):
    return int.from_bytes(hashlib.sha256("|".join(str(v) for v in values).encode()).digest()[:8], "little")


def raw_counts(spikes, ids, windows):
    out = np.empty((len(windows), len(ids)), np.int64)
    for column, cell in enumerate(ids):
        times = np.sort(spikes[spikes[:, 1] == cell, 0])
        out[:, column] = np.searchsorted(times, windows[:, 1], side="left")-np.searchsorted(times, windows[:, 0], side="left")
    return out


def posterior(counts, rates, grid):
    likelihood = counts @ np.log(rates) - .02*rates.sum(axis=0)
    p = np.exp(likelihood-logsumexp(likelihood, axis=1)[:, None])
    mean = p @ grid
    width = np.sqrt(np.maximum(np.sum(p*np.sum(grid**2, axis=1)[None, :], axis=1)-np.sum(mean**2, axis=1), 0))
    entropy = -(p*np.log(np.maximum(p, 1e-300))).sum(axis=1)/np.log(len(grid))
    scaled = (grid-grid.min(axis=0))/np.maximum(np.ptp(grid, axis=0), 1)
    tile = np.minimum(np.floor(3*scaled).astype(int), 2)
    ids = 3*tile[:, 0]+tile[:, 1]
    regional = np.column_stack([p[:, ids == i].sum(axis=1) for i in range(9)])
    radius = .15*np.linalg.norm(np.ptp(grid, axis=0))
    local = np.sum(p*(np.linalg.norm(grid[None, :, :]-mean[:, None, :], axis=2) <= radius), axis=1)
    return dict(p=p, mean=mean, width=width, entropy=entropy, regional=regional, local=local)


def reconstructed_columns(counts, rates, grid, groups, truth):
    a, b, c = [posterior(counts[:, ids], rates[ids], grid) for ids in groups]
    ab_ids = np.concatenate(groups[:2])
    pooled = posterior(counts[:, ab_ids], rates[ab_ids], grid)
    diagonal = float(np.linalg.norm(np.ptp(grid, axis=0)))
    near_a = np.linalg.norm(grid[None, :, :]-a["mean"][:, None, :], axis=2) <= .15*diagonal
    cmass = np.sum(c["p"]*near_a, axis=1)
    result = dict(
        pooled_spikes=counts[:, ab_ids].sum(axis=1), pooled_active=(counts[:, ab_ids] > 0).sum(axis=1),
        pooled_entropy=pooled["entropy"], pooled_width_scaled=pooled["width"]/diagonal,
        pooled_local_mass=pooled["local"], n_observed_cells=len(ab_ids),
        grid_diagonal_cm=diagonal, support_radius_cm=.15*diagonal,
        b_mass_near_a=np.sum(b["p"]*near_a, axis=1), ab_regional_tv=np.abs(a["regional"]-b["regional"]).sum(axis=1)/2,
        ab_separation_scaled=np.linalg.norm(a["mean"]-b["mean"], axis=1)/diagonal,
        c_support=cmass >= .5, c_mass_near_a=cmass,
        c_spikes=counts[:, groups[2]].sum(axis=1), c_active=(counts[:, groups[2]] > 0).sum(axis=1),
        c_entropy=c["entropy"], ac_regional_tv=np.abs(a["regional"]-c["regional"]).sum(axis=1)/2,
        ac_separation_cm=np.linalg.norm(a["mean"]-c["mean"], axis=1), c_width_cm=c["width"], a_width_cm=a["width"],
        a_truth_error_cm=np.linalg.norm(a["mean"]-truth, axis=1) if truth is not None else np.nan,
        c_truth_error_cm=np.linalg.norm(c["mean"]-truth, axis=1) if truth is not None else np.nan,
        a_x_cm=a["mean"][:, 0], a_y_cm=a["mean"][:, 1], c_x_cm=c["mean"][:, 0], c_y_cm=c["mean"][:, 1])
    for side, value, ids in (("a", a, groups[0]), ("b", b, groups[1])):
        result.update({f"{side}_spikes": counts[:, ids].sum(axis=1), f"{side}_active": (counts[:, ids] > 0).sum(axis=1),
                       f"{side}_entropy": value["entropy"], f"{side}_width_scaled": value["width"]/diagonal,
                       f"{side}_local_mass": value["local"]})
        for dim, name in enumerate(("x", "y")):
            result[f"{side}_{name}_scaled"] = (value["mean"][:, dim]-grid[:, dim].min())/diagonal
    return result


def verify_recording(row):
    root = Path(row.artifact_dir)
    freeze = json.loads((root/"frozen_populations.json").read_text())
    if file_sha256(freeze["cache_path"]) != freeze["cache_sha256"]:
        raise ValueError("input cache checksum differs")
    with np.load(freeze["cache_path"], allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    with np.load(root/"audit_arrays.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    mask = data["unit_qc_mask"].astype(bool)
    ids = data["cell_ids"][mask]
    support = data["valid_spatial_bins"] & (data["occupancy_first_half_s"] >= .05)
    grid = data["bin_centers_cm"][support]
    rates = np.maximum(data["rates_first_half_hz"][mask][:, support], 1e-4)
    drift = np.maximum(data["rates_second_half_hz"][mask][:, support], 1e-4)
    for key, expected in (("grid_cm", grid), ("rates_hz", rates), ("drift_hz", drift), ("cell_ids", ids)):
        np.testing.assert_array_equal(arrays[key], expected)
    windows = np.column_stack([arrays["endpoint_start_s"], arrays["endpoint_end_s"]])
    counts = raw_counts(data["spikes"], ids, windows)
    np.testing.assert_array_equal(counts, arrays["endpoint_counts"])
    for name in ("calibration", "test"):
        np.testing.assert_array_equal(raw_counts(data["spikes"], ids, arrays[f"{name}_windows"][:, :2]), arrays[f"{name}_counts"])
        windows_run = arrays[f"{name}_windows"]
        midpoint = (windows_run[:, 0]+windows_run[:, 1])/2
        truth = np.column_stack([np.interp(midpoint, data["position"][:, 0], data["position"][:, dim]) for dim in (1, 2)])
        np.testing.assert_allclose(windows_run[:, 2:], truth, rtol=1e-9, atol=1e-8)
    # Independently recover the last four full native base bins, not just the
    # supplied endpoint counts. This also catches an event-tail clock shift.
    expected_anchors = []
    for e, event in enumerate(data["candidate_event_indices"]):
        lo, hi = data["candidate_offsets"][e:e+2]
        full = np.flatnonzero(np.abs(data["candidate_base_durations_s"][lo:hi]-.005) <= 1e-9)
        if len(full) < 4 or not np.array_equal(full, np.arange(len(full))):
            continue
        base = lo+full[-4:]
        expected_anchors.append([event, data["candidate_base_starts_s"][base[0]], data["candidate_base_starts_s"][base[-1]]+.005])
    expected_anchors = np.asarray(expected_anchors)
    np.testing.assert_allclose(windows, expected_anchors[:, 1:], rtol=0, atol=1e-10)
    np.testing.assert_array_equal(arrays["event_indices"], expected_anchors[:, 0])
    frame = pd.read_csv(root/"event_readouts.csv.gz")
    if file_sha256(root/"event_readouts.csv.gz") != row.readouts_sha256 or frame.duplicated(KEYS).any():
        raise ValueError("changed or duplicated readout rows")
    validated = 0
    for part in freeze["parts"]:
        split = part["split"]
        n = len(ids)//3
        permutation = np.random.default_rng(independent_seed(freeze["seed"], "three_population", row.dataset,
            row.animal, row.session, split)).permutation(len(ids))
        groups = [np.sort(permutation[j*n:(j+1)*n]) for j in range(3)]
        for name, group in zip(("a", "b", "c"), groups, strict=True):
            np.testing.assert_array_equal(part[name], group)
            np.testing.assert_array_equal(part[f"{name}_ids"], ids[group])
        cases = [("real", -1, counts, None, arrays["event_indices"]),
                 ("run_test", -1, arrays["test_counts"], arrays["test_windows"][:, 2:], np.arange(len(arrays["test_counts"])))]
        for source in ("sim_matched", "sim_drift"):
            for draw in range(2):
                rng = np.random.default_rng(independent_seed(freeze["seed"], "three_population_sim", row.dataset,
                    row.animal, row.session, split, source, draw))
                states = rng.integers(len(grid), size=len(counts))
                generate = rates if source == "sim_matched" else drift*rng.lognormal(0, .4, (len(ids), 1))
                probabilities = generate[:, states].T
                probabilities /= probabilities.sum(axis=1, keepdims=True)
                simulated = np.array([rng.multinomial(int(total), p) for total, p in zip(counts.sum(axis=1), probabilities, strict=True)])
                prefix = f"split{split}_{source}_{draw}"
                np.testing.assert_array_equal(arrays[f"{prefix}_counts"], simulated)
                np.testing.assert_array_equal(arrays[f"{prefix}_states"], states)
                cases.append((source, draw, simulated, grid[states], arrays["event_indices"]))
        for source, draw, observations, truth, events in cases:
            local = frame.loc[(frame.split == split) & frame.source.eq(source) & (frame.draw == draw)].sort_values("event_index")
            order = np.argsort(events)
            np.testing.assert_array_equal(local.event_index.to_numpy(), np.asarray(events)[order])
            rebuilt = reconstructed_columns(observations[order], rates, grid, groups, truth[order] if truth is not None else None)
            for column, value in rebuilt.items():
                np.testing.assert_allclose(local[column].to_numpy(float), value, rtol=1e-9, atol=1e-8, equal_nan=True,
                                           err_msg=f"{row.session}:{split}:{source}:{draw}:{column}")
            validated += len(local)
    assert validated == len(frame)
    return dict(dataset=row.dataset, animal=row.animal, session=row.session, rows=validated, endpoints=len(counts),
                raw_run_windows=len(arrays["calibration_windows"])+len(arrays["test_windows"]),
                readout_sha256=file_sha256(root/"event_readouts.csv.gz")), frame


def verify_external_predictions(frame, frozen, actual):
    features = set(frozen["models"]["full"]["features"])
    if not features or any(name.startswith("c_") for name in features):
        raise ValueError("C leakage in frozen feature specification")
    if not set(frozen["models"]["pooled"]["features"]).issubset(features):
        raise ValueError("unfair pooled baseline")
    frame = frame.sort_values(KEYS).reset_index(drop=True)
    actual = actual.sort_values(KEYS).reset_index(drop=True)
    pd.testing.assert_frame_equal(frame[KEYS], actual[KEYS])
    for column in frame:
        if column not in KEYS:
            np.testing.assert_allclose(frame[column].to_numpy(float), actual[column].to_numpy(float), rtol=1e-10, atol=1e-10, equal_nan=True)
    for name, model in frozen["models"].items():
        if model["kind"] == "constant":
            probability = np.full(len(frame), model["prevalence"])
        else:
            x = frame[model["features"]].to_numpy(float)
            x = np.where(np.isfinite(x), x, model["medians"])
            score = np.sum((x-model["mean"])/model["scale"]*model["coef"], axis=1)+model["intercept"]
            probability = expit(score)
        probability = np.clip(probability, 1e-12, 1-1e-12)
        y = frame.c_support.to_numpy(int)
        for column, expected in ((f"prediction_{name}", probability),
                                (f"logloss_{name}", -y*np.log(probability)-(1-y)*np.log1p(-probability)),
                                (f"brier_{name}", (probability-y)**2)):
            np.testing.assert_allclose(actual[column], expected, rtol=1e-9, atol=1e-9)
    for policy, column, direction in (("full", "prediction_full", False), ("pooled", "prediction_pooled", False),
                                      ("a_entropy", "a_entropy", True), ("pooled_entropy", "pooled_entropy", True)):
        for _, local in actual.groupby(KEYS[:-1]):
            order = local.sort_values([column, "event_index"], ascending=[direction, True])
            selected = set(order.event_index.iloc[:(len(order)+1)//2])
            np.testing.assert_array_equal(local[f"retained_{policy}"].to_numpy(), local.event_index.isin(selected).to_numpy())
    return len(actual)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurement-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--external-report", type=Path)
    parser.add_argument("--frozen-model", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    provenance = build_script_provenance(input_paths=dict(auditor=Path(__file__), frozen_model=args.frozen_model,
        protocol=ROOT/"docs/three_population_content_protocol.md"), cwd=ROOT)
    rows, external = [], []
    for root in args.measurement_dirs:
        manifest = json.loads((root/"manifest.json").read_text())
        if manifest["status"] != "complete" or not manifest["inputs_unchanged"]:
            raise ValueError("unfinished measurement")
        catalog = pd.read_csv(root/"sessions.csv")
        for row in catalog.loc[catalog.status.eq("complete")].itertuples(index=False):
            result, frame = verify_recording(row)
            rows.append(result)
            if row.dataset == "autopi_ca1":
                external.append(frame)
            print(json.dumps(result), flush=True)
            pd.DataFrame(rows).to_csv(args.output_dir/"reconstruction_by_session.csv", index=False)
    prediction_rows = 0
    if args.external_report is not None:
        if args.frozen_model is None or not external:
            raise ValueError("external predictions have no independent reconstruction")
        prediction_rows = verify_external_predictions(pd.concat(external), json.loads(args.frozen_model.read_text()),
                                                      pd.read_csv(args.external_report/"external_predictions.csv.gz"))
    report = dict(status="passed", created_at_utc=datetime.now(UTC).isoformat(), provenance=provenance,
        recordings=len(rows), reconstructed_rows=sum(row["rows"] for row in rows),
        reconstructed_endpoints=sum(row["endpoints"] for row in rows), prediction_rows=prediction_rows,
        verified_readouts={row["dataset"]+":"+row["session"]: row["readout_sha256"] for row in rows},
        scope="raw cached timestamps/counts; independent dense Poisson; simulation reconstruction; A/B-only predictors; retention",
        not_verified="raw spike sorting or ground-truth biological replay; encoding maps shared with original cache",
        aggregate_gate_audit="separate_required_before_any_validation_claim")
    (args.output_dir/"independent_audit.json").write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
