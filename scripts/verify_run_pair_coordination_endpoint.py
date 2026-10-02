"""Independently recompute rate controls and physical-lag RUN change endpoints."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import fftconvolve
from scipy.special import xlogy

try:
    from scripts._provenance import build_script_provenance, file_sha256
except ModuleNotFoundError:
    from _provenance import build_script_provenance, file_sha256


def check(value, message):
    if not value:
        raise ValueError(message)


def independent_residuals(bank, period, p):
    phase = np.asarray(bank[f"{period}_theta_phase_rad"])
    check(phase.ndim == 2 and phase.shape[1] > 0, "Missing theta reference")
    valid = np.isfinite(phase).all(axis=1)
    times = bank[f"{period}_time_s"][valid]
    counts = bank[f"{period}_counts"][valid]
    check(len(times) > 0 and np.all(np.diff(times) > 0), "Empty/backward valid RUN clock")
    position = bank[f"{period}_position_cm"][valid]
    direction = bank[f"{period}_direction_rad"][valid]
    speed = bank[f"{period}_speed_cm_s"][valid]
    theta = phase[valid]
    edges = np.asarray(p["speed_edges_cm_s"])
    speed_bin = np.zeros(len(speed), int)
    for boundary in edges[1:-1]:
        speed_bin += speed >= boundary
    coordinates = [np.floor(position[:, 0] / p["position_bin_cm"]).astype(int),
                   np.floor(position[:, 1] / p["position_bin_cm"]).astype(int),
                   np.floor((direction % (2 * np.pi)) / (2 * np.pi / p["direction_bins"])).astype(int),
                   speed_bin]
    coordinates.extend(np.floor(((theta[:, k] + np.pi) % (2 * np.pi)) /
                                (2 * np.pi / p["theta_bins"])).astype(int) for k in range(theta.shape[1]))
    keys = pd.MultiIndex.from_arrays(coordinates)
    labels, _ = pd.factorize(keys, sort=False)
    frame = pd.DataFrame(counts)
    frame["stratum"] = labels
    blocks = np.floor(times / p["crossfit_time_block_s"]).astype(np.int64)
    mean, global_mean = np.full(counts.shape, np.nan), np.full(counts.shape, np.nan)
    seen = np.zeros(len(times), bool)
    fold_diagnostics = []
    for f in range(p["crossfit_folds"]):
        validation = (blocks % p["crossfit_folds"]) == f
        if not validation.any():
            continue
        training = ~validation
        validation_blocks = set(blocks[validation])
        # Check nearest block edges instead of looping through validation intervals.
        block_time = times - blocks * p["crossfit_time_block_s"]
        for i in np.flatnonzero(training):
            if ((blocks[i] - 1 in validation_blocks and block_time[i] < p["crossfit_guard_s"])
                    or (blocks[i] + 1 in validation_blocks and
                        block_time[i] >= p["crossfit_time_block_s"] - p["crossfit_guard_s"])):
                training[i] = False
        if not training.any():
            fold_diagnostics.append({"fold": f, "status": "no_guarded_training_bins",
                                     "validation_bins": int(validation.sum())})
            continue
        totals = frame.loc[training].groupby("stratum").sum()
        exposures = frame.loc[training].groupby("stratum").size() * p["run_bin_s"]
        global_rate = (counts[training].sum(axis=0) + p["global_rate_prior_spikes"]) / (
            training.sum() * p["run_bin_s"] + p["global_rate_prior_exposure_s"])
        targets = labels[validation]
        numerator = totals.reindex(targets, fill_value=0).to_numpy() + p["rate_prior_exposure_s"] * global_rate
        denominator = exposures.reindex(targets, fill_value=0).to_numpy() + p["rate_prior_exposure_s"]
        mean[validation] = np.maximum(p["run_bin_s"] * numerator / denominator[:, None], p["mean_count_floor"])
        global_mean[validation] = np.maximum(p["run_bin_s"] * global_rate, p["mean_count_floor"])
        seen[validation] = np.isin(targets, exposures.index)
        fold_diagnostics.append({"fold": f, "status": "predicted", "training_bins": int(training.sum()),
                                 "validation_bins": int(validation.sum()),
                                 "seen_stratum_fraction": float(seen[validation].mean()),
                                 "training_spikes": int(counts[training].sum())})
    usable = np.isfinite(mean).all(axis=1)
    check(usable.any(), "No out-of-fold predictions")
    residual = (counts[usable] - mean[usable]) / np.sqrt(mean[usable])
    qc = {"source_bins": len(phase), "usable_bins": int(usable.sum()),
          "predicted_bins": int(usable.sum()), "total_bins": len(times),
          "unseen_stratum_fraction": float((~seen[usable]).mean()),
          "heldout_poisson_improvement_over_global": float(np.sum(
              xlogy(counts[usable], mean[usable] / global_mean[usable]) - mean[usable] + global_mean[usable])),
          "folds": fold_diagnostics}
    return residual, times[usable], qc


def independent_coordination(residual, times, p):
    width = p["run_bin_s"]
    indices = np.rint((times - times[0]) / width).astype(int)
    check(np.allclose(times, times[0] + indices * width, rtol=0, atol=1e-8)
          and np.all(np.diff(indices) > 0), "Not a unique physical grid")
    size = int(indices[-1]) + 1
    check(size <= 2_000_000, "Unexpected clock span")
    dense = np.zeros((size, residual.shape[1]))
    dense[indices] = residual
    lags = np.arange(int(np.floor(p["lag_max_s"] / width + 1e-9)) + 1) * width
    kernel = ((lags >= p["lag_min_s"] - 1e-12) & (lags <= p["lag_max_s"] + 1e-12)).astype(float)
    future = fftconvolve(dense[::-1], kernel[:, None], mode="full", axes=0)[:size][::-1]
    products = residual.T @ future[indices]
    present = np.zeros(size)
    present[indices] = 1
    future_present = fftconvolve(present[::-1], kernel, mode="full")[:size][::-1]
    opportunities = int(np.rint(future_present[indices].sum()))
    check(opportunities > 0, "No physical lag pairs")
    return (products - products.T) / opportunities, opportunities


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.endpoint_dir
    m = json.loads((root / "manifest.json").read_text())
    for name, sha in m["outputs_sha256"].items():
        check(file_sha256(root / name) == sha, f"Endpoint output changed: {name}")
    for name, path in m["input_file_paths"].items():
        check(file_sha256(path) == m["input_file_sha256"][name], f"Endpoint input changed: {name}")
    source_manifest_path = Path(m["input_file_paths"]["measurement_manifest"])
    source = source_manifest_path.parent
    source_manifest = json.loads(source_manifest_path.read_text())
    for name, sha in source_manifest["outputs_sha256"].items():
        check(file_sha256(source / name) == sha, f"Measurement output changed: {name}")
    p = json.loads(Path(m["input_file_paths"]["protocol"]).read_text())
    check(p.get("rate_model_family") != "smooth_poisson_glm",
          "This verifier is for the original joint-stratum estimator, not the GLM amendment")
    banks = pd.read_csv(source / "banks.csv")
    source_pauses = pd.read_csv(source / "pauses.csv").set_index(["session", "pause_id"])
    pauses = pd.read_csv(root / "pauses.csv").set_index(["session", "pause_id"])
    pairs = pd.read_csv(root / "pairs.csv", float_precision="round_trip")
    quality = pd.read_csv(root / "period_quality.csv", float_precision="round_trip").set_index(["session", "pause_id", "period"])
    check(pauses.index.is_unique and source_pauses.index.is_unique and
          len(pauses) == len(source_pauses) == len(banks), "Pause denominator/identity failed")
    check(not pairs.duplicated(["session", "pause_id", "unit_a", "unit_b"]).any(), "Duplicate pair endpoint")
    np.testing.assert_allclose(pairs.coordination_change, pairs.post_coordination - pairs.pre_coordination,
                               rtol=0, atol=1e-12)
    check(not pairs.independent_biological_subject.any() and not pairs.association_fit.any(), "Invalid biological claim flags")
    checked_pairs, checked_bins, measured = 0, 0, 0
    for bank_row in banks.itertuples(index=False):
        identity = (bank_row.session, bank_row.pause_id)
        check(file_sha256(bank_row.bank_path) == bank_row.bank_sha256, "RUN bank changed")
        eligible = bool(source_pauses.loc[identity, "theta_run_support_screen_passed"])
        selected = pairs[(pairs.session == bank_row.session) & (pairs.pause_id == bank_row.pause_id)]
        if not eligible:
            check(selected.empty and pauses.loc[identity, "status"] == "frozen_theta_screen_failed", "Theta-failed pause measured")
            continue
        with np.load(bank_row.bank_path, allow_pickle=False) as bank:
            ids = bank["unit_ids"]
            check(len(selected) == len(ids) * (len(ids) - 1) // 2, "Unit-pair completeness failed")
            a, b = np.triu_indices(len(ids), 1)
            selected = selected.set_index(["unit_a", "unit_b"]).loc[list(zip(ids[a], ids[b], strict=True))]
            for period in ("pre", "post"):
                residual, times, qc = independent_residuals(bank, period, p)
                matrix, opportunities = independent_coordination(residual, times, p)
                recorded = quality.loc[(*identity, period)]
                for name, value in qc.items():
                    if name != "folds":
                        np.testing.assert_allclose(recorded[name], value, rtol=1e-10, atol=1e-9, err_msg=name)
                check(json.loads(recorded.fold_diagnostics) == qc["folds"], "Crossfit-fold diagnostics differ")
                check(recorded.physical_lag_opportunities == opportunities, "Physical lag support differs")
                np.testing.assert_allclose(selected[f"{period}_coordination"], matrix[a, b], rtol=1e-8, atol=1e-10)
                checked_bins += len(times)
            check(pauses.loc[identity, "pair_endpoints"] == len(selected), "Pause pair count differs")
            checked_pairs += len(selected)
            measured += 1
        print(f"VERIFIED {bank_row.animal} {bank_row.pause_id}: {len(selected)} dependent pairs", flush=True)
    check(checked_pairs == len(pairs) and measured == m["measured_pauses"], "Endpoint accounting incomplete")
    provenance = build_script_provenance(input_paths={"endpoint_manifest": root / "manifest.json"})
    check(provenance["git_dirty"] is False and provenance["code_commit"] != "unavailable", "Clean committed verifier checkout required")
    result = {**provenance, "verified": True, "checked_pair_endpoints": checked_pairs,
              "checked_run_bins": checked_bins, "checked_frozen_pauses": len(banks), "measured_pauses": measured,
              "method": "Independent pandas-grouped out-of-fold rates and FFT physical-lag cross-products",
              "scope": "Arithmetic, identities, hashes and denominators only; not null calibration, replay validation or association",
              "association_verified": False, "biological_calibration_verified": False,
              "created_at_utc": datetime.now(timezone.utc).isoformat()}
    check(not args.output.exists(), "Do not overwrite a verification artifact")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"COMPLETE verified={checked_pairs} dependent pairs; association_verified=False", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
