"""Measure a candidate covariate-adjusted RUN change endpoint; no association fit."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import xlogy

try:
    from scripts._provenance import build_script_provenance, file_sha256
except ModuleNotFoundError:
    from _provenance import build_script_provenance, file_sha256


def require(condition, message):
    if not condition:
        raise ValueError(message)


def strata(position, direction, speed, theta, p):
    position, direction, speed, theta = map(np.asarray, (position, direction, speed, theta))
    n = len(position)
    require(position.shape == (n, 2) and direction.shape == speed.shape == (n,)
            and theta.ndim == 2 and len(theta) == n, "Covariate dimensions disagree")
    require(all(np.isfinite(x).all() for x in (position, direction, speed, theta)), "Missing covariates cannot become control bins")
    edge = np.asarray(p["speed_edges_cm_s"])
    require(np.all(speed > edge[0]) and np.all(speed <= edge[-1]), "RUN speed outside frozen bounds")
    bins = np.column_stack((np.floor(position / p["position_bin_cm"]).astype(int),
                            np.floor(np.mod(direction, 2 * np.pi) * p["direction_bins"] / (2 * np.pi)).astype(int),
                            np.minimum(np.searchsorted(edge, speed, side="right") - 1, len(edge) - 2),
                            np.floor(np.mod(theta + np.pi, 2 * np.pi) * p["theta_bins"] / (2 * np.pi)).astype(int)))
    _, labels = np.unique(bins, axis=0, return_inverse=True)
    return labels


def crossfit_rates(counts, times, labels, width, p, covariates=None, prepared=None):
    """Rate fitting uses only the period supplied, and never held-out-block spikes."""
    if p.get("rate_model_family") == "smooth_poisson_glm":
        try:
            from scripts._run_pair_rate_glm import crossfit
        except ModuleNotFoundError:
            from _run_pair_rate_glm import crossfit
        return crossfit(counts, times, labels, width, p, covariates, prepared)
    counts, times, labels = np.asarray(counts), np.asarray(times, float), np.asarray(labels, int)
    require(counts.ndim == 2 and len(counts) == len(times) == len(labels) and len(times) > 0,
            "Nonempty aligned RUN counts and clock required")
    require(np.isfinite(counts).all() and np.all(counts >= 0) and np.all(counts == np.floor(counts)), "Invalid spike counts")
    require(np.isfinite(times).all() and np.all(np.diff(times) > 0) and np.all(labels >= 0), "Invalid clock/strata")
    require(width > 0 and p["crossfit_folds"] >= 2, "Positive width and at least two folds required")
    require(p["crossfit_time_block_s"] > 2 * p["crossfit_guard_s"] >= 0,
            "Crossfit blocks must exceed twice the nonnegative guard")
    require(all(p[name] > 0 for name in ("rate_prior_exposure_s", "global_rate_prior_spikes",
                                       "global_rate_prior_exposure_s", "mean_count_floor")),
            "Rate priors and mean count floor must be positive")
    blocks = np.floor(times / p["crossfit_time_block_s"]).astype(np.int64)
    fold = blocks % p["crossfit_folds"]
    mean = np.full(counts.shape, np.nan)
    global_mean = np.full(counts.shape, np.nan)
    seen = np.zeros(len(times), bool)
    diagnostics = []
    n_strata = int(labels.max()) + 1
    for f in range(p["crossfit_folds"]):
        validation = fold == f
        if not validation.any():
            continue
        # Exclude adjacent training endpoints within one maximum coordination lag.
        selected_blocks = np.unique(blocks[validation])
        lo = selected_blocks * p["crossfit_time_block_s"] - p["crossfit_guard_s"]
        hi = (selected_blocks + 1) * p["crossfit_time_block_s"] + p["crossfit_guard_s"]
        training = fold != f
        for a, b in zip(lo, hi, strict=True):
            training &= (times < a) | (times >= b)
        if not training.any():
            diagnostics.append({"fold": f, "status": "no_guarded_training_bins", "validation_bins": int(validation.sum())})
            continue
        global_rate = (counts[training].sum(axis=0) + p["global_rate_prior_spikes"]) / (
            training.sum() * width + p["global_rate_prior_exposure_s"])
        exposure = np.bincount(labels[training], minlength=n_strata) * width
        totals = np.zeros((n_strata, counts.shape[1]))
        np.add.at(totals, labels[training], counts[training])
        prior = p["rate_prior_exposure_s"]
        rates = (totals + prior * global_rate[None, :]) / (exposure[:, None] + prior)
        mean[validation] = np.maximum(width * rates[labels[validation]], p["mean_count_floor"])
        global_mean[validation] = np.maximum(width * global_rate, p["mean_count_floor"])
        seen[validation] = exposure[labels[validation]] > 0
        diagnostics.append({"fold": f, "status": "predicted", "training_bins": int(training.sum()),
                            "validation_bins": int(validation.sum()),
                            "seen_stratum_fraction": float(seen[validation].mean()),
                            "training_spikes": int(counts[training].sum())})
    complete = np.isfinite(mean).all(axis=1)
    score_delta = float(np.sum(xlogy(counts[complete], mean[complete] / global_mean[complete])
                               - mean[complete] + global_mean[complete])) if complete.any() else np.nan
    residual = (counts - mean) / np.sqrt(mean)
    return residual, {"predicted_bins": int(complete.sum()), "total_bins": len(times),
                      "unseen_stratum_fraction": float((~seen[complete]).mean()) if complete.any() else np.nan,
                      "heldout_poisson_improvement_over_global": score_delta, "folds": diagnostics}


def residual_coordination(residual, times, width, low, high):
    """Sum physical-lag cross-products in O(time * cells + observed_time * cells**2)."""
    residual, times = np.asarray(residual, float), np.asarray(times, float)
    require(residual.ndim == 2 and len(residual) == len(times) and len(times) > 0, "Aligned residual clock required")
    require(np.isfinite(residual).all() and np.isfinite(times).all() and np.all(np.diff(times) > 0), "Nonfinite or backward residual clock")
    require(width > 0 and high >= low > 0, "Invalid coordination lags")
    index = np.rint((times - times[0]) / width).astype(np.int64)
    require(np.allclose(times, times[0] + index * width, rtol=0, atol=1e-8)
            and np.all(np.diff(index) > 0), "RUN bins not on a distinct physical grid")
    first = int(np.ceil((low - 1e-12) / width))
    last = int(np.floor((high + 1e-12) / width))
    n = int(index[-1]) + 1
    require(n <= 2_000_000, "Clock span too large for this bounded RUN estimator")
    dense = np.zeros((n, residual.shape[1]))
    dense[index] = residual
    cumulative = np.vstack((np.zeros((1, residual.shape[1])), np.cumsum(dense, axis=0)))
    upper, lower = np.minimum(index + last + 1, n), np.minimum(index + first, n)
    future = cumulative[upper] - cumulative[lower]
    products = residual.T @ future
    present = np.zeros(n, np.int64)
    present[index] = 1
    observed = np.r_[0, present.cumsum()]
    opportunities = int(np.sum(observed[upper] - observed[lower]))
    result = (products - products.T) / opportunities if opportunities else np.full_like(products, np.nan)
    return result, opportunities


def period_endpoint(bank, period, p):
    counts, times = bank[f"{period}_counts"], bank[f"{period}_time_s"]
    theta = bank[f"{period}_theta_phase_rad"]
    widths = bank[f"{period}_bin_duration_s"]
    require(np.allclose(widths, p["run_bin_s"], rtol=0, atol=1e-12), "RUN bin width differs from frozen endpoint")
    require(theta.ndim == 2 and theta.shape[0] == len(times) and theta.shape[1] > 0,
            "At least one aligned LFP theta reference required")
    valid = np.isfinite(theta).all(axis=1)
    if not valid.any():
        return None, {"status": "no_valid_theta_endpoints", "source_bins": len(times), "usable_bins": 0}
    labels = strata(bank[f"{period}_position_cm"][valid], bank[f"{period}_direction_rad"][valid],
                    bank[f"{period}_speed_cm_s"][valid], theta[valid], p)
    covariates = {key: bank[f"{period}_{source}"][valid] for key, source in
                  (("position", "position_cm"), ("direction", "direction_rad"),
                   ("speed", "speed_cm_s"), ("theta", "theta_phase_rad"))}
    if p.get("rate_training_support") == "same_period_full_run":
        pool = support_pool(bank, period, p, counts[valid], times[valid], covariates)
        pool_counts, pool_times, pool_covariates, target = pool
        try:
            from scripts._run_pair_rate_glm import crossfit
        except ModuleNotFoundError:
            from _run_pair_rate_glm import crossfit
        pool_labels = strata(pool_covariates["position"], pool_covariates["direction"], pool_covariates["speed"], pool_covariates["theta"], p)
        fitted, diagnostics, means, baseline = crossfit(pool_counts, pool_times, pool_labels, p["run_bin_s"],
            p, pool_covariates, return_predictions=True)
        residual = fitted[target]
        diagnostics["training_pool_unseen_stratum_fraction"] = diagnostics.pop("unseen_stratum_fraction")
        diagnostics.update(rate_training_support="same_period_full_run", training_pool_bins=len(pool_times),
            predicted_bins=int(np.isfinite(residual).all(axis=1).sum()), total_bins=len(target),
            predicted_mean_count_min=float(means[target].min()), predicted_mean_count_max=float(means[target].max()),
            heldout_poisson_improvement_over_global=float(np.sum(xlogy(counts[valid], means[target] / baseline[target]) - means[target] + baseline[target])))
    else:
        residual, diagnostics = crossfit_rates(counts[valid], times[valid], labels, p["run_bin_s"], p, covariates=covariates)
    predicted = np.isfinite(residual).all(axis=1)
    if p.get("rate_model_family") == "smooth_poisson_glm" and not predicted.all():
        return None, {"status": "incomplete_glm_crossfit_predictions", "source_bins": len(times),
                      "usable_bins": 0, **diagnostics}
    if not predicted.any():
        return None, {"status": "no_crossfit_predictions", "source_bins": len(times), "usable_bins": 0, **diagnostics}
    matrix, opportunities = residual_coordination(residual[predicted], times[valid][predicted],
                                                  p["run_bin_s"], p["lag_min_s"], p["lag_max_s"])
    return matrix if opportunities else None, {"status": "measured" if opportunities else "no_physical_lag_pairs", "source_bins": len(times),
                    "usable_bins": int(predicted.sum()), "physical_lag_opportunities": opportunities, **diagnostics}


def support_pool(bank, period, p, counts, times, covariates):
    prefix = f"{period}_rate_support"
    require(p.get("rate_model_family") == "smooth_poisson_glm", "Support expansion requires the frozen smooth GLM")
    phase = bank[f"{prefix}_theta_phase_rad"]
    require(phase.ndim == 2 and phase.shape[1] > 0, "Native support theta required")
    require(np.allclose(bank[f"{prefix}_bin_duration_s"], p["run_bin_s"], rtol=0, atol=1e-12), "Support bin width changed")
    valid = np.isfinite(phase).all(axis=1)
    pool_times, pool_counts = bank[f"{prefix}_time_s"][valid], bank[f"{prefix}_counts"][valid]
    require(len(pool_times) > 0 and np.all(np.diff(pool_times) > 0), "Empty or backward support clock")
    require(pool_counts.shape == (len(pool_times), counts.shape[1]), "Support population changed")
    pool_covariates = {key: bank[f"{prefix}_{name}"][valid] for key, name in
        (("position", "position_cm"), ("direction", "direction_rad"), ("speed", "speed_cm_s"), ("theta", "theta_phase_rad"))}
    index = np.searchsorted(pool_times, times)
    require(np.all(index < len(pool_times)), "Endpoint clock missing from support")
    require(np.array_equal(pool_times[index], times) and np.array_equal(pool_counts[index], counts), "Endpoint counts or clock changed")
    for key, value in covariates.items():
        require(np.array_equal(pool_covariates[key][index], value), "Endpoint covariates changed")
    return pool_counts, pool_times, pool_covariates, index


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurement-dir", type=Path, required=True)
    parser.add_argument("--measurement-verification", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    p = json.loads(args.protocol.read_text())
    require(p["crossfit_guard_s"] >= p["lag_max_s"], "Crossfit guard must cover the maximum coordination lag")
    root = args.measurement_dir
    manifest = json.loads((root / "manifest.json").read_text())
    verification = json.loads(args.measurement_verification.read_text())
    require(verification.get("verified") and verification.get("input_file_sha256", {}).get("manifest") == file_sha256(root / "manifest.json"),
            "Successful identity-matched independent measurement verification required")
    for name, digest in manifest["outputs_sha256"].items():
        require(file_sha256(root / name) == digest, f"Changed measurement {name}")
    provenance = build_script_provenance(input_paths={"protocol": args.protocol, "measurement_manifest": root / "manifest.json",
                                                     "measurement_verification": args.measurement_verification})
    require(provenance["git_dirty"] is False and provenance["code_commit"] != "unavailable", "Clean committed checkout required")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    pauses = pd.read_csv(root / "pauses.csv").set_index(["session", "pause_id"])
    banks = pd.read_csv(root / "banks.csv")
    require(pauses.index.is_unique and not banks.duplicated(["session", "pause_id"]).any(),
            "Duplicate frozen pause identity")
    require(len(pauses) == len(banks), "Bank and pause denominators disagree")
    require(pauses["theta_run_support_screen_passed"].isin([True, False]).all(),
            "Missing or invalid theta screen status")
    pair_rows, period_rows, pause_rows = [], [], []
    for row in banks.itertuples(index=False):
        identity = {"animal": row.animal, "session": row.session, "pause_id": row.pause_id}
        meta = pauses.loc[(row.session, row.pause_id)]
        require(row.animal == meta.animal, "Animal identity disagrees across pause and bank tables")
        require(file_sha256(row.bank_path) == row.bank_sha256, "Changed RUN bank")
        if not meta.theta_run_support_screen_passed:
            pause_rows.append({**identity, "status": "frozen_theta_screen_failed", "pair_endpoints": 0})
            continue
        with np.load(row.bank_path, allow_pickle=False) as bank:
            ids = bank["unit_ids"]
            pre, pre_qc = period_endpoint(bank, "pre", p)
            post, post_qc = period_endpoint(bank, "post", p)
            for label, qc in (("pre", pre_qc), ("post", post_qc)):
                period_rows.append({**identity, "period": label, **{k: v for k, v in qc.items() if k != "folds"},
                                    "fold_diagnostics": json.dumps(qc.get("folds", []))})
            n_pairs = 0
            if pre is not None and post is not None and pre_qc["status"] == post_qc["status"] == "measured":
                require(np.isfinite(pre).all() and np.isfinite(post).all(), "Nonfinite measured coordination")
                for a in range(len(ids)):
                    for b in range(a + 1, len(ids)):
                        pair_rows.append({**identity, "unit_a": int(ids[a]), "unit_b": int(ids[b]),
                                          "pre_coordination": float(pre[a, b]), "post_coordination": float(post[a, b]),
                                          "coordination_change": float(post[a, b] - pre[a, b]),
                                          "independent_biological_subject": False, "association_fit": False})
                        n_pairs += 1
            pause_rows.append({**identity, "status": "candidate_endpoint_measured" if n_pairs else "endpoint_unavailable",
                               "pair_endpoints": n_pairs})
        print(f"MEASURED {row.session} {row.pause_id}: {n_pairs} dependent endpoints", flush=True)
    require(len(pause_rows) == len(banks), "Frozen pause accounting incomplete")
    frames = {"pairs.csv": pd.DataFrame(pair_rows, columns=None if pair_rows else ["animal", "session", "pause_id", "unit_a", "unit_b", "pre_coordination", "post_coordination", "coordination_change", "independent_biological_subject", "association_fit"]),
              "period_quality.csv": pd.DataFrame(period_rows), "pauses.csv": pd.DataFrame(pause_rows)}
    for name, frame in frames.items():
        frame.to_csv(args.output_dir / name, index=False)
    result = {**provenance, "protocol_id": p["protocol_id"], "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "outputs_sha256": {name: file_sha256(args.output_dir / name) for name in frames},
              "candidate_endpoint_measured": bool(pair_rows), "association_fit": False, "biological_calibration_complete": False,
              "frozen_pauses": len(banks), "measured_pauses": sum(row["pair_endpoints"] > 0 for row in pause_rows),
              "dependent_pair_endpoints": len(pair_rows),
              "replay_validated": False, "preceding_coordination_control_stored_not_fitted": True,
              "inference_unit": "animal", "goal_complete": False}
    (args.output_dir / "manifest.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"COMPLETE candidate RUN endpoints={len(pair_rows)} association_fit=False", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
