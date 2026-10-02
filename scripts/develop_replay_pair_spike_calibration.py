"""Exercise replay-order prediction using generated spikes and refitted RUN endpoints.

One development realization per generator is not biological calibration. Real
replay validation and an independently frozen validation bank remain required.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts._provenance import build_script_provenance, file_sha256  # noqa: E402
from scripts._replay_pair_prediction import (  # noqa: E402
    PairData, aggregate_event_orders, nuisance_features, prediction_check, require,
)
from scripts._run_pair_rate_glm import crossfit  # noqa: E402
from scripts.audit_replay_order_run_coordination import (  # noqa: E402
    order_asymmetry, stable_seed, whole_bin_shuffles,
)
from scripts.check_replay_pair_prediction_development import write_csv  # noqa: E402
from scripts.measure_run_pair_coordination_endpoint import residual_coordination, strata  # noqa: E402

GENERATORS = ("no_update", "rate_only_drift", "shared_theta_input",
              "existing_coordination_only", "order_specific_update", "regression_to_mean")


def run_covariates(duration, width, offset):
    require(duration > 0 and width > 0 and np.isclose(duration / width, round(duration / width)),
            "Complete physical RUN bins required")
    local = (np.arange(round(duration / width)) + .5) * width
    omega = .6 + .1 * np.sin(2 * np.pi * local / 4)
    angle = .6 * local + .1 * 4 / (2 * np.pi) * (1 - np.cos(2 * np.pi * local / 4))
    return local + offset, {
        "position": 24 + 24 * np.column_stack((np.cos(angle), np.sin(angle))),
        "direction": np.mod(angle + np.pi / 2, 2 * np.pi),
        "speed": 24 * omega,
        "theta": np.angle(np.exp(1j * (2 * np.pi * 8 * local + .2 * np.sin(local))))[:, None],
    }


def immigrant_means(covariates, n_units, width, generator, period, p):
    require(generator in GENERATORS and period in {"pre", "post"}, "Unknown generator/period")
    preference = np.arange(n_units) * 2 * np.pi / n_units
    theta_preference = preference.copy()
    if generator == "shared_theta_input" and period == "post":
        theta_preference[::2] += np.pi / 2
    position = covariates["position"]
    spatial = .3 * ((position[:, 0, None] - 24) * np.cos(preference)
                    + (position[:, 1, None] - 24) * np.sin(preference)) / 24
    direction = .2 * np.cos(covariates["direction"][:, None] - preference)
    speed = .2 * np.log(covariates["speed"][:, None] / 14.4)
    theta = .7 * np.cos(covariates["theta"] - theta_preference)
    mean = width * p["run_base_rate_hz"] * np.exp(spatial + direction + speed + theta)
    if generator == "rate_only_drift" and period == "post":
        mean *= np.where(np.arange(n_units) % 2 == 0, 2, .5)
    require(np.isfinite(mean).all() and (mean > 0).all(), "Invalid simulated spike intensity")
    return mean


def copy_immigrants(immigrants, mean, order, probability, lag, rng):
    immigrants, mean, order = np.asarray(immigrants), np.asarray(mean), np.asarray(order)
    require(immigrants.ndim == 2 and mean.shape == immigrants.shape
            and len(order) == immigrants.shape[1] and set(order) == set(range(len(order))),
            "Copying requires an unchanged complete unit population")
    require(0 <= probability <= 1 and isinstance(lag, int) and 0 < lag < len(mean),
            "Invalid copying probability/physical lag")
    counts, expected = immigrants.copy(), mean.copy()
    # Only immigrant spikes seed offspring: avoid an unmodelled recursive cascade.
    for source, target in zip(order[:-1], order[1:], strict=True):
        counts[lag:, target] += rng.binomial(immigrants[:-lag, source], probability)
        expected[lag:, target] += probability * mean[:-lag, source]
    return counts, expected


def generate_period(covariates, order, generator, period, width, p, rng):
    mean = immigrant_means(covariates, len(order), width, generator, period, p)
    counts = rng.poisson(mean)
    if generator == "existing_coordination_only" or (
            generator == "order_specific_update" and period == "post"):
        lag = round(p["coupling_lag_s"] / width)
        require(np.isclose(lag * width, p["coupling_lag_s"]), "Coupling lag must lie on RUN grid")
        counts, mean = copy_immigrants(counts, mean, order, p["coupling_probability"], lag, rng)
    return counts, mean


def measure_period(counts, times, covariates, estimator):
    labels = strata(covariates["position"], covariates["direction"], covariates["speed"],
                    covariates["theta"], estimator)
    residual, qc, predicted_mean, _ = crossfit(counts, times, labels, estimator["run_bin_s"],
                                             estimator, covariates, return_predictions=True)
    require(np.isfinite(residual).all(), "Generated count bank has an incomplete nuisance fit")
    matrix, opportunities = residual_coordination(residual, times, estimator["run_bin_s"],
                                                   estimator["lag_min_s"], estimator["lag_max_s"])
    require(np.isfinite(matrix).all() and opportunities > 0, "Incomplete generated RUN endpoint")
    return matrix, predicted_mean, {**qc, "physical_lag_opportunities": opportunities}


def make_event(order, p, identity, rng, parent):
    n = round(p["event_duration_s"] / p["event_bin_s"])
    require(np.isclose(n * p["event_bin_s"], p["event_duration_s"]), "Complete event bins required")
    counts = rng.poisson(.05, (n, len(order)))
    for rank, unit in enumerate(order):
        start = 2 + 3 * rank
        require(start + 2 <= n, "Synthetic sequence exceeds frozen event duration")
        counts[start:start + 2, unit] += rng.poisson(5, size=2)
    require((counts.sum(axis=0) > 0).all(), "Synthetic event population unexpectedly silent")
    controls = whole_bin_shuffles(counts, p["n_shuffles"], stable_seed(p["seed"], identity))
    for control in controls:
        require(np.array_equal(counts.sum(axis=0), control.sum(axis=0))
                and sorted(map(tuple, counts)) == sorted(map(tuple, control)),
                "Synthetic shuffle changed snapshots or cell participation")
    low, high = parent["order_min_lag_s"], parent["order_max_lag_s"]
    event = {"event_id": identity, "validated_replay": True,
             "unit_ids": np.arange(len(order)), "spike_counts": counts.sum(axis=0),
             "order": order_asymmetry(counts, p["event_bin_s"], low, high),
             "shuffle_order": np.stack([order_asymmetry(c, p["event_bin_s"], low, high)
                                         for c in controls])}
    return event, counts


def development_case(generator, p, parent, estimator, output):
    fields = {k: [] for k in (*PairData.__dataclass_fields__, "oracle_pre", "oracle_post",
                             "pre_rate_a", "pre_rate_b", "event_spikes_a", "event_spikes_b",
                             "participation")}
    quality, inventory = [], []
    for animal in range(p["animals"]):
        for pause in range(p["pauses_per_animal"]):
            identity = f"simulated_rat{animal}/recording/pause{pause}"
            rng = np.random.default_rng(stable_seed(p["seed"], f"{generator}/{identity}"))
            order = rng.permutation(p["units"])
            periods, saved = {}, {"unit_ids": np.arange(p["units"])}
            for period, offset in (("pre", pause * 100), ("post", pause * 100 + p["run_duration_s"] + 1)):
                times, covariates = run_covariates(p["run_duration_s"], estimator["run_bin_s"], offset)
                period_rng = np.random.default_rng(stable_seed(p["seed"], f"{generator}/{identity}/{period}"))
                counts, mean = generate_period(covariates, order, generator, period,
                                               estimator["run_bin_s"], p, period_rng)
                oracle_residual = (counts - mean) / np.sqrt(mean)
                oracle, opportunities = residual_coordination(oracle_residual, times, estimator["run_bin_s"],
                                                               estimator["lag_min_s"], estimator["lag_max_s"])
                measured, estimated_mean, qc = measure_period(counts, times, covariates, estimator)
                require(opportunities == qc["physical_lag_opportunities"], "Oracle/fitted target support differs")
                periods[period] = {"matrix": measured, "oracle": oracle, "counts": counts}
                saved.update({f"{period}_{k}": value for k, value in {
                    "counts": counts, "time_s": times, "oracle_mean": mean,
                    "estimated_mean": estimated_mean, "coordination": measured,
                    "oracle_coordination": oracle, **covariates}.items()})
                quality.append({"generator": generator, "animal": f"simulated_rat{animal}",
                    "pause": identity, "period": period,
                    **{k: v for k, v in qc.items() if k != "folds"},
                    "fold_diagnostics": json.dumps(qc["folds"])})
            if generator == "regression_to_mean":
                order = np.argsort(-periods["pre"]["oracle"].sum(axis=1), kind="stable")
            events = []
            if animal != 0 or pause != 0:
                event_rng = np.random.default_rng(stable_seed(p["seed"], f"{generator}/{identity}/event"))
                event, event_counts = make_event(order, p, generator + "/" + identity, event_rng, parent)
                events.append(event)
                saved["event_counts"] = event_counts
                saved["event_shuffle_order"] = event["shuffle_order"]
            saved["generating_event_order"] = order
            aggregated = aggregate_event_orders(np.arange(p["units"]), events, p["n_shuffles"])
            a, b = aggregated["unit_a"], aggregated["unit_b"]
            pre, post = periods["pre"]["matrix"][a, b], periods["post"]["matrix"][a, b]
            rates = periods["pre"]["counts"].sum(axis=0) / p["run_duration_s"]
            features = nuisance_features(pre, rates[a], rates[b], aggregated["event_spikes_a"],
                                         aggregated["event_spikes_b"], aggregated["participation"])
            values = {"animal": np.repeat(f"simulated_rat{animal}", len(a)),
                "session": np.repeat(f"simulated_rat{animal}/recording", len(a)),
                "pause": np.repeat(identity, len(a)), "unit_a": a, "unit_b": b,
                "pre": pre, "post": post, "baseline": features, "order": aggregated["order"],
                "oracle_pre": periods["pre"]["oracle"][a, b],
                "oracle_post": periods["post"]["oracle"][a, b],
                "pre_rate_a": rates[a], "pre_rate_b": rates[b],
                "event_spikes_a": aggregated["event_spikes_a"],
                "event_spikes_b": aggregated["event_spikes_b"], "participation": aggregated["participation"]}
            for k, value in values.items():
                fields[k].append(value)
            path = output / f"{generator}_rat{animal}_pause{pause}.npz"
            np.savez_compressed(path, **saved)
            inventory.append({"generator": generator, "animal": f"simulated_rat{animal}", "pause": identity,
                "bank": path.name, "bank_sha256": file_sha256(path), "units": p["units"],
                "synthetic_events": len(events), "run_bins_per_period": len(times)})
            print(f"MEASURED {generator}/{identity}: generated counts, separately refitted PRE/POST", flush=True)
    fields = {k: np.concatenate(value) for k, value in fields.items()}
    data = PairData(**{k: fields[k] for k in PairData.__dataclass_fields__})
    result = prediction_check(data, order_penalty=p["order_penalty"])
    originals = [r["order_coefficient_original_units"] for r in result["folds"] if r["condition"] == 0]
    summary = {"generator": generator, **result["summary"],
        "shuffle_mean_animal_gains": json.dumps(result["summary"]["shuffle_mean_animal_gains"]),
        "mean_training_order_coefficient": float(np.mean(originals)),
        "rms_fitted_minus_oracle_change": float(np.sqrt(np.mean((data.change - (
            fields["oracle_post"] - fields["oracle_pre"])) ** 2))),
        "validation_replicates": 0, "development_realizations": 1}
    np.savez_compressed(output / f"{generator}_predictions.npz", **fields,
                        predictions=result["predictions"], baseline_predictions=result["baseline_predictions"])
    return summary, quality, inventory, [{"generator": generator, **r} for r in result["by_animal"]]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    p = json.loads(args.protocol.read_text())
    inputs = {"spike_protocol": args.protocol, **{key: ROOT / p[key] for key in
              ("parent_protocol", "endpoint_protocol", "prediction_protocol")}}
    parent = json.loads(inputs["parent_protocol"].read_text())
    estimator = json.loads(inputs["endpoint_protocol"].read_text())
    require(tuple(p["generators"]) == GENERATORS == tuple(parent["calibration_required"]),
            "Required generator family changed")
    require(p["n_shuffles"] == parent["order_shuffles"] == 20
            and p["units"] >= parent["minimum_eligible_units"] and p["animals"] >= 3,
            "Frozen population/control dimensions changed")
    require(all(p.get(k) is False for k in ("real_association_enabled", "biological_inference_authorized",
                                          "full_calibration_complete", "goal_complete")),
            "Development cannot authorize biological inference")
    provenance = build_script_provenance(input_paths=inputs)
    require(provenance["git_dirty"] is False and provenance["code_commit"] != "unavailable",
            "Clean committed checkout required")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    summaries, quality, inventory, animals = [], [], [], []
    for generator in GENERATORS:
        s, q, i, a = development_case(generator, p, parent, estimator, args.output_dir)
        summaries.append(s)
        quality.extend(q)
        inventory.extend(i)
        animals.extend(a)
        write_csv(args.output_dir / "development_summary.csv", summaries)
    for name, rows in (("period_quality.csv", quality), ("inventory.csv", inventory), ("animals.csv", animals)):
        write_csv(args.output_dir / name, rows)
    paths = sorted(args.output_dir.iterdir())
    manifest = {**provenance, "protocol_id": p["protocol_id"], "seed": p["seed"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "generators": list(GENERATORS),
        "environment_versions": {"python": sys.version, **{k: version(k) for k in
                                 ("numpy", "scipy", "pandas", "scikit-learn")}},
        "outputs_sha256": {path.name: file_sha256(path) for path in paths},
        "real_recordings_used": False, "real_association_fit": False, "replay_validation_complete": False,
        "validation_replicates": 0, "biological_calibration_complete": False,
        "biological_inference_authorized": False, "goal_complete": False}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print("COMPLETE six spike-count development generators; full calibration=False", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
