"""Independent arithmetic reconstruction of known-mixture and selection summaries."""

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def assert_close(a, b):
    if not np.allclose(a, b, atol=1e-11, rtol=0, equal_nan=True):
        raise ValueError("independent composition reconstruction mismatch")
    e = np.abs(np.asarray(a) - np.asarray(b))
    return float(np.max(e[np.isfinite(e)], initial=0))


def run(report, output):
    if output.exists():
        raise ValueError("new verifier output required")
    m = json.loads((report / "manifest.json").read_text())
    if m["git_dirty"] or not m["non_rescoring"] or m["changes_real_data_selection"]:
        raise ValueError("invalid report provenance")
    for name, h in m["output_sha256"].items():
        if sha(report / name) != h:
            raise ValueError("changed report")
    frames = []
    for name, h in m["source_manifests"].items():
        p = Path(name)
        if sha(p) != h:
            raise ValueError("changed calibration manifest")
        cm = json.loads(p.read_text())
        for f, v in cm["output_sha256"].items():
            if sha(p.parent / f) != v:
                raise ValueError("changed calibration table")
        frames.append(pd.read_csv(p.parent / "known_track_calibration.csv"))
    data = pd.concat(frames, ignore_index=True)
    sign = np.where(data.truth_track == 1, 1, -1)
    data["poisson"] = data.evaluation_track2_probability
    data["conditional_count"] = expit(-sign * data.conditional_true_signed_log_odds)
    keys = ["session", "animal", "cohort_stratum", "epoch", "anchor_ripple_positive"]
    mixture = pd.read_csv(report / "known_selection_interventions.csv")
    actual = pd.read_csv(report / "actual_selection_by_split.csv")
    max_error = 0.0
    n_values = 0
    bootstrap_checks = 0
    groups = dict(tuple(data.groupby(keys)))
    for _, rows in mixture.groupby(keys + ["readout"]):
        r0 = rows.iloc[0]
        key = tuple(r0[k] for k in keys)
        g = groups[key]
        g = g[(g.generator == "ordered") & (g.fraction == 1)]
        anchors = sorted(g.anchor_event.unique())
        split_means = []
        paired = []
        for split in range(5):
            s = g[g.split == split].pivot(index="anchor_event", columns="truth_track", values=r0.readout).reindex(anchors)
            pair = s.to_numpy()
            pair[~np.isfinite(pair).all(axis=1)] = np.nan
            good = np.isfinite(pair).all(axis=1)
            split_means.append(pair[good].mean(axis=0) if good.any() else np.full(2, np.nan))
            paired.append(pair)
        means = np.asarray(split_means)
        ok = np.isfinite(means).all(axis=1)
        before = means[ok].mean() if ok.any() else np.nan
        slope = (means[ok, 1] - means[ok, 0]).mean() if ok.any() else np.nan
        # Rebuild primary calibration intervals from explicit resampled anchor bags.
        boot_slopes = None
        if r0.cohort_stratum == "strict_RUN_pass" and r0.epoch == "POST" and r0.anchor_ripple_positive:
            seed_parts = [20260918, *key, "composition-calibration-anchor-bootstrap"]
            seed = int.from_bytes(hashlib.sha256("|".join(map(str, seed_parts)).encode()).digest()[:8], "little")
            counts = np.random.default_rng(seed).multinomial(len(anchors), np.full(len(anchors), 1 / len(anchors)), size=2000)
            boot_slopes = []
            for w in counts:
                indices = np.repeat(np.arange(len(anchors)), w)
                values = []
                for pair in paired:
                    selected = pair[indices]
                    valid = np.isfinite(selected).all(axis=1)
                    if valid.any():
                        values.append((selected[valid, 1] - selected[valid, 0]).mean())
                boot_slopes.append(np.mean(values) if values else np.nan)
            boot_slopes = np.array(boot_slopes)
        for r in rows.itertuples():
            w1, w2 = r.retain_track1, r.retain_track2
            true = w2 / (w1 + w2) - 0.5
            after = (w1 * means[ok, 0] + w2 * means[ok, 1]).mean() / (w1 + w2) if ok.any() else np.nan
            max_error = max(
                max_error,
                assert_close(
                    [before, after, slope, true, true * slope, true * (slope - 1)],
                    [r.readout_before, r.readout_after, r.recovery_slope, r.true_shift, r.observed_shift, r.shift_error],
                ),
            )
            n_values += 6
            if boot_slopes is not None:
                values = true * boot_slopes
                bounds = np.nanquantile(values, [0.025, 0.975])
                max_error = max(max_error, assert_close(bounds, [r.observed_shift_ci025, r.observed_shift_ci975]))
                bootstrap_checks += 1
    for r in actual.to_dict("records"):
        g = groups[tuple(r[k] for k in keys)]
        g = g[(g.generator == r["generator"]) & (g.split == r["split"])]
        full = g[g.fraction == 1].set_index(["anchor_event", "truth_track"])
        half = g[g.fraction == 0.5].set_index(["anchor_event", "truth_track"]).reindex(full.index)
        truth = np.asarray(full.index.get_level_values("truth_track") == 2)
        q = full[r["readout"]].to_numpy()
        f = full.sequence_accepted.to_numpy()
        h = half.sequence_accepted.to_numpy()
        masks = {"all": np.ones(len(q), bool), "full": f, "half": h, "retained": f & h, "lost": f & ~h, "gained": ~f & h}
        reference = {}
        for label, mask in masks.items():
            good = mask & np.isfinite(q)
            reference["true_" + label + "_all"] = float(truth[mask].mean()) if mask.any() else np.nan
            reference["true_" + label + "_supported"] = float(truth[good].mean()) if good.any() else np.nan
            reference["readout_" + label] = float(q[good].mean()) if good.any() else np.nan
            reference["n_" + label] = int(mask.sum())
            reference["n_" + label + "_supported"] = int(good.sum())
        for label, a, b in [("loss", "retained", "full"), ("gain", "half", "retained"), ("total", "half", "full")]:
            for support in ["all", "supported"]:
                reference["true_" + label + "_shift_" + support] = reference["true_" + a + "_" + support] - reference["true_" + b + "_" + support]
            reference["readout_" + label + "_shift"] = reference["readout_" + a] - reference["readout_" + b]
            reference["error_" + label + "_shift"] = reference["readout_" + label + "_shift"] - reference["true_" + label + "_shift_supported"]
        for name, value in reference.items():
            max_error = max(max_error, assert_close(value, r[name]))
            n_values += 1
    if not len(mixture) or not len(actual) or not bootstrap_checks:
        raise ValueError("vacuous verification")
    result = {
        "status": "pass",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "report_manifest_sha256": sha(report / "manifest.json"),
        "verifier_sha256": sha(Path(__file__)),
        "mixture_rows_checked": len(mixture),
        "actual_selection_rows_checked": len(actual),
        "primary_bootstrap_intervals_reconstructed": bootstrap_checks,
        "numeric_fields_checked": n_values,
        "max_absolute_error": max_error,
        "scope": "all source/output hashes and mixture/actual-selection arithmetic; primary intervals reconstructed via explicit anchor bags; no biological validation",
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    run(a.report_dir, a.output)
