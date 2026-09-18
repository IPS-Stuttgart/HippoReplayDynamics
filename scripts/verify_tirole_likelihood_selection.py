"""Independent paired tensors, anchor intervals and conditional-null reconstruction."""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from scripts.verify_tirole_content_bank import digest, direct_correlations, seed
from scripts.verify_tirole_crossfit_mixture import close
from scripts.verify_tirole_measurement_calibration import draw, path
from scripts.verify_tirole_selection_transport import GENERATORS, GROUPS, STATISTICS, reference_ratios, reference_statistics, validate_stage_manifest


def conditional_posterior(counts, rates, valid):
    """Scalar per-bin/per-track multinomial calculation, not production einsum."""
    lam = np.maximum(rates, 1e-4)
    logp = np.log(lam) - np.log(lam.sum(axis=1, keepdims=True))
    values = np.empty((len(counts), 2, rates.shape[2]))
    for t, c in enumerate(counts):
        for track in range(2):
            values[t, track] = (c[:, None] * logp[track]).sum(axis=0) - np.log(valid[track].sum())
            values[t, track, ~valid[track]] = -np.inf
        values[t] = np.exp(values[t] - logsumexp(values[t]))
    values[counts.sum(axis=1) == 0] = 0.0
    return values


def extended_ratios(cube, weights):
    result = reference_ratios(cube, weights)
    for group in ["full", "half"]:
        result[group + "_signed_distortion"] = result[group + "_true_track2_fraction"] - 0.5
        result[group + "_absolute_distortion"] = np.abs(result[group + "_signed_distortion"])
    return result


def run(experiment, report, output):
    if output.exists():
        raise ValueError("new verification artifact required")
    m = json.loads((experiment / "manifest.json").read_text())
    rm = json.loads((report / "manifest.json").read_text())
    original = Path(m["source_transport_dir"])
    bank = Path(m["bank_dir"])
    for folder, is_bank in [(experiment, False), (report, False), (original, False), (bank, True)]:
        meta = json.loads((folder / "manifest.json").read_text())
        validate_stage_manifest(meta, is_bank)
        for name, h in meta["outputs_sha256" if is_bank else "output_sha256"].items():
            if digest(folder / name) != h:
                raise ValueError("changed hashed artifact")
    if (
        digest(original / "manifest.json") != m["source_transport_manifest_sha256"]
        or digest(bank / "manifest.json") != m["bank_manifest_sha256"]
        or digest(experiment / "manifest.json") != rm["source_manifest_sha256"]
        or m["git_dirty"]
        or rm["git_dirty"]
        or m["real_data_rescored"]
        or m["thresholds_changed"]
    ):
        raise ValueError("provenance mismatch")
    original_audit = json.loads((original.parent / (m["session"] + "_independent_verification.json")).read_text())
    if original_audit["status"] != "pass" or original_audit["experiment_manifest_sha256"] != digest(original / "manifest.json"):
        raise ValueError("original simulated observations not independently verified")
    anchors = pd.read_csv(experiment / "count_anchors.csv").sort_values("event_id")
    if digest(experiment / "count_anchors.csv") != digest(original / "count_anchors.csv") or len(anchors) != 40:
        raise ValueError("anchors changed")
    keys = ["anchor_event", "split", "draw", "truth_track", "generator", "repeat"]
    ix = pd.MultiIndex.from_product([anchors.event_id, range(5), range(20), [1, 2], GENERATORS, range(-1, 5)], names=keys)
    scores = {}
    for name, folder in [("poisson", original), ("conditional_count", experiment)]:
        frame = pd.concat([pd.read_csv(p) for p in sorted((folder / "sequence_shards").glob("*.csv"))], ignore_index=True)
        if len(frame) != len(ix) or frame.duplicated(keys).any():
            raise ValueError("incomplete score matrix")
        frame = frame.set_index(keys).reindex(ix)
        if frame.n_inference_spikes.isna().any():
            raise ValueError("missing score keys")
        scores[name] = frame
    fixed = [
        "session",
        "animal",
        "epoch",
        "ripple_positive",
        "true_path_span_cm",
        "fraction",
        "n_inference_cells",
        "n_inference_spikes",
        "n_time_bins",
        "n_active_inference",
        "n_nonempty_bins",
        "sequence_eligible",
    ]
    pd.testing.assert_frame_equal(scores["poisson"][fixed], scores["conditional_count"][fixed], check_exact=False, atol=1e-12, rtol=1e-12)
    content = pd.concat([pd.read_csv(p) for p in sorted((original / "content_shards").glob("*.csv"))], ignore_index=True)
    cx = pd.MultiIndex.from_product([anchors.event_id, range(5), range(20), [1, 2]], names=keys[:4])
    if len(content) != len(cx) or content.duplicated(keys[:4]).any():
        raise ValueError("missing evaluation matrix")
    content = content.set_index(keys[:4]).reindex(cx)
    z = content[["poisson_true_signed_z", "conditional_count_true_signed_z"]].to_numpy().reshape(40, 5, 20, 2, 2)
    shape = (40, 5, 20, 2, 2, 6)
    saved_anchor = pd.read_csv(report / "anchor_statistics.csv")
    summary = pd.read_csv(report / "likelihood_condition_summary.csv")
    intervals = pd.read_csv(report / "primary_anchor_uncertainty.csv")
    paired = pd.read_csv(report / "paired_likelihood_contrasts.csv")
    checks = {"unchanged_score_opportunities": len(ix), "anchor_statistics": 0, "summary_metrics": 0, "likelihood_intervals": 0, "paired_intervals": 0, "conditional_null_cases": 0}
    error = 0.0
    cube_by_likelihood = {}
    for name, frame in scores.items():
        accepted = frame.sequence_accepted.to_numpy(bool).reshape(shape)
        labels = frame.inferred_track.to_numpy(int).reshape(shape)
        for partition, draw_ids in [("all", np.arange(20)), ("even", np.arange(0, 20, 2)), ("odd", np.arange(1, 20, 2))]:
            cube = reference_statistics(accepted, labels, z, draw_ids)
            if partition == "all":
                cube_by_likelihood[name] = cube
            saved = saved_anchor[saved_anchor.likelihood.eq(name) & saved_anchor.draw_partition.eq(partition)]
            ai = pd.MultiIndex.from_product([anchors.event_id, [1, 2], GENERATORS, GROUPS], names=["anchor_event", "truth_track", "generator", "group"])
            if len(saved) != len(ai) or saved.duplicated(ai.names).any():
                raise ValueError("missing anchor statistic keys")
            error = max(error, close(cube.reshape(-1, 7), saved.set_index(ai.names).reindex(ai)[list(STATISTICS)].to_numpy()))
            checks["anchor_statistics"] += len(ai)
            for (epoch, ripple), a in anchors.groupby(["epoch", "primary_ripple_candidate"]):
                chosen = np.flatnonzero(anchors.event_id.isin(a.event_id))
                for g, generator in enumerate(GENERATORS):
                    point = extended_ratios(cube[chosen, :, g], np.ones((1, len(chosen))))
                    saved = summary[
                        summary.likelihood.eq(name)
                        & summary.draw_partition.eq(partition)
                        & summary.epoch.eq(epoch)
                        & summary.ripple_positive.eq(ripple)
                        & summary.generator.eq(generator)
                    ]
                    if len(saved) != 1 or saved.n_original_anchors.iloc[0] != len(chosen):
                        raise ValueError("missing summary/denominator")
                    for metric, value in point.items():
                        error = max(error, close(value[0], saved[metric].iloc[0]))
                        checks["summary_metrics"] += 1
    chosen = np.flatnonzero((anchors.epoch == "POST") & anchors.primary_ripple_candidate)
    n = len(chosen)
    for g, generator in enumerate(GENERATORS):
        weights = np.random.default_rng(seed(20260918, m["session"], generator, "selection-transport-anchor-bootstrap")).multinomial(n, np.full(n, 1 / n), size=2000)
        points = {}
        boots = {}
        for name, cube in cube_by_likelihood.items():
            sub = cube[chosen, :, g]
            points[name] = extended_ratios(sub, np.ones((1, n)))
            boots[name] = extended_ratios(sub, weights)
            for metric, value in points[name].items():
                draws = boots[name][metric]
                finite = draws[np.isfinite(draws)]
                lo, hi = np.quantile(finite, [0.025, 0.975]) if len(finite) >= 1900 and n >= 2 else [np.nan, np.nan]
                row = intervals[intervals.generator.eq(generator) & intervals.likelihood.eq(name) & intervals.metric.eq(metric)]
                if len(row) != 1:
                    raise ValueError("missing likelihood interval")
                error = max(error, close([value[0], lo, hi, len(finite) / 2000], row[["estimate", "ci025", "ci975", "finite_bootstrap_fraction"]].to_numpy()[0]))
                checks["likelihood_intervals"] += 1
        expected = {
            k + "_conditional_minus_poisson": (points["conditional_count"][k] - points["poisson"][k], boots["conditional_count"][k] - boots["poisson"][k])
            for k in points["poisson"]
        }
        for group in ["full", "half"]:
            k = group + "_absolute_distortion"
            expected[group + "_absolute_distortion_reduction"] = (points["poisson"][k] - points["conditional_count"][k], boots["poisson"][k] - boots["conditional_count"][k])
        for metric, (value, draws) in expected.items():
            finite = draws[np.isfinite(draws)]
            lo, hi = np.quantile(finite, [0.025, 0.975]) if len(finite) >= 1900 and n >= 2 else [np.nan, np.nan]
            row = paired[paired.generator.eq(generator) & paired.metric.eq(metric)]
            if len(row) != 1:
                raise ValueError("missing paired interval")
            error = max(error, close([value[0], lo, hi, len(finite) / 2000], row[["estimate", "ci025", "ci975", "finite_bootstrap_fraction"]].to_numpy()[0]))
            checks["paired_intervals"] += 1
    maps = np.load(bank / "RUN_maps.npz")
    arrays = np.load(bank / "event_counts.npz")
    parts = json.loads((bank / "partitions.json").read_text())["splits"]
    frame = scores["conditional_count"].reset_index()
    sample = frame[frame["repeat"].isin([-1, 2]) & frame.sequence_eligible]
    cases = []
    for (generator, repeat, accepted), a in sample.groupby(["generator", "repeat", "sequence_accepted"]):
        row = a.iloc[0]
        eid, split, mc, track = map(int, [row.anchor_event, row["split"], row["draw"], row.truth_track - 1])
        original_counts = arrays["counts"][arrays["offsets"][eid] : arrays["offsets"][eid + 1]]
        inference = np.array(parts[split]["inference"])
        evaluation = np.array(parts[split]["evaluation"])
        positions = path(maps["valid_bins"], len(original_counts), np.random.default_rng(seed(20260918, m["session"], eid, split, "known-path")))
        rng = np.random.default_rng(seed(20260918, m["session"], eid, split, mc, track, "selection-transport-spikes-v1"))
        generated = np.zeros_like(original_counts)
        for ids in [inference, evaluation]:
            generated[:, ids] = draw(original_counts[:, ids].sum(axis=1), np.maximum(maps["rates"][track][:, positions].T[:, ids], 1e-4), rng)
        rng.integers(0, 2, (199, len(evaluation)))
        permutation = rng.permutation(len(generated))
        counts = generated if generator == "ordered" else generated[permutation]
        order = np.random.default_rng(seed(20260918, m["session"], split, repeat, "coverage")).permutation(inference)
        ids = inference if repeat == -1 else np.sort(order[: int(np.ceil(len(inference) * 0.5))])
        cc = counts[:, ids]
        if row.n_inference_spikes != cc.sum():
            raise ValueError("regenerated spike count differs")
        nrng = np.random.default_rng(seed(20260918, m["session"], eid, split, mc, track, generator, "selection-transport-nulls-v1"))
        shifts = nrng.integers(0, len(maps["bin_centers_cm"]), (499, 2, counts.shape[1]))
        perms = np.argsort(nrng.random((499, len(cc))), axis=1)
        lam = maps["rates"][:, ids]
        p = conditional_posterior(cc, lam, maps["valid_bins"])
        corr = direct_correlations(p, maps["bin_centers_cm"])
        temporal = np.array([direct_correlations(p[perm], maps["bin_centers_cm"]) for perm in perms])
        field = []
        for shift in shifts:
            rolled = np.array([[np.roll(lam[k, j], shift[k, uid]) for j, uid in enumerate(ids)] for k in range(2)])
            field.append(direct_correlations(conditional_posterior(cc, rolled, maps["valid_bins"]), maps["bin_centers_cm"]))
        pt = (1 + (temporal >= corr - 1e-12).sum(axis=0)) / 500
        pf = (1 + (np.array(field) >= corr - 1e-12).sum(axis=0)) / 500
        passed = (pt < 0.025) & (pf < 0.025)
        best = int(np.argmax(np.where(passed, corr, -np.inf))) if passed.any() else int(np.argmax(corr))
        error = max(
            error, close(np.r_[pt, pf, corr[best]], row[["track1_p_time", "track2_p_time", "track1_p_field", "track2_p_field", "best_weighted_correlation"]].to_numpy(float))
        )
        if row.sequence_accepted != passed.any() or row.inferred_track != best + 1:
            raise ValueError("independent conditional selection differs")
        cases.append({"generator": generator, "repeat": int(repeat), "accepted": bool(accepted), "anchor": eid, "split": split, "draw": mc, "track": track + 1})
        checks["conditional_null_cases"] += 1
    if not all(checks.values()) or checks["anchor_statistics"] != 4800 or checks["conditional_null_cases"] != len(sample.groupby(["generator", "repeat", "sequence_accepted"])):
        raise ValueError("incomplete independent coverage")
    result = {
        "status": "pass",
        "session": m["session"],
        "created_at_utc": datetime.now(UTC).isoformat(),
        "experiment_manifest_sha256": digest(experiment / "manifest.json"),
        "report_manifest_sha256": digest(report / "manifest.json"),
        "verifier_sha256": digest(Path(__file__)),
        "checks": checks,
        "max_absolute_error": error,
        "null_cases": cases,
        "scope": "identical source opportunities; both likelihood anchor tensors, all summary/paired intervals; independently regenerated conditional shuffle examples; not biological validation",
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment-dir", type=Path, required=True)
    p.add_argument("--report-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    run(a.experiment_dir, a.report_dir, a.output)
