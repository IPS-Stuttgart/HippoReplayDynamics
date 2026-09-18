"""Independent tensor aggregation and stratified spike/sequence reconstruction."""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.verify_tirole_content_bank import digest, direct_correlations, direct_posterior, seed
from scripts.verify_tirole_crossfit_mixture import close
from scripts.verify_tirole_measurement_calibration import direct_content, draw, path

GROUPS = ("full", "half", "retained", "lost", "gained")
GENERATORS = ("ordered", "whole_bin_shuffled")
STATISTICS = ("group_mass", "label2_mass", "correct_label_mass", "poisson_z_numerator", "poisson_z_denominator", "conditional_count_z_numerator", "conditional_count_z_denominator")


def validate_stage_manifest(meta, is_bank=False):
    if is_bank:
        required = {"candidate_events.csv", "partitions.json", "RUN_maps.npz", "event_counts.npz"}
        if (
            meta.get("strict_RUN_preflight_passed") is not True
            or meta.get("git_dirty") is not False
            or meta.get("n_candidates", 0) <= 0
            or set(meta.get("outputs_sha256", {})) != required
        ):
            raise ValueError("invalid primary input bank")
    elif meta.get("status") != "complete":
        raise ValueError("incomplete input")


def reference_statistics(accepted, labels, z, draw_ids):
    """Inputs anchor x split x draw x truth x generator x coverage-condition."""
    accepted = accepted[:, :, draw_ids]
    labels = labels[:, :, draw_ids]
    z = z[:, :, draw_ids]
    full = np.broadcast_to(accepted[..., :1], accepted[..., 1:].shape)
    half = accepted[..., 1:]
    truth = np.array([1, 2])[None, None, None, :, None, None]
    out = []
    for group in GROUPS:
        use = {"full": full, "half": half, "retained": full & half, "lost": full & ~half, "gained": ~full & half}[group]
        inferred = labels[..., :1] if group in ("full", "retained", "lost") else labels[..., 1:]
        columns = [use, use & (inferred == 2), use & (inferred == truth)]
        for readout in range(2):
            value = z[..., readout, None, None]
            available = use & np.isfinite(value)
            columns.extend([np.where(available, value, 0.0), available])
        out.append(np.stack([x.mean(axis=(1, 2, 5)) for x in columns], axis=-1))
    return np.stack(out, axis=-2)  # anchor x truth x generator x group x statistic


def reference_ratios(cube, weights):
    """Vectorized original-anchor bootstrap, independent of the pandas reporter."""
    weights = np.atleast_2d(weights)
    total = np.einsum("da,atgs->dtgs", weights, cube)
    base = weights.sum(axis=1)

    def divide(a, b):
        return np.divide(a, b, out=np.full_like(a, np.nan, dtype=float), where=b > 0)

    result = {}
    for g, name in enumerate(GROUPS):
        mass = total[:, :, g, 0].sum(axis=1)
        result[name + "_true_track2_fraction"] = divide(total[:, 1, g, 0], mass)
        for t in [1, 2]:
            result[name + f"_acceptance_track{t}"] = divide(total[:, t - 1, g, 0], base)
        for stat, label in [(1, "reported_track2_fraction"), (2, "correct_label_fraction")]:
            result[name + "_" + label] = divide(total[:, :, g, stat].sum(axis=1), mass)
        for stat, readout in [(3, "poisson"), (5, "conditional_count")]:
            result[name + "_" + readout + "_true_signed_z"] = divide(total[:, :, g, stat].sum(axis=1), total[:, :, g, stat + 1].sum(axis=1))
    for g in GROUPS[1:]:
        result[g + "_minus_full_true_track2_fraction"] = result[g + "_true_track2_fraction"] - result["full_true_track2_fraction"]
    for t in [1, 2]:
        result[f"loss_given_full_track{t}"] = divide(result[f"lost_acceptance_track{t}"], result[f"full_acceptance_track{t}"])
    return result


def run(experiment, report, output):
    if output.exists():
        raise ValueError("new verification artifact required")
    m = json.loads((experiment / "manifest.json").read_text())
    rm = json.loads((report / "manifest.json").read_text())
    bank = Path(m["bank_dir"])
    original_calibration = Path(m["calibration_dir"])
    for base, key in [(experiment, "output_sha256"), (report, "output_sha256"), (bank, "outputs_sha256"), (original_calibration, "output_sha256")]:
        meta = json.loads((base / "manifest.json").read_text())
        validate_stage_manifest(meta, is_bank=base == bank)
        for name, h in meta[key].items():
            if digest(base / name) != h:
                raise ValueError("changed hashed input")
    if (
        m["git_dirty"]
        or m["real_data_rescored"]
        or rm["source_manifest_sha256"] != digest(experiment / "manifest.json")
        or m["bank_manifest_sha256"] != digest(bank / "manifest.json")
        or m["calibration_manifest_sha256"] != digest(original_calibration / "manifest.json")
    ):
        raise ValueError("broken frozen provenance")
    anchors = pd.read_csv(experiment / "count_anchors.csv").sort_values("event_id")
    old = pd.read_csv(original_calibration / "count_anchors.csv").sort_values("event_id")
    pd.testing.assert_frame_equal(anchors, old, check_exact=False)
    scores = pd.concat([pd.read_csv(p) for p in sorted((experiment / "sequence_shards").glob("*.csv"))], ignore_index=True)
    content = pd.concat([pd.read_csv(p) for p in sorted((experiment / "content_shards").glob("*.csv"))], ignore_index=True)
    keys = ["anchor_event", "split", "draw", "truth_track", "generator", "repeat"]
    ix = pd.MultiIndex.from_product([anchors.event_id, range(5), range(20), [1, 2], GENERATORS, range(-1, 5)], names=keys)
    if len(scores) != len(ix) or scores.duplicated(keys).any():
        raise ValueError("incomplete sequence matrix")
    s = scores.set_index(keys).reindex(ix)
    if s.fraction.isna().any():
        raise ValueError("missing sequence keys")
    cx = pd.MultiIndex.from_product([anchors.event_id, range(5), range(20), [1, 2]], names=keys[:4])
    if len(content) != len(cx) or content.duplicated(keys[:4]).any():
        raise ValueError("incomplete content matrix")
    c = content.set_index(keys[:4]).reindex(cx)
    if c.n_evaluation_spikes.isna().any():
        raise ValueError("missing content keys")
    shape = (len(anchors), 5, 20, 2, 2, 6)
    accepted = s.sequence_accepted.to_numpy(bool).reshape(shape)
    labels = s.inferred_track.to_numpy(int).reshape(shape)
    z = c[["poisson_true_signed_z", "conditional_count_true_signed_z"]].to_numpy().reshape(len(anchors), 5, 20, 2, 2)
    saved_anchor = pd.read_csv(report / "anchor_statistics.csv")
    saved_summary = pd.read_csv(report / "selection_summary.csv")
    saved_ci = pd.read_csv(report / "primary_anchor_uncertainty.csv")
    checks = {"anchor_statistics": 0, "summary_metrics": 0, "intervals": 0, "regenerated_contents": 0, "regenerated_sequence_opportunities": 0, "independent_sequence_nulls": 0}
    maxerr = 0.0
    for partition, draw_ids in [("all", np.arange(20)), ("even", np.arange(0, 20, 2)), ("odd", np.arange(1, 20, 2))]:
        cube = reference_statistics(accepted, labels, z, draw_ids)
        for a, eid in enumerate(anchors.event_id):
            for t in range(2):
                for g, gen in enumerate(GENERATORS):
                    for j, group in enumerate(GROUPS):
                        r = saved_anchor[
                            (saved_anchor.anchor_event == eid)
                            & saved_anchor.truth_track.eq(t + 1)
                            & saved_anchor.generator.eq(gen)
                            & saved_anchor.draw_partition.eq(partition)
                            & saved_anchor.group.eq(group)
                        ]
                        if len(r) != 1:
                            raise ValueError("incomplete anchor aggregation")
                        maxerr = max(maxerr, close(cube[a, t, g, j], r[list(STATISTICS)].to_numpy()[0]))
                        checks["anchor_statistics"] += 1
        for (epoch, ripple), group in anchors.groupby(["epoch", "primary_ripple_candidate"]):
            indices = np.flatnonzero(anchors.event_id.isin(group.event_id))
            for g, gen in enumerate(GENERATORS):
                subset = cube[indices, :, g]
                values = reference_ratios(subset, np.ones((1, len(indices))))
                r = saved_summary[
                    saved_summary.epoch.eq(epoch) & saved_summary.ripple_positive.eq(ripple) & saved_summary.generator.eq(gen) & saved_summary.draw_partition.eq(partition)
                ]
                if len(r) != 1 or r.n_original_anchors.iloc[0] != len(indices):
                    raise ValueError("summary key or denominator mismatch")
                for metric, value in values.items():
                    maxerr = max(maxerr, close(value[0], r[metric].iloc[0]))
                    checks["summary_metrics"] += 1
                if epoch == "POST" and ripple and partition == "all":
                    n = len(indices)
                    weights = np.random.default_rng(seed(20260918, m["session"], gen, "selection-transport-anchor-bootstrap")).multinomial(n, np.full(n, 1 / n), size=2000)
                    bs = reference_ratios(subset, weights)
                    for metric, draws in bs.items():
                        finite = draws[np.isfinite(draws)]
                        lo, hi = np.quantile(finite, [0.025, 0.975]) if len(finite) >= 1900 and n >= 2 else [np.nan, np.nan]
                        row = saved_ci[saved_ci.generator.eq(gen) & saved_ci.metric.eq(metric)]
                        if len(row) != 1:
                            raise ValueError("missing/duplicate interval")
                        maxerr = max(maxerr, close([values[metric][0], lo, hi, len(finite) / 2000], row[["estimate", "ci025", "ci975", "finite_bootstrap_fraction"]].to_numpy()[0]))
                        checks["intervals"] += 1
    arrays = np.load(bank / "event_counts.npz")
    maps = np.load(bank / "RUN_maps.npz")
    parts = json.loads((bank / "partitions.json").read_text())["splits"]
    chosen = set(anchors.groupby(["epoch", "primary_ripple_candidate"]).head(1).event_id)
    contexts = {(eid, split, mc) for eid in chosen for split in [0, 4] for mc in [0, 19]}
    available = scores[scores["repeat"].isin([-1, 2]) & scores.sequence_eligible]
    for _, group in available.groupby(["generator", "repeat", "sequence_accepted"]):
        representative = group.iloc[0]
        contexts.add((int(representative.anchor_event), int(representative["split"]), int(representative["draw"])))
    chosen = {x[0] for x in contexts}
    null_checked_groups = set()
    for eid in sorted(chosen):
        observed = arrays["counts"][arrays["offsets"][eid] : arrays["offsets"][eid + 1]]
        for split in sorted({x[1] for x in contexts if x[0] == eid}):
            a, b = np.asarray(parts[split]["inference"]), np.asarray(parts[split]["evaluation"])
            if set(a) & set(b):
                raise ValueError("cell-role overlap")
            positions = path(maps["valid_bins"], len(observed), np.random.default_rng(seed(20260918, m["session"], eid, split, "known-path")))
            for mc in sorted({x[2] for x in contexts if x[0] == eid and x[1] == split}):
                for t in range(2):
                    rng = np.random.default_rng(seed(20260918, m["session"], eid, split, mc, t, "selection-transport-spikes-v1"))
                    generated = np.zeros_like(observed)
                    for ids in (a, b):
                        generated[:, ids] = draw(observed[:, ids].sum(axis=1), np.maximum(maps["rates"][t][:, positions].T[:, ids], 1e-4), rng)
                    swaps = rng.integers(0, 2, (199, len(b))).astype(bool)
                    r = c.loc[(eid, split, mc, t + 1)]
                    if r.n_evaluation_spikes != generated[:, b].sum():
                        raise ValueError("regenerated B counts mismatch")
                    for conditional, name in [(False, "poisson"), (True, "conditional_count")]:
                        odds, standardized = direct_content(generated[:, b], maps["rates"][:, b], maps["valid_bins"], swaps, conditional)
                        probability = 1 / (1 + np.exp(odds)) if np.isfinite(odds) else np.nan
                        maxerr = max(maxerr, close([probability, (1 - 2 * t) * standardized], [r[name + "_track2_probability"], r[name + "_true_signed_z"]]))
                    checks["regenerated_contents"] += 1
                    permutation = rng.permutation(len(generated))
                    for gen, counts in [("ordered", generated), ("whole_bin_shuffled", generated[permutation])]:
                        nrng = np.random.default_rng(seed(20260918, m["session"], eid, split, mc, t, gen, "selection-transport-nulls-v1"))
                        shifts = nrng.integers(0, len(maps["bin_centers_cm"]), (499, 2, counts.shape[1]))
                        perms = np.argsort(nrng.random((499, len(counts))), axis=1)
                        for repeat in [-1, 2]:
                            order = np.random.default_rng(seed(20260918, m["session"], split, repeat, "coverage")).permutation(a)
                            ids = a if repeat == -1 else np.sort(order[: int(np.ceil(len(a) * 0.5))])
                            cc = counts[:, ids]
                            r = s.loc[(eid, split, mc, t + 1, gen, repeat)]
                            nonempty = cc.sum(axis=1) > 0
                            active = (cc.sum(axis=0) > 0).sum()
                            eligible = active >= 5 and nonempty.sum() >= 5
                            if r.n_inference_spikes != cc.sum() or r.n_inference_cells != len(ids) or r.sequence_eligible != eligible:
                                raise ValueError("opportunity/subset mismatch")
                            checks["regenerated_sequence_opportunities"] += 1
                            coverage = (gen, repeat, bool(r.sequence_accepted))
                            if eligible and coverage not in null_checked_groups:
                                lam = maps["rates"][:, ids]
                                p = direct_posterior(cc, lam, maps["valid_bins"])
                                p[~nonempty] = 0.0
                                corr = direct_correlations(p, maps["bin_centers_cm"])
                                temporal = np.array([direct_correlations(p[permutation], maps["bin_centers_cm"]) for permutation in perms])
                                field = []
                                for shift in shifts:
                                    rolled = np.array([[np.roll(lam[k, j], shift[k, uid]) for j, uid in enumerate(ids)] for k in range(2)])
                                    pp = direct_posterior(cc, rolled, maps["valid_bins"])
                                    pp[~nonempty] = 0.0
                                    field.append(direct_correlations(pp, maps["bin_centers_cm"]))
                                pt = (1 + (temporal >= corr - 1e-12).sum(axis=0)) / 500
                                pf = (1 + (np.array(field) >= corr - 1e-12).sum(axis=0)) / 500
                                maxerr = max(maxerr, close(np.r_[pt, pf], r[["track1_p_time", "track2_p_time", "track1_p_field", "track2_p_field"]].to_numpy(float)))
                                passed = (pt < 0.025) & (pf < 0.025)
                                best = np.argmax(np.where(passed, corr, -np.inf)) if passed.any() else np.argmax(corr)
                                if r.sequence_accepted != bool(passed.any()) or r.inferred_track != best + 1:
                                    raise ValueError("sequence selection mismatch")
                                null_checked_groups.add(coverage)
                                checks["independent_sequence_nulls"] += 1
    expected_null_groups = {tuple(x) for x in available[["generator", "repeat", "sequence_accepted"]].drop_duplicates().to_numpy()}
    if null_checked_groups != expected_null_groups:
        raise ValueError("missing available acceptance-class/null-family coverage")
    if checks["anchor_statistics"] != len(anchors) * 2 * 2 * 5 * 3 or not checks["regenerated_contents"]:
        raise ValueError("vacuous verification")
    result = {
        "status": "pass",
        "session": m["session"],
        "created_at_utc": datetime.now(UTC).isoformat(),
        "experiment_manifest_sha256": digest(experiment / "manifest.json"),
        "report_manifest_sha256": digest(report / "manifest.json"),
        "verifier_sha256": digest(Path(__file__)),
        "checks": checks,
        "max_absolute_error": maxerr,
        "null_test_cases_covered": [{"generator": gen, "repeat": int(rep), "accepted": bool(accepted)} for gen, rep, accepted in sorted(null_checked_groups)],
        "scope": "all hashed inputs and tensor-based original-anchor statistics/intervals; stratified independently regenerated spikes/content/opportunities and covered null-test cases; no biological validation",
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
