"""Independent RUN subset, fixed-B, sequence-null and block-contrast verification."""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

from scripts.verify_tirole_content_bank import digest, direct_correlations, direct_posterior, seed
from scripts.verify_tirole_crossfit_mixture import close

ARMS = ["full", "track1_information", "track2_information"] + [f"random_{r}_{s}" for r in range(5) for s in ["a", "b"]]
CONTRIBUTIONS = ["acceptance", "eligible", "label2", "q_num", "q_den", "z_num", "z_den", "duration_num", "evalspikes_num", "infspikes_num"]


def reference_information(rates, occupancy, valid):
    information = np.zeros(rates.shape[:2])
    means = np.zeros(rates.shape[:2])
    for track in range(2):
        keep = valid[track] & (occupancy[track] > 0)
        weight = occupancy[track, keep] / occupancy[track, keep].sum()
        for unit in range(rates.shape[1]):
            lam = rates[track, unit, keep]
            average = float(weight @ lam)
            if average > 0:
                positive = lam > 0
                relative = lam[positive] / average
                information[track, unit] = max(0.0, float((weight[positive] * relative) @ np.log2(relative)))
            means[track, unit] = rates[track, unit] @ occupancy[track] / occupancy[track].sum()
    return information, means.mean(axis=0)


def reference_subsets(ids, information, rates, session, split):
    ordered = sorted(ids, key=lambda i: (rates[i], seed(20260918, session, split, int(i), "RUN-rate-pair-tie")))
    pairs = [ordered[i : i + 2] for i in range(0, len(ids) - 1, 2)]
    shared = ordered[-1:] if len(ids) % 2 else []
    first, second = [], []
    for pair in pairs:
        rank = sorted(pair, key=lambda i: (information[1, i] - information[0, i], seed(20260918, session, split, int(i), "RUN-information-tie")))
        first.append(rank[0])
        second.append(rank[1])
    arms = {"full": sorted(ids), "track1_information": sorted(first + shared), "track2_information": sorted(second + shared)}
    for repeat in range(5):
        flips = np.random.default_rng(seed(20260918, session, split, repeat, "rate-paired-random-half-v1")).integers(0, 2, len(pairs))
        arms[f"random_{repeat}_a"] = sorted([p[f] for p, f in zip(pairs, flips, strict=True)] + shared)
        arms[f"random_{repeat}_b"] = sorted([p[1 - f] for p, f in zip(pairs, flips, strict=True)] + shared)
    return arms, pairs, shared


def reference_cube(accepted, eligible, labels, q, z, duration, evaluation_spikes, inference_spikes):
    """Event x split x arm inputs, averaged within original event only."""
    cols = [accepted, eligible, accepted & (labels == 2)]
    for values in [q, z]:
        good = accepted & np.isfinite(values)
        cols.extend([np.where(good, values, 0.0), good])
    cols.extend(np.where(accepted, values, 0.0) for values in [duration, evaluation_spikes, inference_spikes])
    return np.stack(cols, axis=-1).mean(axis=1)


def reference_statistics(cube, weights):
    total = np.einsum("we,eas->was", np.atleast_2d(weights), cube)

    def ratio(a, b):
        return np.divide(a, b, out=np.full_like(a, np.nan, dtype=float), where=b > 0)

    return {
        "acceptance_fraction": ratio(total[..., 0], np.atleast_2d(weights).sum(axis=1)[:, None]),
        "eligible_fraction": ratio(total[..., 1], np.atleast_2d(weights).sum(axis=1)[:, None]),
        "selected_event_equivalents": total[..., 0],
        "classifier_track2_fraction": ratio(total[..., 2], total[..., 0]),
        "conditional_track2_score_mean": ratio(total[..., 3], total[..., 4]),
        "conditional_track2_identity_z_mean": ratio(total[..., 5], total[..., 6]),
        "mean_duration_s": ratio(total[..., 7], total[..., 0]),
        "mean_evaluation_spikes": ratio(total[..., 8], total[..., 0]),
        "mean_inference_spikes": ratio(total[..., 9], total[..., 0]),
    }


def hashed_manifest(folder, bank=False):
    m = json.loads((folder / "manifest.json").read_text())
    if bank:
        if not m["strict_RUN_preflight_passed"] or m["git_dirty"]:
            raise ValueError("invalid primary bank")
    elif m["status"] != "complete" or m.get("git_dirty", False):
        raise ValueError("unfinished/dirty source")
    for name, checksum in m["outputs_sha256" if bank else "output_sha256"].items():
        if digest(folder / name) != checksum:
            raise ValueError("changed hashed source")
    return m


def run(experiment, report, output):
    if output.exists():
        raise ValueError("new verification artifact required")
    m = hashed_manifest(experiment)
    rm = hashed_manifest(report)
    bank = Path(m["bank_dir"])
    control = Path(m["evaluation_control_dir"])
    bm = hashed_manifest(bank, True)
    cm = hashed_manifest(control)
    if (
        digest(bank / "manifest.json") != m["bank_manifest_sha256"]
        or digest(control / "manifest.json") != m["evaluation_manifest_sha256"]
        or digest(experiment / "manifest.json") != rm["source_manifest_sha256"]
    ):
        raise ValueError("input provenance mismatch")
    rows = pd.read_csv(experiment / "event_coverage_scores.csv")
    events = pd.read_csv(bank / "candidate_events.csv")
    events = events[events.ripple_supported].sort_values("event_id")
    session = bm["session"]
    keys = ["event_id", "split", "arm"]
    ix = pd.MultiIndex.from_product([events.event_id, range(5), ARMS], names=keys)
    if len(rows) != len(ix) or rows.duplicated(keys).any() or set(rows.session) != {session} or events.event_id.tolist() != m["selected_event_ids"]:
        raise ValueError("incomplete score identities")
    s = rows.set_index(keys).reindex(ix)
    if s.n_inference_units.isna().any():
        raise ValueError("missing arm or event")
    parts = json.loads((bank / "partitions.json").read_text())
    maps = np.load(bank / "RUN_maps.npz")
    arrays = np.load(bank / "event_counts.npz")
    info, rates = reference_information(maps["rates"], maps["occupancy_s"], maps["valid_bins"])
    maxerr = 0.0
    checks = {
        "RUN_subsets": 0,
        "fixed_B_rows": 0,
        "sequence_opportunities": 0,
        "sequence_null_reconstructions": 0,
        "original_full_rows": 0,
        "event_contributions": 0,
        "arm_metrics": 0,
        "contrasts_and_intervals": 0,
    }
    unit = pd.read_csv(experiment / "RUN_unit_information.csv").sort_values("unit_index")
    maxerr = max(maxerr, close(np.c_[rates, info.T], unit[["RUN_rate_hz", "track1_information_bits_per_spike", "track2_information_bits_per_spike"]].to_numpy()))
    stored_subsets = pd.read_csv(experiment / "RUN_only_subsets.csv").set_index(["split", "arm"])
    pair_table = pd.read_csv(experiment / "RUN_rate_pairs.csv")
    btable = pd.read_csv(control / "evaluation_identity_by_event_split.csv")
    btable = btable[btable.session == session].set_index(["event_id", "split"])
    subsets = {}
    for split, part in enumerate(parts["splits"]):
        arms, pairs, shared = reference_subsets(part["inference"], info, rates, session, split)
        subsets[split] = arms
        for k, pair in enumerate(pairs):
            p = pair_table[pair_table.split.eq(split) & pair_table.pair.eq(k)]
            if len(p) != 1 or p.first_unit.iloc[0] != pair[0] or p.second_unit.iloc[0] != pair[1] or json.loads(p.shared_singleton.iloc[0]) != shared:
                raise ValueError("RUN pairing mismatch")
            maxerr = max(maxerr, close(abs(np.log(rates[pair[0]]) - np.log(rates[pair[1]])), p.log_RUN_rate_gap.iloc[0]))
        for arm, ids in arms.items():
            row = stored_subsets.loc[(split, arm)]
            if json.loads(row.unit_indices) != ids or row.n_units != len(ids) or set(ids) & (set(part["evaluation"]) | set(parts["detector"])):
                raise ValueError("subset/role leakage")
            maxerr = max(
                maxerr,
                close(
                    [rates[ids].mean(), info[0, ids].mean(), info[1, ids].mean(), (info[0, ids] - info[1, ids]).mean()],
                    row[["mean_RUN_rate_hz", "mean_track1_information_bits_per_spike", "mean_track2_information_bits_per_spike", "mean_track1_minus_track2_information"]].to_numpy(
                        float
                    ),
                ),
            )
            checks["RUN_subsets"] += 1
    for eid in events.event_id:
        c = arrays["counts"][arrays["offsets"][eid] : arrays["offsets"][eid + 1]]
        for split, part in enumerate(parts["splits"]):
            b = btable.loc[(eid, split)]
            observed = s.loc[(eid, split)]
            maxerr = max(maxerr, close(observed.evaluation_conditional_track2_probability, np.full(13, expit(-b.conditional_observed_log_odds))))
            maxerr = max(maxerr, close(observed.evaluation_conditional_track2_identity_z, np.full(13, -b.conditional_identity_z_log_odds)))
            if not observed.n_evaluation_spikes.eq(c[:, part["evaluation"]].sum()).all():
                raise ValueError("B spike total mismatch")
            checks["fixed_B_rows"] += 13
            for arm, ids in subsets[split].items():
                row = observed.loc[arm]
                cc = c[:, ids]
                active = int((cc.sum(axis=0) > 0).sum())
                nonempty = int((cc.sum(axis=1) > 0).sum())
                if (
                    row.n_inference_units != len(ids)
                    or row.n_inference_spikes != cc.sum()
                    or row.n_active_inference != active
                    or row.n_nonempty_bins != nonempty
                    or row.sequence_eligible != (active >= 5 and nonempty >= 5)
                ):
                    raise ValueError("independent opportunity check failed")
                if not row.sequence_eligible and (row.sequence_accepted or row.inferred_track != -1):
                    raise ValueError("ineligible event accepted")
                checks["sequence_opportunities"] += 1
    # Full arm must reproduce the original frozen experiment, not a changed classifier.
    for entry in cm["scorer_manifests"]:
        oldpath = Path(entry["manifest"])
        old = json.loads(oldpath.read_text())
        if old["session"] != session or old["repeat"] != 0:
            continue
        if digest(oldpath) != entry["sha256"] or digest(oldpath.parent / "event_content_coverage_scores.csv") != old["output_sha256"]:
            raise ValueError("original full-reference checksum changed")
        original = pd.read_csv(oldpath.parent / "event_content_coverage_scores.csv")
        original = original[original.fraction.eq(1) & original.arm.eq("real")]
        columns = ["sequence_eligible", "sequence_accepted", "inferred_track", "best_weighted_correlation", "track1_p_time", "track2_p_time", "track1_p_field", "track2_p_field"]
        for row in original.itertuples():
            expected = np.array([getattr(row, col) for col in columns], float)
            maxerr = max(maxerr, close(expected, s.loc[(row.event_id, row.split, "full"), columns].to_numpy(float)))
            checks["original_full_rows"] += 1
    if checks["original_full_rows"] != len(events) * 5:
        raise ValueError("missing original full-reference events")
    # Every arm: an eligible accepted and rejected example when available.
    null_cases = []
    for (arm, accepted), g in rows[rows.sequence_eligible].groupby(["arm", "sequence_accepted"]):
        row = g.sort_values(["event_id", "split"]).iloc[0]
        eid, split = int(row.event_id), int(row["split"])
        ids = subsets[split][arm]
        counts = arrays["counts"][arrays["offsets"][eid] : arrays["offsets"][eid + 1]]
        cc = counts[:, ids]
        nonempty = cc.sum(axis=1) > 0
        lam = maps["rates"][:, ids]
        rng = np.random.default_rng(seed(20260918, session, eid, split, "sequence-nulls"))
        shifts = rng.integers(0, len(maps["bin_centers_cm"]), (499, 2, counts.shape[1]))
        perms = np.argsort(rng.random((499, len(cc))), axis=1)
        posterior = direct_posterior(cc, lam, maps["valid_bins"])
        posterior[~nonempty] = 0.0
        corr = direct_correlations(posterior, maps["bin_centers_cm"])
        temporal = np.array([direct_correlations(posterior[p], maps["bin_centers_cm"]) for p in perms])
        field = []
        for shift in shifts:
            rolled = np.array([[np.roll(lam[k, j], shift[k, uid]) for j, uid in enumerate(ids)] for k in range(2)])
            pp = direct_posterior(cc, rolled, maps["valid_bins"])
            pp[~nonempty] = 0.0
            field.append(direct_correlations(pp, maps["bin_centers_cm"]))
        pt = (1 + (temporal >= corr - 1e-12).sum(axis=0)) / 500
        pf = (1 + (np.array(field) >= corr - 1e-12).sum(axis=0)) / 500
        passed = (pt < 0.025) & (pf < 0.025)
        best = int(np.argmax(np.where(passed, corr, -np.inf))) if passed.any() else int(np.argmax(corr))
        maxerr = max(
            maxerr, close(np.r_[pt, pf, corr[best]], row[["track1_p_time", "track2_p_time", "track1_p_field", "track2_p_field", "best_weighted_correlation"]].to_numpy(float))
        )
        if row.sequence_accepted != passed.any() or row.inferred_track != best + 1:
            raise ValueError("independent shuffle decision differs")
        null_cases.append({"arm": arm, "accepted": bool(accepted), "event_id": eid, "split": split})
        checks["sequence_null_reconstructions"] += 1
    shape = (len(events), 5, 13)

    def array(col, dtype=float):
        return s[col].to_numpy(dtype).reshape(shape)

    cube = reference_cube(
        array("sequence_accepted", bool),
        array("sequence_eligible", bool),
        array("inferred_track"),
        array("evaluation_conditional_track2_probability"),
        array("evaluation_conditional_track2_identity_z"),
        array("duration_s"),
        array("n_evaluation_spikes"),
        array("n_inference_spikes"),
    )
    saved_events = pd.read_csv(report / "event_contributions.csv").set_index(["event_id", "arm"])
    ex = pd.MultiIndex.from_product([events.event_id, ARMS])
    maxerr = max(maxerr, close(cube.reshape(-1, len(CONTRIBUTIONS)), saved_events.reindex(ex)[CONTRIBUTIONS].to_numpy(float)))
    checks["event_contributions"] = len(ex)
    summary = pd.read_csv(report / "by_stratum_arm_summary.csv")
    contrasts = pd.read_csv(report / "contrast_summary.csv")
    for (epoch, ripple), g in events.groupby(["epoch", "primary_ripple_candidate"]):
        positions = np.flatnonzero(events.event_id.isin(g.event_id))
        sub = cube[positions]
        point = reference_statistics(sub, np.ones(len(g)))
        blocks, reverse = np.unique(np.floor(g.start_s / 60).astype(int), return_inverse=True)
        rng = np.random.default_rng(seed(20260918, session, epoch, bool(ripple), "coverage-intervention-timeblock-bootstrap"))
        weights = rng.multinomial(len(blocks), np.full(len(blocks), 1 / len(blocks)), size=2000)[:, reverse]
        boots = reference_statistics(sub, weights)
        for j, arm in enumerate(ARMS):
            saved = summary[summary.epoch.eq(epoch) & summary.ripple_positive.eq(ripple) & summary.arm.eq(arm)]
            if len(saved) != 1 or saved.n_candidate_events.iloc[0] != len(g) or saved.n_time_blocks.iloc[0] != len(blocks):
                raise ValueError("summary denominator mismatch")
            for name, values in point.items():
                maxerr = max(maxerr, close(values[0, j], saved[name].iloc[0]))
                checks["arm_metrics"] += 1
            for col, slot in [("unique_selected_events", 0), ("unique_q_supported_events", 4), ("unique_z_supported_events", 6)]:
                if saved[col].iloc[0] != (sub[:, j, slot] > 0).sum():
                    raise ValueError("unique support count mismatch")
        for label, i, j in [("targeted", 1, 2)] + [(f"random_{r}", 3 + 2 * r, 4 + 2 * r) for r in range(5)]:
            saved = contrasts[contrasts.epoch.eq(epoch) & contrasts.ripple_positive.eq(ripple) & contrasts.contrast.eq(label)]
            if len(saved) != 1:
                raise ValueError("missing contrast")
            for name in ["conditional_track2_score_mean", "conditional_track2_identity_z_mean", "acceptance_fraction", "classifier_track2_fraction"]:
                delta = boots[name][:, j] - boots[name][:, i]
                finite = delta[np.isfinite(delta)]
                lo, hi = np.quantile(finite, [0.025, 0.975]) if len(finite) >= 1900 and len(blocks) >= 2 else [np.nan, np.nan]
                expected = [point[name][0, j] - point[name][0, i], lo, hi, len(finite) / 2000]
                maxerr = max(maxerr, close(expected, saved[[name + suffix for suffix in ["_delta", "_ci025", "_ci975", "_finite_bootstrap_fraction"]]].to_numpy()[0]))
                checks["contrasts_and_intervals"] += 1
    gates = pd.read_csv(report / "gate_summary.csv").set_index("gate").passed.to_dict()
    primary = contrasts[contrasts.epoch.eq("POST") & contrasts.ripple_positive & contrasts.contrast.eq("targeted")]
    support = summary[summary.epoch.eq("POST") & summary.ripple_positive & summary.arm.isin(ARMS[1:3])]
    expected_gates = {
        "complete_paired_scoring": True,
        "unchanged_evaluation_readout": True,
        "equal_half_cell_counts": True,
        "positive_RUN_selectivity_contrast_each_split": all(
            (info[0, subsets[k][ARMS[1]]] - info[1, subsets[k][ARMS[1]]]).mean() > (info[0, subsets[k][ARMS[2]]] - info[1, subsets[k][ARMS[2]]]).mean() for k in range(5)
        ),
        "five_supported_events_each_targeted_arm": len(support) == 2 and bool((support.unique_q_supported_events >= 5).all() and (support.unique_z_supported_events >= 5).all()),
    }
    for gate, metric in [("primary_content_contrast_positive", "conditional_track2_score_mean"), ("identity_control_contrast_positive", "conditional_track2_identity_z_mean")]:
        expected_gates[gate] = len(primary) == 1 and bool((primary[metric + "_delta"] > 0).all() and (primary[metric + "_ci025"] > 0).all())
    expected_gates["session_mechanistic_readout"] = all(expected_gates.values())
    if gates != expected_gates or rm["session_mechanistic_readout_passed"] != expected_gates["session_mechanistic_readout"]:
        raise ValueError("independent gate reconstruction differs")
    checks["gates"] = len(gates)
    if not all(checks.values()):
        raise ValueError("vacuous verification")
    result = {
        "status": "pass",
        "session": session,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "experiment_manifest_sha256": digest(experiment / "manifest.json"),
        "report_manifest_sha256": digest(report / "manifest.json"),
        "verifier_sha256": digest(Path(__file__)),
        "checks": checks,
        "max_absolute_error": maxerr,
        "null_cases": null_cases,
        "scope": "RUN-defined subset reconstruction, every fixed independent readout/opportunity, original full-score reproduction, per-arm null examples, all event/arm summaries and paired block intervals; not biological ground truth",
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
