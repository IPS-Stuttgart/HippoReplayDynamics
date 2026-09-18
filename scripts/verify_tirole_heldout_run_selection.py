"""Raw-count, fold-exclusion, paired-tensor and direct-null RUN verification."""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.special import expit

from hipporeplayimm.tirole_two_track import fit_maps, load_session
from scripts.verify_tirole_content_bank import digest, direct_correlations, direct_posterior, seed
from scripts.verify_tirole_crossfit_mixture import close

GROUPS = ["all", "full", "half", "retained", "lost", "gained", "lost_support", "lost_sequence"]
READOUTS = ["evaluation_true_probability", "evaluation_true_z", "evaluation_correct"]


def compare_map_arrays(expected, saved):
    if np.asarray(expected).dtype.kind == "b":
        np.testing.assert_array_equal(expected, saved)
        return 0.0
    return close(expected, saved)


def reference_cube(frame, content, ids, order, likelihood):
    keys = ["window_id", "split", "repeat"]
    ix = pd.MultiIndex.from_product([ids, range(5), range(-1, 5)], names=keys)
    a = frame[frame.order.eq(order) & frame.likelihood.eq(likelihood)].set_index(keys).reindex(ix)
    if a.sequence_accepted.isna().any():
        raise ValueError("missing original score")
    accepted = a.sequence_accepted.to_numpy(bool).reshape(len(ids), 5, 6)
    eligible = a.sequence_eligible.to_numpy(bool).reshape(len(ids), 5, 6)
    labels = a.inferred_track.to_numpy().reshape(len(ids), 5, 6)
    f = np.broadcast_to(accepted[:, :, :1], (len(ids), 5, 5))
    h = accepted[:, :, 1:]
    groups = np.stack([np.ones_like(h), f, h, f & h, f & ~h, ~f & h, f & ~h & ~eligible[:, :, 1:], f & ~h & eligible[:, :, 1:]], axis=-1)
    cols = ["group_mass", "decoded_label_mass", "decoded_correct_mass", "decoded_track2_mass", *[v + s for v in READOUTS for s in ["_numerator", "_denominator"]]]
    cube = np.zeros((len(ids), len(GROUPS), len(cols)))
    bi = pd.MultiIndex.from_product([ids, range(5)], names=["window_id", "split"])
    b = content.set_index(bi.names).reindex(bi)
    truth = b.truth_track.to_numpy().reshape(len(ids), 5)[:, :, None]
    for g, name in enumerate(GROUPS):
        use = groups[:, :, :, g]
        label = labels[:, :, 1:] if name in ["half", "gained"] else np.broadcast_to(labels[:, :, :1], h.shape)
        cube[:, g, 0] = use.mean(axis=(1, 2))
        cube[:, g, 1] = (use & np.isin(label, [1, 2])).mean(axis=(1, 2))
        cube[:, g, 2] = (use & (label == truth)).mean(axis=(1, 2))
        cube[:, g, 3] = (use & (label == 2)).mean(axis=(1, 2))
        for c, col in enumerate(READOUTS):
            value = b[col].to_numpy().reshape(len(ids), 5, 1)
            finite = use & np.isfinite(value)
            cube[:, g, 4 + 2 * c] = np.where(finite, value, 0).mean(axis=(1, 2))
            cube[:, g, 5 + 2 * c] = finite.mean(axis=(1, 2))
    return cube, cols


def reference_metrics(cube, truth, weights):
    w = np.atleast_2d(weights)
    result = {}
    total = np.tensordot(w, cube, axes=(1, 0))

    def divide(a, b):
        return np.divide(a, b, out=np.full_like(a, np.nan, dtype=float), where=b > 0)

    for g, name in enumerate(GROUPS):
        mass = total[:, g, 0]
        result[name + "_mass"] = divide(mass, w.sum(axis=1))
        result[name + "_true_track2_fraction"] = divide(w @ (cube[:, g, 0] * (truth == 2)), mass)
        for t in [1, 2]:
            result[name + f"_acceptance_track{t}"] = divide(w @ (cube[:, g, 0] * (truth == t)), w @ (truth == t))
        result[name + "_decoded_correct_fraction"] = divide(total[:, g, 2], total[:, g, 1])
        result[name + "_decoded_track2_fraction"] = divide(total[:, g, 3], total[:, g, 1])
        for c, col in enumerate(READOUTS):
            den = total[:, g, 5 + 2 * c]
            result[name + "_" + col] = divide(total[:, g, 4 + 2 * c], den)
            result[name + "_" + col + "_finite_fraction"] = divide(den, mass)
    result["half_minus_full_true_track2_fraction"] = result["half_true_track2_fraction"] - result["full_true_track2_fraction"]
    result["half_minus_full_acceptance"] = result["half_mass"] - result["full_mass"]
    for t in [1, 2]:
        result[f"loss_given_full_track{t}"] = divide(result[f"lost_acceptance_track{t}"], result[f"full_acceptance_track{t}"])
    return result


def reference_weights(windows, rng):
    w = np.zeros((2000, len(windows)))
    adequate = True
    for track in [1, 2]:
        for fold in range(5):
            idx = np.flatnonzero((windows.truth_track == track) & (windows.fold == fold))
            block = windows.time_block.to_numpy()[idx]
            unique = np.unique(block)
            adequate &= len(unique) >= 2
            if not len(unique):
                continue
            counts = rng.multinomial(len(unique), np.repeat(1 / len(unique), len(unique)), size=2000)
            expanded = np.stack([counts[:, np.flatnonzero(unique == b)[0]] for b in block], axis=1).astype(float)
            w[:, idx] = expanded * len(idx) / expanded.sum(axis=1)[:, None]
    return w, adequate


def run(source, report, output):
    if output.exists():
        raise ValueError("new verification output required")
    m = json.loads((source / "manifest.json").read_text())
    rm = json.loads((report / "manifest.json").read_text())
    for folder, meta in [(source, m), (report, rm)]:
        if meta["status"] != "complete" or meta["git_dirty"]:
            raise ValueError("incomplete provenance")
        for name, h in meta["output_sha256"].items():
            p = (folder / name).resolve()
            if not p.is_relative_to(folder.resolve()) or digest(p) != h:
                raise ValueError("changed or unsafe artifact")
    if digest(source / "manifest.json") != rm["source_manifest_sha256"]:
        raise ValueError("different report source")
    for raw in m["source_files"]:
        if digest(raw["path"]) != raw["sha256"]:
            raise ValueError("changed recording")
    session = load_session(Path(m["dataset_root"]), m["session"])
    windows = pd.read_csv(source / "RUN_windows.csv", float_precision="round_trip").sort_values("window_id").reset_index(drop=True)
    arrays = np.load(source / "RUN_counts.npz")
    ids = windows.window_id.to_list()
    parts = json.loads((source / "partitions.json").read_text())
    scores = pd.read_csv(source / "RUN_sequence_scores.csv")
    content = pd.read_csv(source / "RUN_independent_content.csv")
    if (
        len(scores) != len(ids) * 120
        or scores.duplicated(["window_id", "split", "order", "likelihood", "repeat"]).any()
        or len(content) != len(ids) * 5
        or content.duplicated(["window_id", "split"]).any()
    ):
        raise ValueError("incomplete experiment")
    error = close(ids, arrays["window_ids"])
    checks = {"raw_count_entries": 0, "training_folds": 0, "score_opportunities": 0, "summary_metrics": 0, "intervals": 0, "direct_sequence_cases": 0, "direct_B_readouts": 0}
    candidates = []
    runmask = session.run_mask()
    for eid in range(int(np.floor(session.times[-1] - session.times[0]))):
        start = session.times[0] + eid
        end = start + 1.0
        left, right = np.searchsorted(session.times, [start, end])
        for t in [1, 2]:
            if right > left and (runmask[left:right] & np.isfinite(session.positions[t - 1, left:right])).all():
                candidates.append((eid, t, eid // 10 % 5))
    candidate = pd.DataFrame(candidates, columns=["window_id", "truth_track", "fold"])
    selected = []
    for fold in range(5):
        by = [candidate[(candidate.fold == fold) & (candidate.truth_track == t)].window_id.to_list() for t in [1, 2]]
        n = min(20, *map(len, by))
        for t, values in enumerate(by, 1):
            selected.extend(sorted(values, key=lambda eid: seed(20260918, m["session"], fold, t, eid, "RUN-known-context-window"))[:n])
    if sorted(selected) != ids:
        raise ValueError("behavior-only selection changed")
    raw = loadmat(Path(m["dataset_root"]) / (m["session"] + "_extracted_clusters.mat"), simplify_cells=True)["clusters"]
    st = np.asarray(raw["spike_times"])
    unit = np.asarray(raw["spike_id"])
    maps = {}
    for fold in range(5):
        saved = np.load(source / "fold_maps" / f"{fold}.npz")
        relative = session.times - session.times[0]
        block = np.floor(relative / 10).astype(int)
        phase = relative - block * 10
        train = ~((block % 5 == fold) | ((phase < 1) & ((block - 1) % 5 == fold)) | ((phase > 9) & ((block + 1) % 5 == fold)))
        np.testing.assert_array_equal(train, saved["training_samples"])
        fitted = fit_maps(session, train)
        for key, value in fitted.items():
            error = max(error, compare_map_arrays(value, saved[key]))
        detector = parts[fold]["reserved_detector_units"]
        common = set(saved["common_units"])
        for part in parts[fold]["splits"]:
            a, b = set(part["inference"]), set(part["evaluation"])
            if a & b or (a | b) & set(detector) or a | b | set(detector) != common:
                raise ValueError("cell-role leakage")
        maps[fold] = saved
        checks["training_folds"] += 1
    for i, row in windows.iterrows():
        edges = row.start_s + np.arange(11) * 0.1
        use = (st >= edges[0]) & (st < edges[-1])
        counts = np.stack([np.histogram(st[use & (unit == u)], edges)[0] for u in arrays["unit_ids"]], axis=1)
        np.testing.assert_array_equal(counts, arrays["counts"][i])
        checks["raw_count_entries"] += counts.size
        for j in range(10):
            take = (session.times >= edges[j]) & (session.times < edges[j + 1])
            actual = session.positions[int(row.truth_track) - 1, take]
            error = max(error, close(actual.mean(), arrays["truth_bin_position_cm"][i, j]))
        if maps[int(row.fold)]["training_samples"][(session.times >= row.start_s) & (session.times < row.end_s)].any():
            raise ValueError("test/training overlap")
    # Reconstruct all activity opportunities with paired whole-bin observations.
    indexed = scores.set_index(["window_id", "split", "order", "likelihood", "repeat"])
    bindex = content.set_index(["window_id", "split"])
    for i, row in windows.iterrows():
        fold = int(row.fold)
        c = arrays["counts"][i]
        for split, part in enumerate(parts[fold]["splits"]):
            evaluation = np.array(part["evaluation"])
            inference = np.array(part["inference"])
            b = bindex.loc[row.window_id, split]
            if b.n_evaluation_spikes != c[:, evaluation].sum() or b.n_evaluation_active_units != (c[:, evaluation].sum(axis=0) > 0).sum():
                raise ValueError("independent support mismatch")
            for repeat in range(-1, 5):
                perm = np.random.default_rng(seed(20260918, f"{m['session']}:RUNfold{fold}", split, repeat, "coverage")).permutation(inference)
                use = inference if repeat == -1 else np.sort(perm[: int(np.ceil(len(inference) / 2))])
                obs = c[:, use]
                active = (obs.sum(axis=0) > 0).sum()
                nonempty = (obs.sum(axis=1) > 0).sum()
                for order in ["original", "whole_bin_shuffled"]:
                    for likelihood in ["poisson", "conditional_count"]:
                        s = indexed.loc[row.window_id, split, order, likelihood, repeat]
                        error = max(error, close([len(use), obs.sum(), active, nonempty], [s.n_inference_cells, s.n_inference_spikes, s.n_active_inference, s.n_nonempty_bins]))
                        if bool(s.sequence_eligible) != (active >= 5 and nonempty >= 5):
                            raise ValueError("eligibility mismatch")
                        checks["score_opportunities"] += 1
    stat = pd.read_csv(report / "window_statistics.csv")
    summary = pd.read_csv(report / "RUN_selection_summary.csv")
    ci = pd.read_csv(report / "RUN_block_uncertainty.csv")
    weights, adequate = reference_weights(windows, np.random.default_rng(seed(20260918, m["session"], "RUN-stratified-block-bootstrap")))
    points = {}
    draws = {}
    for order in ["original", "whole_bin_shuffled"]:
        for likelihood in ["poisson", "conditional_count"]:
            cube, cols = reference_cube(scores, content, ids, order, likelihood)
            ix = pd.MultiIndex.from_product([ids, GROUPS], names=["window_id", "group"])
            saved = stat[(stat.order == order) & (stat.likelihood == likelihood)].set_index(ix.names).reindex(ix)
            error = max(error, close(cube.reshape(-1, len(cols)), saved[cols].to_numpy()))
            point = reference_metrics(cube, windows.truth_track.to_numpy(), np.ones(len(ids)))
            boot = reference_metrics(cube, windows.truth_track.to_numpy(), weights)
            points[order, likelihood] = point
            draws[order, likelihood] = boot
            record = summary[(summary.order == order) & (summary.likelihood == likelihood)].iloc[0]
            for metric, value in point.items():
                error = max(error, close(value[0], record[metric]))
                checks["summary_metrics"] += 1
    for r in ci.itertuples():
        if r.order == "paired_order":
            metric = r.metric.replace("_original_minus_shuffled_acceptance", "_mass")
            point = points["original", r.likelihood][metric] - points["whole_bin_shuffled", r.likelihood][metric]
            boot = draws["original", r.likelihood][metric] - draws["whole_bin_shuffled", r.likelihood][metric]
        else:
            point = points[r.order, r.likelihood][r.metric]
            boot = draws[r.order, r.likelihood][r.metric]
        finite = boot[np.isfinite(boot)]
        lo, hi = np.quantile(finite, [0.025, 0.975]) if adequate and len(finite) >= 1900 else [np.nan, np.nan]
        error = max(error, close([point[0], lo, hi, len(finite) / 2000], [r.estimate, r.ci025, r.ci975, r.finite_bootstrap_fraction]))
        checks["intervals"] += 1
    # Verify one direct 499+499-null example in each available condition/status.
    sample = scores[scores.sequence_eligible & scores.repeat.isin([-1, 2])]
    for _, r in sample.groupby(["order", "likelihood", "repeat", "sequence_accepted"]).first().reset_index().iterrows():
        fold, split, eid, repeat = map(int, [r["fold"], r["split"], r["window_id"], r["repeat"]])
        i = ids.index(eid)
        saved = maps[fold]
        c = arrays["counts"][i]
        if r.order == "whole_bin_shuffled":
            c = c[np.random.default_rng(seed(20260918, m["session"], eid, fold, split, "RUN-order-negative-control")).permutation(10)]
        inference = np.array(parts[fold]["splits"][split]["inference"])
        perm = np.random.default_rng(seed(20260918, f"{m['session']}:RUNfold{fold}", split, repeat, "coverage")).permutation(inference)
        use = inference if repeat == -1 else np.sort(perm[: int(np.ceil(len(inference) / 2))])
        obs = c[:, use]
        lam = 5 * np.maximum(saved["rates"][:, use], 1e-4)
        rng = np.random.default_rng(seed(20260918, m["session"], eid, fold, split, r.order, "RUN-selection-nulls"))
        shifts = rng.integers(0, len(saved["bin_centers_cm"]), (499, 2, c.shape[1]))
        perms = np.argsort(rng.random((499, 10)), axis=1)
        conditional = r.likelihood == "conditional_count"
        p = direct_posterior(obs, lam, saved["valid_bins"], conditional)
        p[obs.sum(axis=1) == 0] = 0
        corr = direct_correlations(p, saved["bin_centers_cm"])
        tnull = np.array([direct_correlations(p[k], saved["bin_centers_cm"]) for k in perms])
        fnull = []
        for shift in shifts:
            rolled = np.array([[np.roll(lam[k, j], shift[k, u]) for j, u in enumerate(use)] for k in range(2)])
            fp = direct_posterior(obs, rolled, saved["valid_bins"], conditional)
            fp[obs.sum(axis=1) == 0] = 0
            fnull.append(direct_correlations(fp, saved["bin_centers_cm"]))
        pt = (1 + (tnull >= corr - 1e-12).sum(axis=0)) / 500
        pf = (1 + (np.array(fnull) >= corr - 1e-12).sum(axis=0)) / 500
        passed = (pt < 0.025) & (pf < 0.025)
        best = int(np.argmax(np.where(passed, corr, -np.inf))) if passed.any() else int(np.argmax(corr))
        error = max(error, close(np.r_[pt, pf, corr[best]], r[["track1_p_time", "track2_p_time", "track1_p_field", "track2_p_field", "best_weighted_correlation"]].to_numpy(float)))
        if bool(r.sequence_accepted) != passed.any() or r.inferred_track != best + 1:
            raise ValueError("direct null decision differs")
        checks["direct_sequence_cases"] += 1
    # Fixed first window of each fold/split: independent conditional identity null.
    for fold in range(5):
        first = windows[windows.fold == fold].iloc[0]
        i = ids.index(int(first.window_id))
        saved = maps[fold]
        occ = saved["occupancy_s"]
        mean = (saved["rates"] * (occ / occ.sum(axis=1)[:, None])[:, None]).sum(axis=2).mean(axis=0)
        for split, part in enumerate(parts[fold]["splits"]):
            use = np.array(part["evaluation"])
            cc = arrays["counts"][i][:, use]
            cc = cc[cc.sum(axis=1) > 0]
            lam = 5 * np.maximum(saved["rates"][:, use], 1e-4)
            groups = np.array_split(np.argsort(mean[use], kind="stable"), max(1, len(use) // 3))
            rng = np.random.default_rng(seed(20260918, m["session"], fold, split, "RUN-evaluation-identity"))
            perms = np.tile(np.arange(len(use)), (199, 1))
            for group in groups:
                for per in perms:
                    per[group] = rng.permutation(group)
            odds = []
            for per in np.vstack([np.arange(len(use)), perms]):
                p = direct_posterior(cc, lam[:, per], saved["valid_bins"], True)
                mass = p.sum(axis=(0, 2))
                odds.append(np.log(max(mass[0], 1e-300)) - np.log(max(mass[1], 1e-300)))
            sign = 1 if first.truth_track == 1 else -1
            sd = np.std(odds[1:], ddof=1)
            z = sign * (odds[0] - np.mean(odds[1:])) / sd if sd > 1e-12 else np.nan
            target = bindex.loc[first.window_id, split]
            expected = [expit(sign * odds[0]), z, float(sign * odds[0] > 0)] if len(cc) else [np.nan] * 3
            error = max(error, close(expected, target[READOUTS].to_numpy(float)))
            checks["direct_B_readouts"] += 1
    result = {
        "status": "pass",
        "session": m["session"],
        "created_at_utc": datetime.now(UTC).isoformat(),
        "experiment_manifest_sha256": digest(source / "manifest.json"),
        "report_manifest_sha256": digest(report / "manifest.json"),
        "verifier_sha256": digest(Path(__file__)),
        "checks": checks,
        "max_absolute_error": error,
        "scope": "all raw counts, behavioral selection, training exclusions, opportunities, window tensors and intervals; stratified direct null examples; 25 direct independent content examples; not biological replay validation",
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--experiment-dir", type=Path, required=True)
    p.add_argument("--report-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    run(a.experiment_dir, a.report_dir, a.output)
