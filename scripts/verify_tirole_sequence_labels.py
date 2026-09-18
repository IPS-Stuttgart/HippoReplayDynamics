"""Independent likelihood, frozen-decision and event-weighted label audit."""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

from scripts.verify_tirole_content_bank import digest, direct_posterior, seed
from scripts.verify_tirole_crossfit_mixture import close
from scripts.verify_tirole_heldout_run_selection import reference_weights

GROUPS = ["full", "half", "retained", "lost", "gained"]
METRICS = [
    "selection_mass",
    "sequence_label_track2_fraction",
    "A_poisson_agreement",
    "A_conditional_agreement",
    "B_agreement",
    "A_poisson_label_mass",
    "A_conditional_label_mass",
    "B_label_mass",
    "B_signed_z",
    "both_oppose_poisson",
    "both_oppose_conditional",
    "label_behavior_agreement",
    "retained_label_switch",
]


def reference_contributions(rows):
    ids = sorted(rows.event_id.unique())
    keys = ["event_id", "split", "repeat"]
    ix = pd.MultiIndex.from_product([ids, range(5), range(-1, 5)], names=keys)
    a = rows.set_index(keys).reindex(ix)

    def arr(col):
        return a[col].to_numpy().reshape(len(ids), 5, 6)

    accepted = arr("sequence_accepted").astype(bool)
    labels = arr("inferred_track")
    shape = (len(ids), 5, 5)
    full = np.broadcast_to(accepted[:, :, :1], shape)
    half = accepted[:, :, 1:]
    masks = [full, half, full & half, full & ~half, ~full & half]
    cube = np.zeros((len(ids), 5, len(METRICS), 2))
    confusion = []

    def pref(q):
        p = np.full(q.shape, np.nan)
        p[q < 0.5 - 1e-12] = 1
        p[q > 0.5 + 1e-12] = 2
        return p

    for g, (name, mask) in enumerate(zip(GROUPS, masks, strict=True)):

        def values(col, name=name):
            v = arr(col)
            return np.broadcast_to(v[:, :, :1], shape) if name in ["full", "lost"] else v[:, :, 1:]

        label = values("inferred_track")
        pa = values("A_poisson_track2_mass")
        ca = values("A_conditional_track2_mass")
        pb = values("B_track2_mass")
        z = values("B_track2_z")
        pp, cp, bp = pref(pa), pref(ca), pref(pb)
        truth = values("truth_track")
        measures = [
            (mask, np.ones(shape, bool)),
            (label == 2, mask),
            (pp == label, mask & np.isfinite(pp)),
            (cp == label, mask & np.isfinite(cp)),
            (bp == label, mask & np.isfinite(bp)),
            (np.where(label == 2, pa, 1 - pa), mask & np.isfinite(pa)),
            (np.where(label == 2, ca, 1 - ca), mask & np.isfinite(ca)),
            (np.where(label == 2, pb, 1 - pb), mask & np.isfinite(pb)),
            (np.where(label == 2, z, -z), mask & np.isfinite(z)),
            ((pp != label) & (bp != label), mask & np.isfinite(pp) & np.isfinite(bp)),
            ((cp != label) & (bp != label), mask & np.isfinite(cp) & np.isfinite(bp)),
            (label == truth, mask & np.isfinite(truth)),
            (labels[:, :, :1] != labels[:, :, 1:], mask if name == "retained" else np.zeros(shape, bool)),
        ]
        for k, (value, valid) in enumerate(measures):
            cube[:, g, k, 0] = np.where(valid, value, 0).mean(axis=(1, 2))
            cube[:, g, k, 1] = valid.mean(axis=(1, 2))
        for ref, p in [("behavior", truth), ("A_poisson", pp), ("A_conditional", cp), ("B", bp)]:
            p = np.where(np.isfinite(p), p, -1)
            total = mask.mean(axis=(1, 2)).sum()
            for lab in [1, 2]:
                for target in [1, 2, -1]:
                    mass = (mask & (label == lab) & (p == target)).mean(axis=(1, 2)).sum()
                    confusion.append(
                        {
                            "group": name,
                            "reference": ref,
                            "sequence_label": lab,
                            "reference_label": target,
                            "selection_weight": mass,
                            "total_selection_weight": total,
                            "joint_fraction": mass / total if total else np.nan,
                        }
                    )
    return cube, pd.DataFrame(confusion)


def checked(folder, key="output_sha256"):
    m = json.loads((folder / "manifest.json").read_text())
    for name, h in m[key].items():
        p = (folder / name).resolve()
        if not p.is_relative_to(folder.resolve()) or digest(p) != h:
            raise ValueError("changed source")
    return m


def run(source, output):
    if output.exists():
        raise ValueError("new independent verification output required")
    m = checked(source)
    root = Path(m["source_root"])
    session = m["session"]
    if m["status"] != "complete" or m["git_dirty"] or not m["original_sequence_decisions_unchanged"]:
        raise ValueError("invalid source audit")
    checks = {"original_decisions": 0, "A_readouts": 0, "B_readouts": 0, "event_metric_contributions": 0, "summary_intervals": 0, "confusion_rows": 0}
    error = 0.0
    summary = pd.read_csv(source / "label_agreement_summary.csv")
    saved_confusion = pd.read_csv(source / "label_confusion.csv")
    for domain in ["RUN", "POST"]:
        rows = pd.read_csv(source / (domain + "_label_rows.csv"))
        ids = sorted(rows.event_id.unique())
        keys = ["event_id", "split", "repeat"]
        ix = pd.MultiIndex.from_product([ids, range(5), range(-1, 5)], names=keys)
        actual = rows.set_index(keys).reindex(ix)
        if len(rows) != len(ix) or rows.duplicated(keys).any() or actual.sequence_accepted.isna().any():
            raise ValueError("incomplete audit rows")
        if domain == "RUN":
            folder = root / "heldout_RUN_selection_v1" / session
            checked(folder)
            if digest(folder / "manifest.json") != m["inputs"][domain]["sha256"]:
                raise ValueError("changed RUN source")
            v = folder.parent / (session + "_independent_verification_v2.json")
            if digest(v) != m["inputs"][domain]["verification_sha256"] or json.loads(v.read_text())["status"] != "pass":
                raise ValueError("RUN verification changed")
            original = pd.read_csv(folder / "RUN_sequence_scores.csv").rename(columns={"window_id": "event_id"})
            original = original[original.order.eq("original") & original.likelihood.eq("poisson")]
            b = pd.read_csv(folder / "RUN_independent_content.csv").rename(columns={"window_id": "event_id"}).set_index(["event_id", "split"])
            windows = pd.read_csv(folder / "RUN_windows.csv").rename(columns={"window_id": "event_id"}).sort_values("event_id").reset_index(drop=True)
            arrays = np.load(folder / "RUN_counts.npz")
            lookup = dict(zip(arrays["window_ids"], range(len(arrays["window_ids"])), strict=True))
            parts = json.loads((folder / "partitions.json").read_text())
            maps = {f: np.load(folder / "fold_maps" / f"{f}.npz") for f in range(5)}
            expected_b = {
                key: (
                    float(r.evaluation_true_probability if r.truth_track == 2 else 1 - r.evaluation_true_probability),
                    float(r.evaluation_true_z if r.truth_track == 2 else -r.evaluation_true_z),
                )
                for key, r in b.iterrows()
            }
        else:
            bank = Path(m["inputs"][domain]["bank"])
            checked(bank, "outputs_sha256")
            if digest(bank / "manifest.json") != m["inputs"][domain]["bank_manifest_sha256"]:
                raise ValueError("changed replay bank")
            frames = []
            for entry in m["inputs"][domain]["score_manifests"]:
                p = Path(entry["manifest"])
                sm = json.loads(p.read_text())
                if digest(p) != entry["sha256"] or digest(p.parent / "event_content_coverage_scores.csv") != sm["output_sha256"]:
                    raise ValueError("changed original decision source")
                frames.append(pd.read_csv(p.parent / "event_content_coverage_scores.csv"))
            all_scores = pd.concat(frames, ignore_index=True)
            all_scores = all_scores[all_scores.epoch.eq("POST") & all_scores.arm.eq("real")]
            full = all_scores[all_scores.fraction.eq(1) & all_scores.repeat.eq(0)].copy()
            full["repeat"] = -1
            original = pd.concat([full, all_scores[all_scores.fraction.eq(0.5)]], ignore_index=True)
            windows = pd.read_csv(bank / "candidate_events.csv")
            windows = windows[windows.primary_ripple_candidate & windows.epoch.eq("POST")].sort_values("event_id").reset_index(drop=True)
            windows["truth_track"] = np.nan
            windows["fold"] = -1
            windows["time_block"] = np.floor(windows.start_s / 60).astype(int)
            arrays = np.load(bank / "event_counts.npz")
            maps = {-1: np.load(bank / "RUN_maps.npz")}
            parts = json.loads((bank / "partitions.json").read_text())
            ef = root / "evaluation_identity_control_v1"
            checked(ef)
            if digest(ef / "manifest.json") != m["inputs"][domain]["evaluation_manifest_sha256"]:
                raise ValueError("changed B control")
            b = pd.read_csv(ef / "evaluation_identity_by_event_split.csv")
            b = b[b.session.eq(session)].set_index(["event_id", "split"])
            expected_b = {key: (float(expit(-r.conditional_observed_log_odds)), float(-r.conditional_identity_z_log_odds)) for key, r in b.iterrows()}
        if windows.event_id.to_list() != ids:
            raise ValueError("event cohort changed")
        original = original.set_index(keys).reindex(ix)
        for col in ["sequence_accepted", "sequence_eligible", "inferred_track", "n_inference_spikes"]:
            np.testing.assert_array_equal(original[col].to_numpy(), actual[col].to_numpy())
        checks["original_decisions"] += len(ix)
        for r in rows.to_dict("records"):
            eid, split, repeat, fold = map(int, [r["event_id"], r["split"], r["repeat"], r["fold"]])
            if domain == "RUN":
                counts = arrays["counts"][lookup[eid]]
                part = parts[fold]["splits"][split]
                population = f"{session}:RUNfold{fold}"
                duration = 0.1
            else:
                counts = arrays["counts"][arrays["offsets"][eid] : arrays["offsets"][eid + 1]]
                part = parts["splits"][split]
                population = session
                duration = 0.02
            inference = np.asarray(part["inference"])
            perm = np.random.default_rng(seed(20260918, population, split, repeat, "coverage")).permutation(inference)
            use = inference if repeat == -1 else np.sort(perm[: int(np.ceil(len(inference) * 0.5))])
            cc = counts[:, use]
            cc = cc[cc.sum(axis=1) > 0]
            if cc.sum() != r["n_inference_spikes"]:
                raise ValueError("wrong subset counts")
            lam = np.maximum(maps[fold]["rates"][:, use], 1e-4) * (duration / 0.02)
            q = [float(direct_posterior(cc, lam, maps[fold]["valid_bins"], conditional)[:, 1].sum(axis=1).mean()) for conditional in [False, True]] if len(cc) else [np.nan, np.nan]
            error = max(error, close(q, [r["A_poisson_track2_mass"], r["A_conditional_track2_mass"]]), close(expected_b[eid, split], [r["B_track2_mass"], r["B_track2_z"]]))
            checks["A_readouts"] += 2
            checks["B_readouts"] += 1
        cube, conf = reference_contributions(rows)
        saved = pd.read_csv(source / (domain + "_event_contributions.csv"))
        ci = pd.MultiIndex.from_product([ids, GROUPS], names=["event_id", "group"])
        cols = [k + suffix for k in METRICS for suffix in ["_num", "_den"]]
        error = max(error, close(cube.reshape(-1, len(cols)), saved.set_index(ci.names).reindex(ci)[cols].to_numpy()))
        checks["event_metric_contributions"] += cube.size
        ckeys = ["group", "reference", "sequence_label", "reference_label"]
        sc = saved_confusion[saved_confusion.domain.eq(domain)].set_index(ckeys).reindex(pd.MultiIndex.from_frame(conf[ckeys]))
        for col in ["selection_weight", "total_selection_weight", "joint_fraction"]:
            error = max(error, close(conf[col], sc[col]))
        checks["confusion_rows"] += len(conf)
        if domain == "RUN":
            w, adequate = reference_weights(windows, np.random.default_rng(seed(20260918, session, "label-audit-RUN-blocks")))
            blocks = np.array([f"{int(t)}:{int(f)}:{int(b)}" for t, f, b in windows[["truth_track", "fold", "time_block"]].to_numpy()])
        else:
            blocks = windows.time_block.to_numpy()
            u, bc = np.unique(blocks, return_inverse=True)
            adequate = len(u) >= 2
            w = np.random.default_rng(seed(20260918, session, "label-audit-POST-blocks")).multinomial(len(u), np.full(len(u), 1 / len(u)), size=2000)[:, bc]
        all_weights = np.vstack([np.ones(len(ids)), w])
        total = np.tensordot(all_weights, cube, axes=(1, 0))
        den = total[:, :, :, 1]
        estimates = np.divide(total[:, :, :, 0], den, out=np.full_like(den, np.nan), where=den > 0)
        for row in summary[summary.domain.eq(domain)].itertuples():
            k = METRICS.index(row.metric)
            if row.group == "half_minus_full":
                values = estimates[:, 1, k] - estimates[:, 0, k]
                count = min(len(np.unique(blocks[cube[:, g, k, 1] > 0])) for g in [0, 1])
                denominator = np.nan
            else:
                g = GROUPS.index(row.group)
                values = estimates[:, g, k]
                count = len(np.unique(blocks[cube[:, g, k, 1] > 0]))
                denominator = den[0, g, k]
            finite = values[1:][np.isfinite(values[1:])]
            ok = adequate and count >= 2 and len(finite) >= 1900
            lo, hi = np.quantile(finite, [0.025, 0.975]) if ok else [np.nan, np.nan]
            error = max(
                error,
                close(
                    [values[0], lo, hi, denominator, count, len(finite) / 2000],
                    [row.estimate, row.ci025, row.ci975, row.effective_denominator, row.informative_blocks, row.finite_bootstrap_fraction],
                ),
            )
            if bool(row.interval_supported) != ok:
                raise ValueError("interval support differs")
            checks["summary_intervals"] += 1
    if checks["summary_intervals"] != 140 or checks["confusion_rows"] != 240:
        raise ValueError("incomplete verification")
    result = {
        "status": "pass",
        "session": session,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_manifest_sha256": digest(source / "manifest.json"),
        "verifier_sha256": digest(Path(__file__)),
        "checks": checks,
        "max_absolute_error": error,
        "scope": "unchanged source decisions and B readouts; every A population posterior; independent tensor aggregation, confusion counts and all block intervals; not replay ground truth",
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--experiment-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    run(a.experiment_dir, a.output)
