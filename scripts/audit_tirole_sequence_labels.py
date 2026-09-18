"""Audit frozen sequence labels against inference and independent track content."""

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from hipporeplayimm.tirole_two_track import decode_counts, file_sha256
from hipporeplayimm.two_track_content import field_shift_posteriors, nested_subsets, stable_seed
from scripts.audit_tirole_experience_coverage import load_fixed_evaluation
from scripts.report_tirole_heldout_run_selection import block_weights as run_block_weights
from scripts.score_tirole_content_coverage import validate_bank

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
META = ["event_id", "start_s", "truth_track", "fold", "time_block"]


def checked(folder):
    m = json.loads((folder / "manifest.json").read_text())
    if m.get("status") != "complete" or m.get("git_dirty", False):
        raise ValueError("incomplete or dirty source")
    for name, h in m["output_sha256"].items():
        p = (folder / name).resolve()
        if not p.is_relative_to(folder.resolve()) or file_sha256(p) != h:
            raise ValueError("changed or unsafe source")
    return m


def track_masses(counts, rates, valid, duration):
    cc = counts[counts.sum(axis=1) > 0]
    if not len(cc):
        return np.nan, np.nan
    p = decode_counts(cc, rates, duration, valid)
    c = field_shift_posteriors(cc, rates, valid, np.zeros((1, *rates.shape[:2]), int), conditional_count=True)[0]
    return float(p[:, 1].sum(axis=1).mean()), float(c[:, 1].sum(axis=1).mean())


def preference(q):
    q = np.asarray(q, float)
    if ((q[np.isfinite(q)] < 0) | (q[np.isfinite(q)] > 1)).any():
        raise ValueError("invalid track mass")
    return np.where(q > 0.5 + 1e-12, 2, np.where(q < 0.5 - 1e-12, 1, np.nan))


def load_run(source, session):
    folder = source / "heldout_RUN_selection_v1" / session
    m = checked(folder)
    audit = source / "heldout_RUN_selection_v1" / (session + "_independent_verification_v2.json")
    v = json.loads(audit.read_text())
    if v["status"] != "pass" or v["experiment_manifest_sha256"] != file_sha256(folder / "manifest.json"):
        raise ValueError("unverified RUN calibration")
    windows = pd.read_csv(folder / "RUN_windows.csv").rename(columns={"window_id": "event_id"})
    scores = pd.read_csv(folder / "RUN_sequence_scores.csv").rename(columns={"window_id": "event_id"})
    scores = scores[scores.order.eq("original") & scores.likelihood.eq("poisson")].copy()
    content = pd.read_csv(folder / "RUN_independent_content.csv").rename(columns={"window_id": "event_id"})
    content["B_track2_mass"] = np.where(content.truth_track.eq(2), content.evaluation_true_probability, 1 - content.evaluation_true_probability)
    content["B_track2_z"] = np.where(content.truth_track.eq(2), content.evaluation_true_z, -content.evaluation_true_z)
    arrays = np.load(folder / "RUN_counts.npz")
    parts = json.loads((folder / "partitions.json").read_text())
    maps = {f: np.load(folder / "fold_maps" / f"{f}.npz") for f in range(5)}
    lookup = {int(e): i for i, e in enumerate(arrays["window_ids"])}

    def observation(eid, fold, split, repeat):
        a = np.array(parts[fold]["splits"][split]["inference"])
        ids = a if repeat == -1 else nested_subsets(a, f"{session}:RUNfold{fold}", split, repeat)[0.5]
        return arrays["counts"][lookup[eid]][:, ids], maps[fold]["rates"][:, ids], maps[fold]["valid_bins"], 0.1

    return (
        windows,
        scores,
        content,
        observation,
        {"manifest": str(folder / "manifest.json"), "sha256": file_sha256(folder / "manifest.json"), "verification_sha256": file_sha256(audit), "code_commit": m["code_commit"]},
    )


def load_rest(source, session):
    bank = source / ("candidate_banks_v2" if session == "RAT3_SESS2" else "candidate_banks_v3") / session
    m, parts, events, counts, offsets, maps = validate_bank(bank)
    if m["cohort_stratum"] != "strict_RUN_pass":
        raise ValueError("primary cohort changed")
    windows = events[events.primary_ripple_candidate & events.epoch.eq("POST")].copy()
    windows["truth_track"] = np.nan
    windows["fold"] = -1
    windows["time_block"] = np.floor(windows.start_s / 60).astype(int)
    campaign = source / ("campaign_RAT3_SESS2_v1" if session == "RAT3_SESS2" else "campaign_remaining_stage2_v1")
    state = json.loads((campaign / "campaign_status.json").read_text())
    if state["status"] != "complete" or state["failed"]:
        raise ValueError("unfinished source campaign")
    frames = []
    sources = []
    for job in state["completed"]:
        folder = campaign / job["job"]
        meta = json.loads((folder / "manifest.json").read_text())
        if meta["session"] != session or meta["candidate_stratum"] != "ripple":
            continue
        p = folder / "event_content_coverage_scores.csv"
        if (
            meta["status"] != "complete"
            or meta["git_dirty"]
            or meta["technical_pilot_only"]
            or meta["bank_manifest_sha256"] != file_sha256(bank / "manifest.json")
            or meta["output_sha256"] != file_sha256(p)
        ):
            raise ValueError("invalid replay score source")
        x = pd.read_csv(p)
        if len(x) != meta["expected_score_rows"]:
            raise ValueError("incomplete replay score file")
        x = x[x.event_id.isin(windows.event_id) & x.arm.eq("real") & x.fraction.isin([1.0, 0.5])]
        if len(x) != 2 * len(windows) or x.duplicated(["event_id", "fraction"]).any():
            raise ValueError("missing paired replay coverage")
        frames.append(x)
        sources.append({"manifest": str(folder / "manifest.json"), "sha256": file_sha256(folder / "manifest.json")})
    x = pd.concat(frames, ignore_index=True)
    if len(frames) != 25 or x.duplicated(["event_id", "split", "repeat", "fraction"]).any():
        raise ValueError("incomplete frozen cell splits")
    full = x[x.fraction.eq(1.0)]
    cols = ["sequence_accepted", "sequence_eligible", "inferred_track", "n_inference_spikes"]
    if (full.groupby(["event_id", "split"])[cols].nunique(dropna=False) > 1).any().any():
        raise ValueError("full reference changed across repeats")
    full = full[full.repeat.eq(0)].copy()
    full["repeat"] = -1
    scores = pd.concat([full, x[x.fraction.eq(0.5)]], ignore_index=True)
    scores["truth_track"] = np.nan
    scores["fold"] = -1
    scores["time_block"] = np.floor(scores.start_s / 60).astype(int)
    fixed = load_fixed_evaluation(source / "evaluation_identity_control_v1", file_sha256(bank / "manifest.json"), session, parts).reset_index()
    fixed = fixed[fixed.event_id.isin(windows.event_id)].copy()
    fixed["B_track2_mass"] = expit(-fixed.conditional_observed_log_odds)
    fixed["B_track2_z"] = -fixed.conditional_identity_z_log_odds

    def observation(eid, fold, split, repeat):
        a = np.array(parts["splits"][split]["inference"])
        ids = a if repeat == -1 else nested_subsets(a, session, split, repeat)[0.5]
        return counts[offsets[eid] : offsets[eid + 1]][:, ids], maps["rates"][:, ids], maps["valid_bins"], 0.02

    return (
        windows,
        scores,
        fixed,
        observation,
        {
            "bank": str(bank),
            "bank_manifest_sha256": file_sha256(bank / "manifest.json"),
            "score_manifests": sources,
            "evaluation_manifest_sha256": file_sha256(source / "evaluation_identity_control_v1" / "manifest.json"),
        },
    )


def enrich(windows, scores, content, observation):
    keys = ["event_id", "split", "repeat"]
    ix = pd.MultiIndex.from_product([sorted(windows.event_id), range(5), range(-1, 5)], names=keys)
    if len(scores) != len(ix) or scores.duplicated(keys).any() or len(ix.difference(pd.MultiIndex.from_frame(scores[keys]))):
        raise ValueError("incomplete frozen score cube")
    bi = pd.MultiIndex.from_product([sorted(windows.event_id), range(5)], names=["event_id", "split"])
    if len(content) != len(bi) or content.duplicated(bi.names).any() or len(bi.difference(pd.MultiIndex.from_frame(content[bi.names]))):
        raise ValueError("incomplete B readouts")
    out = []
    for r in scores.sort_values(keys).to_dict("records"):
        c, rates, valid, duration = observation(int(r["event_id"]), int(r["fold"]), int(r["split"]), int(r["repeat"]))
        if c.sum() != r["n_inference_spikes"]:
            raise ValueError("changed inference observations")
        p, q = track_masses(c, rates, valid, duration)
        out.append(
            {
                **{k: r[k] for k in [*META, "split", "repeat", "sequence_accepted", "sequence_eligible", "inferred_track", "n_inference_spikes"]},
                "A_poisson_track2_mass": p,
                "A_conditional_track2_mass": q,
            }
        )
    return pd.DataFrame(out).merge(content[["event_id", "split", "B_track2_mass", "B_track2_z"]], on=["event_id", "split"], validate="many_to_one")


def paired_groups(rows):
    full = rows[rows.repeat.eq(-1)].set_index(["event_id", "split"])
    half = rows[rows.repeat.ge(0)].copy()
    f = full.reindex(pd.MultiIndex.from_frame(half[["event_id", "split"]])).reset_index(drop=True)
    h = half.reset_index(drop=True)
    if f.sequence_accepted.isna().any():
        raise ValueError("missing paired full label")
    for col in ["B_track2_mass", "B_track2_z"]:
        if not np.allclose(f[col], h[col], equal_nan=True):
            raise ValueError("B changes under thinning")
    af, ah = f.sequence_accepted.to_numpy(bool), h.sequence_accepted.to_numpy(bool)
    for name, use in zip(GROUPS, [af, ah, af & ah, af & ~ah, ~af & ah], strict=True):
        yield name, f if name in ["full", "lost"] else h, f, h, use


def contributions(rows):
    out = []
    for name, a, f, h, use in paired_groups(rows):
        label = a.inferred_track.to_numpy()
        label_valid = np.isin(label, [1, 2])
        if not label_valid[use].all():
            raise ValueError("accepted group lacks label")
        z = h[META].copy()
        z["group"] = name

        def add(metric, value, valid, use=use, z=z):
            valid = np.asarray(valid, bool) & use
            z[metric + "_num"] = np.where(valid, value, 0.0)
            z[metric + "_den"] = valid.astype(float)

        z["selection_mass_num"] = use.astype(float)
        z["selection_mass_den"] = 1.0
        add("sequence_label_track2_fraction", label == 2, label_valid)
        bp = preference(a.B_track2_mass)
        ap = {}
        for key in ["poisson", "conditional"]:
            q = a["A_" + key + "_track2_mass"].to_numpy()
            ap[key] = preference(q)
            add("A_" + key + "_agreement", ap[key] == label, np.isfinite(ap[key]))
            add("A_" + key + "_label_mass", np.where(label == 2, q, 1 - q), np.isfinite(q))
            add("both_oppose_" + key, (ap[key] != label) & (bp != label), np.isfinite(ap[key]) & np.isfinite(bp))
        add("B_agreement", bp == label, np.isfinite(bp))
        add("B_label_mass", np.where(label == 2, a.B_track2_mass, 1 - a.B_track2_mass), np.isfinite(a.B_track2_mass))
        add("B_signed_z", np.where(label == 2, a.B_track2_z, -a.B_track2_z), np.isfinite(a.B_track2_z))
        add("label_behavior_agreement", label == a.truth_track, np.isfinite(a.truth_track))
        add("retained_label_switch", f.inferred_track.ne(h.inferred_track), use if name == "retained" else np.zeros(len(h), bool))
        cols = [c for c in z if c.endswith(("_num", "_den"))]
        z = z.groupby([*META, "group"], dropna=False)[cols].mean().reset_index()
        out.append(z)
    return pd.concat(out, ignore_index=True)


def confusion(rows):
    out = []
    for group, a, _f, h, use in paired_groups(rows):

        def weight(values, event_ids=h.event_id):
            return pd.Series(values.astype(float)).groupby(event_ids).mean().sum()

        total = weight(use)
        references = {"behavior": a.truth_track.to_numpy(), **{k: preference(a[k + "_track2_mass"]) for k in ["A_poisson", "A_conditional", "B"]}}
        for reference, values in references.items():
            values = np.where(np.isfinite(values), values, -1)
            for label in [1, 2]:
                for target in [1, 2, -1]:
                    mass = weight(use & a.inferred_track.eq(label).to_numpy() & (values == target))
                    out.append(
                        {
                            "group": group,
                            "reference": reference,
                            "sequence_label": label,
                            "reference_label": target,
                            "selection_weight": mass,
                            "total_selection_weight": total,
                            "joint_fraction": mass / total if total else np.nan,
                        }
                    )
    return pd.DataFrame(out)


def summarize(table, windows, domain, session):
    windows = windows.sort_values("event_id").reset_index(drop=True)
    n = len(windows)
    if domain == "RUN":
        weights, strata, adequate = run_block_weights(windows.rename(columns={"event_id": "window_id"}), stable_seed(20260918, session, "label-audit-RUN-blocks"))
        blocks = np.array([f"{int(t)}:{int(f)}:{int(b)}" for t, f, b in windows[["truth_track", "fold", "time_block"]].to_numpy()])
    else:
        blocks = windows.time_block.to_numpy()
        unique, ix = np.unique(blocks, return_inverse=True)
        weights = np.random.default_rng(stable_seed(20260918, session, "label-audit-POST-blocks")).multinomial(len(unique), np.full(len(unique), 1 / len(unique)), size=2000)[:, ix]
        strata = pd.DataFrame([{"n_windows": n, "n_blocks": len(unique)}])
        adequate = len(unique) >= 2
    rows = []
    cache = {}

    def ratio(a, b):
        return np.divide(a, b, out=np.full_like(a, np.nan, dtype=float), where=b > 0)

    def make(group, metric, point, draws, informative, denom):
        finite = draws[np.isfinite(draws)]
        ok = adequate and informative >= 2 and len(finite) >= 1900
        lo, hi = np.quantile(finite, [0.025, 0.975]) if ok else (np.nan, np.nan)
        return {
            "session": session,
            "domain": domain,
            "group": group,
            "metric": metric,
            "estimate": float(point),
            "ci025": lo,
            "ci975": hi,
            "n_windows": n,
            "effective_denominator": float(denom),
            "informative_blocks": informative,
            "finite_bootstrap_fraction": len(finite) / 2000,
            "interval_supported": bool(ok),
        }

    for group in GROUPS:
        a = table[table.group.eq(group)].set_index("event_id").reindex(windows.event_id)
        for metric in METRICS:
            num = a[metric + "_num"].to_numpy()
            den = a[metric + "_den"].to_numpy()
            if not np.isfinite(num).all() or not np.isfinite(den).all():
                raise ValueError("nonfinite window contributions")
            point = num.sum() / den.sum() if den.sum() > 0 else np.nan
            draws = ratio(weights @ num, weights @ den)
            informative = len(np.unique(blocks[den > 0]))
            rows.append(make(group, metric, point, draws, informative, den.sum()))
            cache[group, metric] = (point, draws, informative)
    for metric in ["sequence_label_track2_fraction", "A_poisson_agreement", "A_conditional_agreement", "B_agreement", "label_behavior_agreement"]:
        f, fb, fi = cache["full", metric]
        h, hb, hi = cache["half", metric]
        rows.append(make("half_minus_full", metric, h - f, hb - fb, min(fi, hi), np.nan))
    return pd.DataFrame(rows), strata


def run(source, session, output):
    if session not in ["RAT3_SESS2", "RAT5_SESS2"] or output.exists():
        raise ValueError("new immutable primary-session audit required")
    git = lambda *a: subprocess.check_output(["git", *a], cwd=ROOT, text=True).strip()
    commit = git("rev-parse", "HEAD")
    if git("status", "--porcelain"):
        raise ValueError("freeze before auditing labels")
    sources = {}
    summary = []
    confusions = []
    audit_counts = {}
    output.mkdir(parents=True)
    for domain, loader in [("RUN", load_run), ("POST", load_rest)]:
        windows, scores, content, observation, inputs = loader(source, session)
        sources[domain] = inputs
        rows = enrich(windows, scores, content, observation)
        a = contributions(rows)
        s, strata = summarize(a, windows, domain, session)
        summary.append(s)
        cf = confusion(rows)
        cf["domain"] = domain
        cf["session"] = session
        confusions.append(cf)
        rows.to_csv(output / (domain + "_label_rows.csv"), index=False)
        a.to_csv(output / (domain + "_event_contributions.csv"), index=False)
        strata.to_csv(output / (domain + "_bootstrap_strata.csv"), index=False)
        audit_counts[domain] = {"events": len(windows), "subset_rows": len(rows), "unchanged_sequence_decisions": len(scores), "independent_readouts": len(content)}
        print(json.dumps({"session": session, "domain": domain, **audit_counts[domain]}), flush=True)
    table = pd.concat(summary, ignore_index=True)
    table.to_csv(output / "label_agreement_summary.csv", index=False)
    pd.concat(confusions, ignore_index=True).to_csv(output / "label_confusion.csv", index=False)
    text = [
        "# Sequence-label diagnostic",
        "",
        "Original sequence decisions are unchanged. Mean per-bin track mass is not a calibrated event probability or an experience prevalence.",
        "RUN has behavioral track labels; POST has no replay ground truth. Population agreement is not accuracy or proof of replay.",
        "",
    ]
    for domain in ["RUN", "POST"]:
        text.append("## " + domain)
        for group in ["full", "half", "lost", "gained"]:
            for metric in ["A_poisson_agreement", "B_agreement", "both_oppose_poisson"]:
                r = table[(table.domain == domain) & (table.group == group) & (table.metric == metric)].iloc[0]
                text.append(f"- {group} {metric}: {r.estimate:.6f} [{r.ci025:.6f}, {r.ci975:.6f}]; denominator={r.effective_denominator:.3f}")
    text += [
        "",
        "No new thresholds, detections, sequence nulls, selection rules or biological prevalence corrections.",
        "Any proposed replacement rule requires new held-out calibration before application to sleep replay.",
    ]
    (output / "sequence_label_audit.md").write_text("\n".join(text) + "\n")
    if git("status", "--porcelain") or git("rev-parse", "HEAD") != commit:
        raise ValueError("code changed during audit")
    m = {
        "status": "complete",
        "session": session,
        "code_commit": commit,
        "git_dirty": False,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "command_line": sys.argv,
        "source_root": str(source.resolve()),
        "inputs": sources,
        "counts": audit_counts,
        "original_sequence_decisions_unchanged": True,
        "new_sequence_scoring": False,
        "new_A_sequenceless_readouts": True,
        "B_readouts_reused": True,
        "behavior_truth_only_for_RUN": True,
        "biological_replay_bias_established": False,
        "output_sha256": {p.name: file_sha256(p) for p in output.iterdir()},
    }
    (output / "manifest.json").write_text(json.dumps(m, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-dir", type=Path, required=True)
    p.add_argument("--session", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    run(a.source_dir, a.session, a.output_dir)
