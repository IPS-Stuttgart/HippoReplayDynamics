"""Frozen local-rest cell-rate sensitivity for independent track content."""

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from hipporeplayimm.tirole_two_track import file_sha256, load_session
from hipporeplayimm.two_track_content import stable_seed
from scripts.report_tirole_content_coverage import event_summaries, load_campaigns, summarize_content
from scripts.report_tirole_content_uncertainty import block_weights, mean_interval
from scripts.score_tirole_content_coverage import validate_bank


def rest_masks(session, events):
    t = session.times
    ontrack = np.isfinite(session.positions).any(axis=0)
    first, last = t[ontrack].min(), t[ontrack].max()
    keep = session.sleepbox & np.isfinite(session.speed) & (session.speed <= 5)
    dt = float(np.median(np.diff(t)))
    # The half-sample padding prevents nearest-sample allocation across a guard.
    difference = np.zeros(len(t) + 1, int)
    for e in events.to_dict("records"):
        lo = np.searchsorted(t, e["start_s"] - 1 - dt / 2, side="left")
        hi = np.searchsorted(t, e["end_s"] + 1 + dt / 2, side="right")
        difference[lo] += 1
        difference[hi] -= 1
    keep &= np.cumsum(difference[:-1]) == 0
    return {"PRE": keep & (t < first), "POST": keep & (t > last)}


def baseline_counts(session, mask, lo, hi):
    exposure = float(mask[lo:hi].sum() * np.median(np.diff(session.times)))
    # Spike sample indices are monotonic except for out-of-range -1 tails.
    inside = (session.spike_samples >= lo) & (session.spike_samples < hi)
    ix = np.flatnonzero(inside)
    ix = ix[mask[session.spike_samples[ix]]]
    c = np.bincount(session.spike_units[ix], minlength=len(session.unit_ids))
    return exposure, c


def local_null_counts(observed, baseline, rng, n_null=199):
    c = np.asarray(observed)
    base = np.asarray(baseline, float)
    if c.ndim != 2 or base.shape != (c.shape[1],) or (base < 0).any() or not np.isfinite(base).all() or (c < 0).any() or not np.equal(c, np.floor(c)).all():
        raise ValueError("valid observed counts and outside-event baseline required")
    p = (base + 0.5) / (base + 0.5).sum()
    null = np.stack([rng.multinomial(int(n), p, size=n_null) for n in c.sum(axis=1)], axis=1)
    return np.concatenate([c[None], null], axis=0)


def batch_odds(counts, rates, valid, conditional=False):
    if counts.ndim != 3 or rates.shape[:2] != (2, counts.shape[2]) or valid.shape != (2, rates.shape[2]):
        raise ValueError("incompatible population/map dimensions")
    lam = np.maximum(rates, 1e-4)
    if conditional:
        lam = lam / lam.sum(axis=1, keepdims=True)
    ll = np.einsum("stu,kup->stkp", counts, np.log(lam))
    if not conditional:
        ll -= 0.02 * lam.sum(axis=1)[None, None]
    ll -= np.log(valid.sum(axis=1))[None, None, :, None]
    ll[:, :, ~valid] = -np.inf
    post = np.exp(ll - logsumexp(ll, axis=(-2, -1), keepdims=True))
    post *= (counts.sum(axis=2) > 0)[:, :, None, None]
    mass = post.sum(axis=(1, 3))
    odds = np.log(np.maximum(mass[:, 0], 1e-300)) - np.log(np.maximum(mass[:, 1], 1e-300))
    odds[counts.sum(axis=(1, 2)) == 0] = np.nan
    return odds


def run(source, dataset, output):
    if output.exists():
        raise ValueError("new local-control output directory required")
    git = lambda *a: subprocess.check_output(["git", *a], cwd=ROOT, text=True).strip()
    commit = git("rev-parse", "HEAD")
    if git("status", "--porcelain"):
        raise ValueError("freeze local control before execution")
    campaigns = [source / n for n in ["campaign_RAT3_SESS2_v1", "campaign_diagnostic_stage1_v1", "campaign_remaining_stage2_v1"]]
    scores, manifests = load_campaigns(campaigns)
    preflight = source / "RUN_preflight_v1" / "two_track_preflight_manifest.json"
    pm = json.loads(preflight.read_text())
    rows, inputs = [], {str(preflight): file_sha256(preflight)}
    bank_events, metadata = {}, {}
    for session in sorted(scores.session.unique()):
        bank = source / ("candidate_banks_v2" if session == "RAT3_SESS2" else "candidate_banks_v3") / session
        info, parts, events, counts, offsets, maps = validate_bank(bank)
        bank_events[session] = events.set_index("event_id")
        metadata[session] = info
        digest = file_sha256(bank / "manifest.json")
        inputs[str(bank / "manifest.json")] = digest
        for m in manifests:
            sm = json.loads(Path(m["manifest"]).read_text())
            if sm["session"] == session and sm["bank_manifest_sha256"] != digest:
                raise ValueError("sequence bank mismatch")
        for f in pm["input_files"]:
            if f["session"] == session:
                p = dataset / Path(f["path"]).name
                if file_sha256(p) != f["sha256"]:
                    raise ValueError("changed raw input")
                inputs[str(p)] = f["sha256"]
        data = load_session(dataset, session)
        if not np.array_equal(data.unit_ids, maps["unit_ids"]):
            raise ValueError("raw/bank unit identities differ")
        masks = rest_masks(data, events)
        wanted = set(scores.loc[scores.session == session, "event_id"])
        for event in events[events.event_id.isin(wanted)].to_dict("records"):
            eid = int(event["event_id"])
            mid = 0.5 * (event["start_s"] + event["end_s"])
            for radius in [60, 120]:
                lo, hi = np.searchsorted(data.times, [mid - radius, mid + radius])
                exposure, b = baseline_counts(data, masks[event["epoch"]], lo, hi)
                for split, part in enumerate(parts["splits"]):
                    ids = np.array(part["evaluation"])
                    observed = counts[offsets[eid] : offsets[eid + 1]][:, ids]
                    row = {
                        "session": session,
                        "event_id": eid,
                        "split": split,
                        "radius_s": radius,
                        "baseline_exposure_s": exposure,
                        "n_baseline_spikes": int(b[ids].sum()),
                        "n_baseline_active_cells": int((b[ids] > 0).sum()),
                        "n_evaluation_spikes": int(observed.sum()),
                        "status": "complete" if exposure >= 20 else "insufficient_baseline_exposure",
                    }
                    generated = None
                    if exposure >= 20:
                        generated = local_null_counts(observed, b[ids], np.random.default_rng(stable_seed(20260918, session, eid, split, radius, "local-rest-identity")))
                    for conditional, prefix in [(False, ""), (True, "conditional_")]:
                        odds = np.full(200, np.nan) if generated is None else batch_odds(generated, maps["rates"][:, ids], maps["valid_bins"], conditional)
                        sd = float(np.std(odds[1:], ddof=1))
                        mean = float(np.mean(odds[1:]))
                        row.update(
                            {
                                prefix + "observed_log_odds": float(odds[0]),
                                prefix + "null_mean_log_odds": mean,
                                prefix + "null_sd_log_odds": sd,
                                prefix + "local_z_log_odds": float((odds[0] - mean) / sd) if np.isfinite(sd) and sd > 1e-12 else np.nan,
                            }
                        )
                    rows.append(row)
        print(json.dumps({"session": session, "events_complete": len(wanted)}), flush=True)
    readout = pd.DataFrame(rows)
    summaries, eventframes, uncertainty = [], [], []
    for radius in [60, 120]:
        joined = scores.merge(readout[readout.radius_s == radius], on=["session", "event_id", "split"], validate="many_to_one")
        for prefix in ["", "conditional_"]:
            orig = joined[prefix + "evaluation_log_odds"].to_numpy()
            checked = joined[prefix + "observed_log_odds"].to_numpy()
            good = np.isfinite(checked)
            if not np.allclose(orig[good], checked[good], atol=1e-10):
                raise ValueError("observed readout changed during local-null control")
            joined[prefix + "evaluation_z_log_odds"] = joined[prefix + "local_z_log_odds"]
        es = event_summaries(joined)
        es["radius_s"] = radius
        eventframes.append(es)
        sm = summarize_content(es)
        sm["radius_s"] = radius
        summaries.append(sm)
        for session in sorted(scores.session.unique()):
            info = metadata[session]
            times = bank_events[session]
            for epoch in ["PRE", "POST"]:
                for tier in ["ripple_z3", "ripple_z5", "mua_only"]:
                    d = es[
                        (es.session == session)
                        & (es.epoch == epoch)
                        & (es.fraction == 0.5)
                        & (es.group == "lost")
                        & (es.candidate_stratum == ("mua_only" if tier == "mua_only" else "ripple"))
                    ]
                    if tier == "ripple_z5":
                        d = d[d.ripple_peak_z >= 5]
                    if d.empty:
                        continue
                    codes, weights = block_weights(times.loc[d.event_id, "start_s"], 60, stable_seed(20260918, session, epoch, tier, "local-null-blocks"))
                    for name in ["signed_evaluation_z", "signed_conditional_evaluation_z"]:
                        uncertainty.append(
                            {
                                "session": session,
                                "animal": info["animal"],
                                "cohort_stratum": info["cohort_stratum"],
                                "epoch": epoch,
                                "candidate_tier": tier,
                                "radius_s": radius,
                                "metric": name,
                                **mean_interval(d[name].to_numpy(), codes, weights),
                            }
                        )
    if git("status", "--porcelain") or git("rev-parse", "HEAD") != commit:
        raise ValueError("code changed during control")
    output.mkdir(parents=True)
    for name, frame in {
        "local_rest_by_event_split": readout,
        "local_rest_event_summary": pd.concat(eventframes),
        "local_rest_content_summary": pd.concat(summaries),
        "local_rest_conditional_intervals": pd.DataFrame(uncertainty),
    }.items():
        frame.to_csv(output / (name + ".csv"), index=False)
    m = {
        "status": "complete",
        "code_commit": commit,
        "git_dirty": False,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "command_line": sys.argv,
        "source_sha256": inputs,
        "source_score_manifests": manifests,
        "n_unique_events": int(readout[["session", "event_id"]].drop_duplicates().shape[0]),
        "baseline_radii_s": [60, 120],
        "minimum_exposure_s": 20,
        "guard_s": 1,
        "pseudocount_per_cell": 0.5,
        "n_null": 199,
        "changes_selection": False,
        "sleep_stage_control": False,
        "biological_confirmation": False,
        "output_sha256": {p.name: file_sha256(p) for p in output.iterdir()},
    }
    (output / "manifest.json").write_text(json.dumps(m, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["source-dir", "dataset-root", "output-dir"]:
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    run(a.source_dir, a.dataset_root, a.output_dir)
