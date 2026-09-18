"""Independently regenerate a stratified calibration subset and reconstruct scores."""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.verify_tirole_content_bank import digest, direct_correlations, direct_posterior, seed
from scripts.verify_tirole_future_forecasts import reference_null, reference_scores


def draw(totals, weights, rng):
    return np.vstack([rng.multinomial(int(n), w / w.sum()) for n, w in zip(totals, weights, strict=True)])


def path(valid, n, rng):
    runs = np.split(np.flatnonzero(valid.all(axis=0)), np.flatnonzero(np.diff(np.flatnonzero(valid.all(axis=0))) != 1) + 1)
    longest = max(runs, key=len)
    lo, hi = int(longest[0]), int(longest[-1] + 1)
    span = int(rng.integers(3, hi - lo))
    start = int(rng.integers(lo, hi - span))
    p = np.rint(np.linspace(start, start + span, n)).astype(int)
    return p if rng.random() < 0.5 else p[::-1]


def check_close(a, b):
    if not np.allclose(a, b, atol=1e-8, rtol=1e-10, equal_nan=True):
        raise ValueError("independent reconstruction mismatch")
    err = np.abs(np.asarray(a) - np.asarray(b))
    return float(np.max(err[np.isfinite(err)], initial=0))


def direct_content(c, rates, valid, swaps, conditional):
    c = c[c.sum(axis=1) > 0]
    if not len(c):
        return np.nan, np.nan
    odds = []
    for swap in np.vstack([np.zeros(len(swaps[0]), bool), swaps]):
        lam = rates.copy()
        lam[:, swap] = rates[::-1, swap]
        p = direct_posterior(c, lam, valid, conditional)
        mass = p.sum(axis=(0, 2))
        odds.append(np.log(max(mass[0], 1e-300)) - np.log(max(mass[1], 1e-300)))
    sd = np.std(odds[1:], ddof=1)
    return odds[0], (odds[0] - np.mean(odds[1:])) / sd if sd > 1e-12 else np.nan


def run(bank, forecast, calibration, output):
    if output.exists():
        raise ValueError("new verifier output required")
    cm = json.loads((calibration / "manifest.json").read_text())
    bm = json.loads((bank / "manifest.json").read_text())
    fm = json.loads((forecast / "manifest.json").read_text())
    if (
        cm["status"] != "complete"
        or cm["git_dirty"]
        or cm["bank_manifest_sha256"] != digest(bank / "manifest.json")
        or cm["source_forecast_manifest_sha256"] != digest(forecast / "manifest.json")
    ):
        raise ValueError("invalid provenance")
    for base, manifest, key in [(bank, bm, "outputs_sha256"), (forecast, fm, "output_sha256"), (calibration, cm, "output_sha256")]:
        for name, h in manifest[key].items():
            if digest(base / name) != h:
                raise ValueError("source checksum mismatch")
    anchors = pd.read_csv(calibration / "count_anchors.csv")
    events = pd.read_csv(bank / "candidate_events.csv")
    expected = []
    for _, g in events[events.ripple_supported].groupby("epoch"):
        expected.extend(sorted(g.event_id, key=lambda e: seed(20260918, bm["session"], e, "measurement-calibration-anchor"))[:20])
    if set(anchors.event_id) != set(expected):
        raise ValueError("score-dependent or wrong anchor selection")
    content = pd.read_csv(calibration / "known_track_calibration.csv")
    temporal = pd.read_csv(calibration / "known_temporal_calibration.csv")
    arrays = np.load(bank / "event_counts.npz")
    maps = np.load(bank / "RUN_maps.npz")
    counts, offsets = arrays["counts"], arrays["offsets"]
    rates, valid, centers = maps["rates"], maps["valid_bins"], maps["bin_centers_cm"]
    parts = json.loads((bank / "partitions.json").read_text())["splits"]
    selected = set(anchors.groupby("epoch").head(2).event_id)
    checks, seqchecks, maxerr = 0, 0, 0.0
    session = bm["session"]
    for rank, event in enumerate(anchors.sort_values("event_id").to_dict("records")):
        eid = int(event["event_id"])
        if eid not in selected:
            continue
        orig = counts[offsets[eid] : offsets[eid + 1]]
        for split, part in enumerate(parts):
            train, held = np.array(part["inference"]), np.array(part["evaluation"])
            if set(train) & set(held):
                raise ValueError("cell-role overlap")
            repeat = (rank + split) % 5
            order = np.random.default_rng(seed(20260918, session, split, repeat, "coverage")).permutation(train)
            truthpath = path(valid, len(orig), np.random.default_rng(seed(20260918, session, eid, split, "known-path")))
            for track in range(2):
                rng = np.random.default_rng(seed(20260918, session, eid, split, track, "known-track-spikes"))
                generated = np.zeros_like(orig)
                for ids in (train, held):
                    generated[:, ids] = draw(orig[:, ids].sum(axis=1), np.maximum(rates[track][:, truthpath].T[:, ids], 1e-4), rng)
                swaps = rng.integers(0, 2, (199, len(held))).astype(bool)
                ref = [direct_content(generated[:, held], rates[:, held], valid, swaps, v) for v in [False, True]]
                shuffled = generated[rng.permutation(len(orig))]
                for generator, c in [("ordered", generated), ("whole_bin_shuffled", shuffled)]:
                    nrng = np.random.default_rng(seed(20260918, session, eid, split, track, generator, "sequence-nulls"))
                    shifts = nrng.integers(0, len(centers), (499, 2, counts.shape[1]))
                    perms = np.argsort(nrng.random((499, len(c))), axis=1)
                    for fraction in [1.0, 0.5]:
                        ids = np.sort(order[: int(np.ceil(len(order) * fraction))])
                        r = content[
                            (content.anchor_event == eid)
                            & (content.split == split)
                            & (content.truth_track == track + 1)
                            & (content.generator == generator)
                            & (content.fraction == fraction)
                        ]
                        if len(r) != 1:
                            raise ValueError("missing or duplicate content row")
                        r = r.iloc[0]
                        cc = c[:, ids]
                        active = (cc.sum(axis=0) > 0).sum()
                        nonempty = cc.sum(axis=1) > 0
                        eligible = active >= 5 and nonempty.sum() >= 5
                        if r.sequence_eligible != eligible or cc.sum() != r.n_inference_spikes or c[:, held].sum() != r.n_evaluation_spikes:
                            raise ValueError("generated counts or opportunity mismatch")
                        sign = 1 - 2 * track
                        for val, prefix in zip(ref, ["evaluation", "conditional"], strict=True):
                            maxerr = max(maxerr, check_close(sign * np.array(val), [r[prefix + "_true_signed_log_odds"], r[prefix + "_true_signed_z"]]))
                        checks += 1
                        if eligible and seqchecks < 8:
                            lam = rates[:, ids]
                            post = direct_posterior(cc, lam, valid)
                            post[~nonempty] = 0
                            corr = direct_correlations(post, centers)
                            tnull = np.array([direct_correlations(post[p], centers) for p in perms])
                            fnull = []
                            for shift in shifts:
                                rolled = np.array([[np.roll(lam[k, j], shift[k, uid]) for j, uid in enumerate(ids)] for k in range(2)])
                                p = direct_posterior(cc, rolled, valid)
                                p[~nonempty] = 0
                                fnull.append(direct_correlations(p, centers))
                            pt = (1 + (tnull >= corr - 1e-12).sum(axis=0)) / 500
                            pf = (1 + (np.array(fnull) >= corr - 1e-12).sum(axis=0)) / 500
                            check_close(np.r_[pt, pf], r[["track1_p_time", "track2_p_time", "track1_p_field", "track2_p_field"]].to_numpy(float))
                            passed = (pt < 0.025) & (pf < 0.025)
                            best = np.argmax(np.where(passed, corr, -np.inf)) if passed.any() else np.argmax(corr)
                            if r.sequence_accepted != bool(passed.any()) or r.inferred_track != best + 1:
                                raise ValueError("wrong sequence classification")
                            seqchecks += 1
    folds = json.loads((forecast / "folds.json").read_text())
    nforecasts = 0
    for record in folds:
        fit = np.load(forecast / f"fit_{record['fold']}.npz")
        pop = fit["population"]
        null = reference_null(fit["transition"])
        ids = sorted(set(record["test_ids"]) & set(anchors.event_id))[:2]
        for eid in ids:
            orig = counts[offsets[eid] : offsets[eid + 1]][:, pop]
            for split in [0, 2, 4]:
                part = parts[split]
                repeat = 0
                order = np.random.default_rng(seed(20260918, session, split, repeat, "coverage")).permutation(part["inference"])
                train = np.searchsorted(pop, np.sort(order[: int(np.ceil(len(order) * 0.5))]))
                held = np.searchsorted(pop, part["evaluation"])
                for generator, matrix in [("dynamic", fit["transition"]), ("matched_null", null)]:
                    rng = np.random.default_rng(seed(20260918, session, eid, split, repeat, "known-temporal-generator"))
                    state = int(rng.choice(len(fit["initial"]), p=fit["initial"]))
                    states = [state]
                    for _ in range(len(orig) - 1):
                        state = int(rng.choice(len(matrix), p=matrix[state]))
                        states.append(state)
                    c = np.zeros_like(orig)
                    for use in [train, held]:
                        c[:, use] = draw(orig[:, use].sum(axis=1), fit["probabilities"][use][:, states].T, rng)
                    for h in [1, 2, 4]:
                        if len(c) <= h:
                            continue
                        r = temporal[
                            (temporal.anchor_event == eid) & (temporal.split == split) & (temporal.repeat == repeat) & (temporal.generator == generator) & (temporal.horizon == h)
                        ]
                        if len(r) != 1:
                            raise ValueError("missing predictive row")
                        r = r.iloc[0]
                        if c[h:][:, held].sum() != r.n_heldout_target_spikes:
                            raise ValueError("target counts mismatch")
                        for name, value in reference_scores(c, train, held, fit, null, h).items():
                            maxerr = max(maxerr, check_close(value, r["score_" + name]))
                        nforecasts += 1
    if not checks or not nforecasts:
        raise ValueError("vacuous verification")
    result = {
        "status": "pass",
        "session": session,
        "calibration_manifest_sha256": digest(calibration / "manifest.json"),
        "verifier_sha256": digest(Path(__file__)),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "content_rows_reconstructed": checks,
        "sequence_null_rows_reconstructed": seqchecks,
        "forecast_rows_reconstructed": nforecasts,
        "max_absolute_score_error": maxerr,
        "scope": "all input hashes and score-independent anchors; stratified independently regenerated counts, direct content/null and dense forward predictive scores; no biological validation",
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["bank-dir", "forecast-dir", "calibration-dir", "output"]:
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    run(a.bank_dir, a.forecast_dir, a.calibration_dir, a.output)
