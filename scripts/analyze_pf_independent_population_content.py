#!/usr/bin/env python3
"""RUN-frozen disjoint-population agreement and conditional known-path controls."""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.special import softmax
from scipy.sparse import csr_matrix
from scipy.stats import ks_2samp

from scripts._provenance import build_script_provenance, file_sha256
from scripts.analyze_pf_matched_population_content import descriptors_match, quality_match, run_error
from scripts.analyze_replay_coverage_subsampling import event_windows
from scripts.analyze_replay_coverage_shuffle_baseline import stable_seed

COHORT = "all_fixed_candidates"
GENERATORS = ("matched_map", "drift_gain")


def load_checked(path, expected=None):
    if not Path(path).is_file() or (expected is not None and file_sha256(path) != expected):
        raise ValueError(f"input missing or changed: {path}")
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def regions(grid, reference=None):
    reference = grid if reference is None else reference
    scaled = (grid-reference.min(axis=0))/np.maximum(np.ptp(reference, axis=0), 1)
    tile = np.clip((3*scaled).astype(int), 0, 2)
    return tile[:, 0]*3+tile[:, 1]


def profile(rates, region):
    values = np.array([rates[:, region == i].mean() if np.any(region == i) else 0 for i in range(9)])
    return values/max(values.sum(), 1e-12)


def profile_gap(a, b):
    return float(np.max(np.abs(a-b)/np.maximum((a+b)/2, 1e-12)))


def balanced_partitions(eligible, rates, grid, seed, n_candidates=128):
    """Pair cells by RUN map shape and peak, then randomize assignment within pairs."""
    rng = np.random.default_rng(seed)
    normalized = rates[eligible]/np.maximum(np.linalg.norm(rates[eligible], axis=1)[:, None], 1e-12)
    peak = grid[rates[eligible].argmax(axis=1)]
    distance = cdist(normalized, normalized, "sqeuclidean") + cdist(peak/40, peak/40, "sqeuclidean")
    np.fill_diagonal(distance, np.inf)
    available = set(range(len(eligible)))
    pairs = []
    while len(available) >= 2:
        i = int(rng.choice(sorted(available)))
        available.remove(i)
        j = min(available, key=lambda j: (distance[i, j], j))
        available.remove(j)
        pairs.append([eligible[i], eligible[j]])
    pairs = np.array(pairs)
    output = []
    for candidate in range(n_candidates):
        side = rng.integers(0, 2, len(pairs))
        output.append(dict(candidate_id=candidate,
            a=np.sort(pairs[np.arange(len(pairs)), side]).tolist(),
            b=np.sort(pairs[np.arange(len(pairs)), 1-side]).tolist()))
    return output


def descriptors(units, indices, error):
    u = units.loc[indices]
    return dict(n_cells=len(indices), mean_rate_hz=float(u.mean_rate_hz.mean()),
        median_stability=float(u.stability.median()), median_area_cm2=float(u.area_cm2.median()),
        match_median_cm=float(np.median(error)), match_p75_cm=float(np.quantile(error, .75)))


def freeze_session(rec, args):
    session = rec["session"]
    source = args.matched_dir/session.replace("/", "_")
    checkpoint = json.loads((source/"freeze_checkpoint.json").read_text())
    data = load_checked(source/"run_validation.npz", checkpoint["run_validation_sha256"])
    units = pd.read_csv(source/"early_unit_descriptors.csv")
    eligible = np.flatnonzero(units.eligible.astype(bool))
    parts = balanced_partitions(eligible, data["early_rates"], data["grid_cm"], stable_seed(args.seed, session))
    region = regions(data["grid_cm"])
    candidates, qualified = [], []
    for part in parts:
        desc = {}
        for side in ("a", "b"):
            ix = part[side]
            err = run_error(data["match_counts"][:, ix], data["early_rates"][ix], data["grid_cm"], data["match_windows"][:, 2:])
            desc[side] = descriptors(units, ix, err)
        gap = profile_gap(*[profile(data["early_rates"][part[s]], region) for s in ("a", "b")])
        passed = descriptors_match(desc["a"], desc["b"]) and quality_match(desc["a"], desc["b"]) and gap <= .2
        candidates.append(dict(candidate_id=part["candidate_id"], gap=gap, passed=passed,
            **{f"{side}_{key}": val for side in desc for key, val in desc[side].items()}))
        if passed:
            qualified.append(dict(**part, gap=gap, descriptors=desc))
    folder = args.output_dir/session.replace("/", "_")
    folder.mkdir()
    pd.DataFrame(candidates).to_csv(folder/"matching_candidates.csv", index=False)
    selected = sorted(qualified, key=lambda p: (p["gap"], p["candidate_id"]))[:args.splits]
    frozen = dict(session=session, animal=rec["animal"], selected=selected,
        created_at_utc=datetime.now(UTC).isoformat(), no_replay_used=True,
        run_cache_sha256=file_sha256(source/"run_validation.npz"),
        unit_descriptors_sha256=file_sha256(source/"early_unit_descriptors.csv"))
    path = folder/"selected_before_confirmation.json"
    path.write_text(json.dumps(frozen, indent=2))
    before = file_sha256(path)
    run_rows, local_rows = [], []
    truth_region = regions(data["confirm_windows"][:, 2:], data["grid_cm"])
    for split, part in enumerate(selected):
        errors = {}
        for side in ("a", "b"):
            ix = part[side]
            errors[side] = run_error(data["confirm_counts"][:, ix], data["early_rates"][ix], data["grid_cm"], data["confirm_windows"][:, 2:])
            part["descriptors"][side].update(confirm_median_cm=float(np.median(errors[side])),
                confirm_p75_cm=float(np.quantile(errors[side], .75)))
        passed = quality_match(part["descriptors"]["a"], part["descriptors"]["b"], "confirm")
        supported, matched = 0, 0
        for region_id in range(9):
            mask = truth_region == region_id
            n = int(mask.sum())
            a, b = [float(np.median(errors[s][mask])) if n else np.nan for s in ("a", "b")]
            ok = n >= 10 and abs(a-b) <= 5
            supported += n >= 10
            matched += ok
            local_rows.append(dict(session=session, animal=rec["animal"], split=split,
                region=region_id, n_windows=n, a_error_cm=a, b_error_cm=b, local_match=ok))
        part.update(split=split, confirmed=passed, local_match_all=supported == matched == 9)
        run_rows.append(dict(session=session, animal=rec["animal"], split=split,
            candidate_id=part["candidate_id"], confirmed=passed, n_cells=len(part["a"]), gap=part["gap"],
            disjoint=not bool(set(part["a"]) & set(part["b"])), local_regions_supported=supported,
            local_regions_matched=matched, local_match_all=part["local_match_all"],
            **{f"{s}_{k}":v for s in ("a", "b") for k, v in part["descriptors"][s].items()}))
    if before != file_sha256(path):
        raise ValueError("frozen partition changed during confirmation")
    if not selected:
        run_rows.append(dict(session=session, animal=rec["animal"], split=-1, confirmed=False,
            status="no_matching_partition", disjoint=False, local_match_all=False))
    state = dict(**frozen, selection_sha256=before, confirmation=selected)
    (folder/"confirmation.json").write_text(json.dumps(state, indent=2))
    print(session, "RUN matching", len(qualified), "confirmed", sum(p["confirmed"] for p in selected), flush=True)
    return state, run_rows, local_rows


def decode(counts, rates, grid):
    logp = csr_matrix(counts, dtype=float)@np.log(rates)-.02*rates.sum(axis=0)
    p = softmax(logp, axis=1)
    mean = p@grid
    width = np.sqrt(np.maximum(p@np.square(grid).sum(axis=1)-np.square(mean).sum(axis=1), 0))
    entropy = -(p*np.log(np.maximum(p, 1e-300))).sum(axis=1)/np.log(p.shape[1])
    return dict(p=p, mean=mean, width=width, entropy=entropy)


def longest_run(mask):
    edge = np.diff(np.r_[False, mask, False].astype(int))
    starts, ends = np.flatnonzero(edge == 1), np.flatnonzero(edge == -1)
    return int(np.max(ends-starts)) if len(starts) else 0


def endpoint_readout(da, db, ca, cb, grid, endpoint, local_error, local_tree, rates_a, truth_a=None, truth_b=None):
    i = int(endpoint)
    if not 0 <= i < len(ca) or len(ca) != len(cb):
        raise ValueError("invalid frozen endpoint")
    pa, pb = da["p"][i], db["p"][i]
    ma, mb = da["mean"][i], db["mean"][i]
    distance = np.linalg.norm(grid-ma, axis=1)
    rdist, ridx = local_tree.query(ma, k=min(20, len(local_error)))
    keep = rdist <= 40
    near_error = float(np.median(local_error[ridx[keep]])) if keep.sum() >= 5 else np.nan
    sep = np.linalg.norm(da["mean"][:i+1]-db["mean"][:i+1], axis=1)
    resolved = (sep >= 40) & (da["width"][:i+1] <= 20) & (db["width"][:i+1] <= 20)
    nearest = int(np.argmin(distance))
    return dict(a_spikes=int(ca[i].sum()), b_spikes=int(cb[i].sum()),
        a_active=int((ca[i]>0).sum()), b_active=int((cb[i]>0).sum()), n_cells=ca.shape[1],
        a_entropy=float(da["entropy"][i]), a_width_cm=float(da["width"][i]), b_width_cm=float(db["width"][i]),
        a_peak=float(pa.max()), a_local_run_error_cm=near_error,
        a_stability_cm=float(np.median(np.linalg.norm(da["mean"][max(0,i-3):i+1]-ma, axis=1))),
        a_coverage=float(rates_a[:, nearest].sum()/max(rates_a.sum(axis=0).mean(), 1e-12)),
        b_mass20=float(pb[distance <= 20].sum()), b_mass40=float(pb[distance <= 40].sum()),
        posterior_overlap=float(np.minimum(pa,pb).sum()), endpoint_separation_cm=float(np.linalg.norm(ma-mb)),
        path_median_separation_cm=float(np.median(sep)), persistent_conflict_ms=5*longest_run(resolved),
        resolved_disagreement=bool(resolved[-1]), a_x_cm=float(ma[0]), a_y_cm=float(ma[1]),
        b_x_cm=float(mb[0]), b_y_cm=float(mb[1]),
        a_truth_error_cm=float(np.linalg.norm(ma-truth_a[i])) if truth_a is not None else np.nan,
        b_truth_error_cm=float(np.linalg.norm(mb-truth_b[i])) if truth_b is not None else np.nan,
        truth_separation_cm=float(np.linalg.norm(truth_a[i]-truth_b[i])) if truth_a is not None else np.nan)


def reflected_path(grid, durations, rng):
    start = grid[rng.integers(len(grid))]
    speed = float(rng.choice([0,100,300,600,1200]))
    angle = rng.uniform(0, 2*np.pi)
    t = np.cumsum(durations)-durations/2
    raw = start+speed*t[:, None]*np.array([np.cos(angle), np.sin(angle)])
    lo, span = grid.min(axis=0), np.maximum(np.ptp(grid, axis=0), 1)
    folded = (raw-lo) % (2*span)
    xy = lo+np.minimum(folded, 2*span-folded)
    return xy, speed


def simulate_counts(totals, rates, state, rng, gains=None):
    weights = rates[:, state].T.copy()
    if gains is not None:
        weights *= gains[None, :]
    weights /= weights.sum(axis=1, keepdims=True)
    counts = np.array([rng.multinomial(int(n), p) for n,p in zip(totals, weights, strict=True)], dtype=int)
    if not np.array_equal(counts.sum(axis=1), totals):
        raise ValueError("conditional simulator changed base spike totals")
    return counts


def truth_frames(xy, durations):
    # Same overlapping support as event_windows, without its integer count cast.
    n = int(np.isclose(durations, .005, atol=1e-9, rtol=0).sum())
    if n < 4:
        return np.empty((0, 2))
    cumulative = np.vstack([np.zeros((1, 2)), np.cumsum(xy[:n], axis=0)])
    return (cumulative[4:]-cumulative[:-4])/4


def score_session(rec, state, args, previous):
    session = rec["session"]
    folder = args.output_dir/session.replace("/", "_")
    if file_sha256(folder/"selected_before_confirmation.json") != state["selection_sha256"]:
        raise ValueError("partitions changed after freeze")
    run = load_checked(args.matched_dir/session.replace("/", "_")/"run_validation.npz", state["run_cache_sha256"])
    arrays = load_checked(rec["input_arrays_path"], rec["input_arrays_sha256"])
    source = load_checked(rec["source"]["source_cache_path"], rec["source"]["source_cache_sha256"])
    second = source["rates_second_half_hz"][source["unit_qc_mask"].astype(bool)][:, arrays["support"]]
    early, grid = run["early_rates"], arrays["grid_cm"]
    if not np.array_equal(run["cell_ids"], arrays["cell_ids"]) or not np.allclose(run["grid_cm"], grid):
        raise ValueError("RUN/replay cell or grid alignment mismatch")
    events = previous[previous.session.eq(session) & previous.population_replicate.eq(0)]
    lookup = {key:i for i,key in enumerate(arrays["window_uids"])}
    tree, local_tree = cKDTree(grid), cKDTree(run["match_windows"][:, 2:])
    rows, audit = [], []
    for part in state["confirmation"]:
        if not part["confirmed"]:
            continue
        ia, ib = part["a"], part["b"]
        if len(ia) != len(ib) or set(ia) & set(ib):
            raise ValueError("populations not equal and disjoint")
        ea = run_error(run["match_counts"][:, ia], early[ia], grid, run["match_windows"][:, 2:])
        split = part["split"]
        for event in events.itertuples():
            w = lookup[event.window_uid]
            fa, fb = arrays["frame_offsets"][2*w:2*w+2]
            ba, bb = arrays["base_offsets"][w:w+2]
            counts = arrays["frame_counts"][fa:fb]
            base, durations = arrays["base_counts"][ba:bb], arrays["base_durations_s"][ba:bb]
            if not np.array_equal(event_windows(base, durations), counts):
                raise ValueError("overlapping frame reconstruction failed")
            ca, cb = counts[:,ia], counts[:,ib]
            da, db = decode(ca,early[ia],grid), decode(cb,early[ib],grid)
            identity = dict(animal=rec["animal"], session=session, window_uid=event.window_uid, split=split)
            endpoints = [(COHORT,event.fixed_endpoint_frame)]
            if event.accepted_full:
                endpoints.append(("full_accepted_segment",event.full_segment_endpoint_frame))
            for cohort, endpoint in endpoints:
                rows.append(dict(**identity, cohort=cohort, source="real", generator="observed", draw=-1,
                    endpoint_frame=int(endpoint), projection_error_cm=np.nan,
                    **endpoint_readout(da,db,ca,cb,grid,endpoint,ea,local_tree,early[ia])))
            for gen in GENERATORS:
                for draw in range(args.null_draws):
                    rng = np.random.default_rng(stable_seed(args.seed,f"{session}/{split}/{event.window_uid}/{gen}/{draw}"))
                    xy, speed = reflected_path(grid,durations,rng)
                    projection, states = tree.query(xy)
                    truth = truth_frames(grid[states],durations)
                    generate_rates = early if gen == "matched_map" else second
                    ga = rng.lognormal(-.045,.3,len(ia)) if gen == "drift_gain" else None
                    gb = rng.lognormal(-.045,.3,len(ib)) if gen == "drift_gain" else None
                    sa = simulate_counts(base[:,ia].sum(axis=1), generate_rates[ia],states,rng,ga)
                    sb = simulate_counts(base[:,ib].sum(axis=1), generate_rates[ib],states,rng,gb)
                    ac,bc = event_windows(sa,durations),event_windows(sb,durations)
                    ad,bd = decode(ac,early[ia],grid),decode(bc,early[ib],grid)
                    for cohort,endpoint in endpoints:
                        rows.append(dict(**identity,cohort=cohort,source="sim_calibration" if draw == 0 else "sim_test",
                            generator=gen,draw=draw,endpoint_frame=int(endpoint),projection_error_cm=float(projection.max()),
                            speed_cm_s=speed,**endpoint_readout(ad,bd,ac,bc,grid,endpoint,ea,local_tree,early[ia],truth,truth)))
                    if draw == 1:
                        alternate = grid.max(axis=0)+grid.min(axis=0)-xy
                        _, alt_states = tree.query(alternate)
                        alt_truth = truth_frames(grid[alt_states],durations)
                        sc = simulate_counts(base[:,ib].sum(axis=1),generate_rates[ib],alt_states,rng,gb)
                        cc = event_windows(sc,durations)
                        cd = decode(cc,early[ib],grid)
                        for cohort,endpoint in endpoints:
                            rows.append(dict(**identity,cohort=cohort,source="sim_conflict",generator=gen,
                                draw=draw,endpoint_frame=int(endpoint),projection_error_cm=float(projection.max()),speed_cm_s=speed,
                                **endpoint_readout(ad,cd,ac,cc,grid,endpoint,ea,local_tree,early[ia],truth,alt_truth)))
                    if len(audit) < 6:
                        audit.append(dict(**identity,generator=gen,draw=draw,
                            base_totals_match=bool(np.array_equal(sa.sum(axis=1),base[:,ia].sum(axis=1)) and np.array_equal(sb.sum(axis=1),base[:,ib].sum(axis=1))),
                            frame_totals_match=bool(np.array_equal(ac.sum(axis=1),ca.sum(axis=1)) and np.array_equal(bc.sum(axis=1),cb.sum(axis=1))),
                            endpoint=int(event.fixed_endpoint_frame)))
        print(session,"split",split,"rows",len(rows),flush=True)
    frame=pd.DataFrame(rows)
    frame.to_csv(folder/"event_readouts.csv.gz",index=False)
    (folder/"simulation_invariant_examples.json").write_text(json.dumps(audit,indent=2))
    return frame


def support_stratum(frame):
    spike=np.minimum(frame.a_spikes,frame.b_spikes).to_numpy()
    active=np.minimum(frame.a_active,frame.b_active).to_numpy()
    return pd.Series(np.digitize(spike,[3,6])*3+np.digitize(active,[2,4]),index=frame.index)


def summarize_disagreement(events, status, output):
    primary=events[events.split.eq(0) & events.cohort.eq(COHORT)].copy()
    primary["stratum"]=support_stratum(primary)
    primary["calibration_stratum"]=np.digitize(np.minimum(primary.a_spikes,primary.b_spikes),[3,6])
    results,calibration,paired=[],[],[]
    for session,g in primary.groupby("session"):
        animal=g.animal.iloc[0]
        real=g[g.source.eq("real")]
        for gen in GENERATORS:
            sims=g[g.generator.eq(gen)]
            for subset in ("all", "spike_active_supported"):
                def eligible(f):
                    return f if subset == "all" else f[(f.a_spikes>=3)&(f.b_spikes>=3)&(f.a_active>=2)&(f.b_active>=2)]
                r=eligible(real)
                cal=eligible(sims[sims.source.eq("sim_calibration")])
                test=eligible(sims[sims.source.eq("sim_test")])
                conflict=eligible(sims[sims.source.eq("sim_conflict") & sims.truth_separation_cm.ge(40)])
                threshold={}
                for stratum,c in cal.groupby("calibration_stratum"):
                    if len(c)>=20:
                        threshold[stratum]=float(np.quantile(c.endpoint_separation_cm,.95,method="higher"))
                for f,tag in [(r,"real"),(test,"null_test"),(conflict,"injected_conflict")]:
                    valid=f[f.calibration_stratum.isin(threshold)].copy()
                    valid["tail"]=valid.endpoint_separation_cm > valid.calibration_stratum.map(threshold)
                    calibration.append(dict(animal=animal,session=session,generator=gen,subset=subset,source=tag,
                        calibration_stratification="minimum_spike_count_only",
                        events=len(f),calibrated_events=len(valid),coverage=len(valid)/len(f) if len(f) else np.nan,
                        tail_fraction=valid["tail"].mean() if len(valid) else np.nan))
                overlap=set(r.stratum)&set(test.stratum)
                coverage=r.stratum.isin(overlap).mean() if len(r) else np.nan
                weights=r[r.stratum.isin(overlap)].stratum.value_counts(normalize=True)
                real_mean=sim_mean=0.
                for stratum,weight in weights.items():
                    real_mean+=weight*r[r.stratum.eq(stratum)].endpoint_separation_cm.mean()
                    sim_mean+=weight*test[test.stratum.eq(stratum)].endpoint_separation_cm.mean()
                active_real=np.minimum(r.a_active,r.b_active)
                active_sim=np.minimum(test.a_active,test.b_active)
                ks=ks_2samp(active_real,active_sim).statistic if len(r) and len(test) else np.nan
                error=np.median(np.r_[test.a_truth_error_cm,test.b_truth_error_cm]) if len(test) else np.nan
                ka=ks_2samp(r.a_active,test.a_active).statistic if len(r) and len(test) else np.nan
                kb=ks_2samp(r.b_active,test.b_active).statistic if len(r) and len(test) else np.nan
                error_a=test.a_truth_error_cm.median()
                error_b=test.b_truth_error_cm.median()
                results.append(dict(animal=animal,session=session,generator=gen,subset=subset,real_events=len(r),
                    null_events=len(test),known_path_median_error_cm=error,active_support_ks=ks,
                    a_known_path_median_error_cm=error_a,b_known_path_median_error_cm=error_b,
                    a_active_support_ks=ka,b_active_support_ks=kb,
                    real_endpoint_separation_median_cm=r.endpoint_separation_cm.median(),
                    null_endpoint_separation_median_cm=test.endpoint_separation_cm.median(),
                    real_persistent_conflict_fraction=(r.persistent_conflict_ms>=20).mean() if len(r) else np.nan,
                    null_persistent_conflict_fraction=(test.persistent_conflict_ms>=20).mean() if len(test) else np.nan,
                    matched_support_coverage=coverage,matched_real_mean_separation_cm=real_mean if len(weights) else np.nan,
                    matched_null_mean_separation_cm=sim_mean if len(weights) else np.nan,
                    matched_excess_cm=real_mean-sim_mean if len(weights) else np.nan,
                    calibration_resolution_pass=bool(np.isfinite([error_a,error_b,ka,kb]).all()
                        and max(error_a,error_b)<=20 and max(ka,kb)<=.1)))
            same=real[["window_uid","endpoint_separation_cm"]].merge(
                sims[sims.source.eq("sim_test")][["window_uid","endpoint_separation_cm"]],on="window_uid",suffixes=("_real","_null"),validate="one_to_many")
            same["session"],same["animal"],same["generator"]=session,animal,gen
            paired.append(same)
    result,cal=pd.DataFrame(results),pd.DataFrame(calibration)
    result.to_csv(output/"disagreement_by_session.csv",index=False)
    cal.to_csv(output/"null_calibration_by_session.csv",index=False)
    if paired:
        pd.concat(paired,ignore_index=True).to_csv(output/"paired_null_disagreement.csv.gz",index=False)
    metrics=result.select_dtypes("number").columns.tolist()
    result.groupby(["animal","generator","subset"])[metrics].mean().reset_index().to_csv(output/"disagreement_by_animal.csv",index=False)
    gate=[]
    for gen in GENERATORS:
        r=result[result.generator.eq(gen) & result.subset.eq("spike_active_supported")]
        c=cal[cal.generator.eq(gen)&cal.subset.eq("spike_active_supported")]
        null=c[c.source.eq("null_test")]
        injected=c[c.source.eq("injected_conflict")]
        represented=c.groupby("session").source.nunique()
        coverage_pass=bool(len(c) and len(represented)==len(r) and (represented==3).all()
            and c.coverage.notna().all() and (c.coverage>=.8).all() and (c.calibrated_events>=40).all())
        checks=dict(nonempty=bool(len(r) and r.real_events.sum()>0),
            calibration_coverage_and_denominators=coverage_pass,
            known_path_recovery=bool(len(r) and r.calibration_resolution_pass.all()),
            null_fpr=bool(len(null) and null.tail_fraction.notna().all() and (null.tail_fraction<=.075).all()),
            conflict_sensitivity=bool(len(injected) and injected.tail_fraction.notna().all() and (injected.tail_fraction>=.8).all()),
            local_run_matched=bool(len(status[status.split.eq(0)&status.confirmed]) and status[status.split.eq(0)&status.confirmed].local_match_all.all()))
        checks["biological_conflict_interpretable"]=all(checks.values())
        gate.extend(dict(generator=gen,gate=k,passed=v) for k,v in checks.items())
    gate.append(dict(generator="overall",gate="biological_conflict_interpretable",
        passed=all(row["passed"] for row in gate if row["gate"]=="biological_conflict_interpretable")
        and sum(row["gate"]=="biological_conflict_interpretable" for row in gate)==len(GENERATORS)))
    pd.DataFrame(gate).to_csv(output/"disagreement_gate_summary.csv",index=False)
    lines=["# Independent Population Content Pilot","","Disagreement is not by itself evidence for multiplexed replay.",
        "",f"Primary real events: {len(primary[primary.source.eq('real')])}; sessions: {primary.session.nunique()}; rats: {primary.animal.nunique()}.",
        "", "## Calibration", ""]
    for row in result[result.subset.eq("spike_active_supported")].itertuples():
        lines.append(f"- {row.session}, {row.generator}: A/B known-path median errors {row.a_known_path_median_error_cm:.1f}/{row.b_known_path_median_error_cm:.1f} cm; A/B active-support KS {row.a_active_support_ks:.3f}/{row.b_active_support_ks:.3f}; support-stratified mean disagreement excess {row.matched_excess_cm:+.1f} cm.")
    lines.extend(["", "## Interpretation", "",
        "Biological conflicts require local RUN matching, simulator recovery, active-support calibration, controlled false positives AND power to detect injected conflicts. Failed gates prohibit that claim.",
        "Conditional simulations preserve observed 5-ms total counts exactly but not cell correlations or exact active-unit counts. Drift/gain sensitivity is not an exhaustive biological null.",
        "Primary null thresholds use minimum-spike strata only. Joint minimum-spike/active strata are a separate reweighting sensitivity, not exact joint population support matching.",
        "Post-review technical hardening requires >=80% calibrated coverage and >=40 calibrated events per real/null/conflict session case, per-population error/active-count gates, and BOTH generators to pass. These stricter guards cannot rescue a failed scientific gate.",
        "Population selection uses early RUN and a later matching block. Confirmation failures are not replaced. Original full-RUN cell eligibility and pooled-event ascertainment remain conditioning boundaries.",
        "Fine/coarse support predictions are in the separate reliability report; held-out population agreement must not be equated with ground-truth correctness."])
    (output/"disagreement_report.md").write_text("\n".join(lines)+"\n")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-manifest",type=Path,required=True)
    parser.add_argument("--matched-dir",type=Path,required=True)
    parser.add_argument("--previous-result-dir",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--seed",type=int,default=20260914)
    parser.add_argument("--splits",type=int,default=3)
    parser.add_argument("--null-draws",type=int,default=2)
    args=parser.parse_args()
    if args.null_draws < 2 or args.splits < 1:
        parser.error("need >=2 null draws and >=1 partition")
    args.output_dir.mkdir(parents=True,exist_ok=False)
    protocol=ROOT/"docs/pf_independent_population_content_protocol.md"
    inputs=dict(benchmark=args.benchmark_manifest,matched_manifest=args.matched_dir/"manifest.json",
        endpoints=args.previous_result_dir/"event_content.csv.gz",protocol=protocol,producer=Path(__file__))
    manifest=build_script_provenance(input_paths=inputs)
    manifest.update(status="run_selection",parameters={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
    path=args.output_dir/"manifest.json"
    path.write_text(json.dumps(manifest,indent=2))
    (args.output_dir/"protocol.md").write_text(protocol.read_text())
    started=time.monotonic()
    try:
        records=[r for r in json.loads(args.benchmark_manifest.read_text())["results"] if r["dataset"]=="pfeiffer_foster"]
        states,run_rows,local_rows=[],[],[]
        for rec in records:
            state,rows,local=freeze_session(rec,args)
            states.append(state);run_rows.extend(rows);local_rows.extend(local)
        status=pd.DataFrame(run_rows)
        status.to_csv(args.output_dir/"population_confirmation.csv",index=False)
        pd.DataFrame(local_rows).to_csv(args.output_dir/"regional_run_confirmation.csv",index=False)
        frozen=args.output_dir/"frozen_populations.json"
        frozen.write_text(json.dumps(states,indent=2))
        manifest.update(status="scoring_frozen_populations",frozen_sha256=file_sha256(frozen))
        path.write_text(json.dumps(manifest,indent=2))
        previous=pd.read_csv(args.previous_result_dir/"event_content.csv.gz")
        frames=[score_session(rec,state,args,previous) for rec,state in zip(records,states,strict=True)]
        populated=[f for f in frames if not f.empty]
        if not populated:
            raise ValueError("no confirmed disjoint population pair; no vacuous evaluation")
        events=pd.concat(populated,ignore_index=True)
        events.to_csv(args.output_dir/"event_readouts.csv.gz",index=False)
        summarize_disagreement(events,status,args.output_dir)
        manifest.update(status="complete",runtime_s=time.monotonic()-started,rows=len(events),
            completed_at_utc=datetime.now(UTC).isoformat(),event_readouts_sha256=file_sha256(args.output_dir/"event_readouts.csv.gz"))
    except Exception as exc:
        manifest.update(status="failed",error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        path.write_text(json.dumps(manifest,indent=2))


if __name__ == "__main__":
    main()
