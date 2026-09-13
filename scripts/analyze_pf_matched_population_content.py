#!/usr/bin/env python3
"""Match PF neuronal populations using RUN only, then audit fixed replay content."""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.special import softmax

from hipporeplayimm.data import load_replay_session
from hipporeplayimm.encoding import EncodingConfig, _speed_cm_s, _times_in_intervals, fit_place_field_encoding
from scripts._provenance import build_script_provenance, file_sha256

GATES = dict(rate_relative=.10, stability_absolute=.05, area_relative=.20,
             median_error_difference_cm=2., p75_error_difference_cm=5.,
             max_median_error_cm=20., max_p75_error_cm=40., min_coverage_ratio=1.5)


def relative_difference(a, b):
    return abs(a-b)/max((abs(a)+abs(b))/2, 1e-12)


def quality_match(a, b, prefix="match"):
    return bool(all(np.isfinite([a[f"{prefix}_median_cm"], b[f"{prefix}_median_cm"],
                                 a[f"{prefix}_p75_cm"], b[f"{prefix}_p75_cm"]]))
        and abs(a[f"{prefix}_median_cm"]-b[f"{prefix}_median_cm"]) <= GATES["median_error_difference_cm"]
        and abs(a[f"{prefix}_p75_cm"]-b[f"{prefix}_p75_cm"]) <= GATES["p75_error_difference_cm"]
        and max(a[f"{prefix}_median_cm"], b[f"{prefix}_median_cm"]) <= GATES["max_median_error_cm"]
        and max(a[f"{prefix}_p75_cm"], b[f"{prefix}_p75_cm"]) <= GATES["max_p75_error_cm"])


def descriptors_match(a, b):
    return bool(a["n_cells"] == b["n_cells"] and a["n_cells"] >= 20
        and relative_difference(a["mean_rate_hz"], b["mean_rate_hz"]) <= GATES["rate_relative"]
        and abs(a["median_stability"]-b["median_stability"]) <= GATES["stability_absolute"]
        and relative_difference(a["median_area_cm2"], b["median_area_cm2"]) <= GATES["area_relative"])


def choose_pair(candidates, family):
    rows = candidates.to_dict("records")
    best, best_ratio, matched = None, -np.inf, 0
    for i, a in enumerate(rows):
        for b in rows[i+1:]:
            allowed = (a["kind"] == b["kind"] == family) if family != "targeted" else {a["kind"], b["kind"]} == {"home_high", "home_low"}
            if not allowed or not descriptors_match(a, b) or not quality_match(a, b):
                continue
            high, low = (a, b) if a["coverage_score"] >= b["coverage_score"] else (b, a)
            ratio = high["coverage_score"]/max(low["coverage_score"], 1e-12)
            matched += 1
            if family == "random":
                return (high, low), matched
            if ratio >= GATES["min_coverage_ratio"] and ratio > best_ratio:
                best, best_ratio = (high, low), ratio
    return best, matched


def population_candidates(eligible, units, seed, per_kind=64, tetrode_attempts=256):
    rng = np.random.default_rng(seed)
    n = len(eligible)//2
    score = units.loc[eligible, "coverage_score"].to_numpy()
    z = np.clip((score-score.mean())/max(score.std(), 1e-9), -3, 3)
    populations, seen = [], set()
    for kind, sign in [("home_high", 1), ("home_low", -1), ("random", 0)]:
        weight = np.exp(sign*1.5*z)
        for _ in range(per_kind):
            ix = tuple(sorted(rng.choice(eligible, n, replace=False, p=weight/weight.sum()).tolist()))
            if (kind, ix) not in seen:
                populations.append(dict(kind=kind, indices=list(ix)))
                seen.add((kind, ix))
    electrodes = units.loc[eligible, "tetrode"].to_numpy(int)
    unique = np.unique(electrodes)
    for _ in range(tetrode_attempts):
        keep = rng.choice(unique, len(unique)//2, replace=False)
        ix = tuple(eligible[np.isin(electrodes, keep)].tolist())
        if .35*len(eligible) <= len(ix) <= .65*len(eligible) and ("whole_tetrode", ix) not in seen:
            populations.append(dict(kind="whole_tetrode", indices=list(ix)))
            seen.add(("whole_tetrode", ix))
    for i, pop in enumerate(populations):
        pop["candidate_id"] = i
    return populations


def run_windows(position, intervals, spikes, cell_ids, start, end, cap=512):
    t, xy = position[:, 0], position[:, 1:3]
    speed = _speed_cm_s(t, xy)
    starts = np.arange(start, end-.25, .25)
    rows = []
    for a in starts:
        b = a+.25
        left, right = np.searchsorted(t, [a, b])
        left, right = max(left-1, 0), min(right+1, len(t))
        times = t[left:right]
        if len(times) < 3 or times[0] > a or times[-1] < b or np.max(np.diff(times)) > .1:
            continue
        if not np.isfinite(xy[left:right]).all() or not ((speed[left:right] >= 10) & (speed[left:right] <= 200)).all():
            continue
        if not any(a >= lo and b <= hi for lo, hi in intervals):
            continue
        center = np.array([np.interp(a+.125, t, xy[:, d]) for d in range(2)])
        rows.append([a, b, *center])
    if len(rows) < 100:
        raise ValueError(f"fewer than 100 eligible RUN windows in {start}..{end}: {len(rows)}")
    windows = np.array(rows)[np.linspace(0, len(rows)-1, min(cap, len(rows)), dtype=int)]
    counts = np.empty((len(windows), len(cell_ids)), int)
    for i, cell in enumerate(cell_ids):
        ts = np.sort(spikes[spikes[:, 1] == cell, 0])
        counts[:, i] = np.searchsorted(ts, windows[:, 1])-np.searchsorted(ts, windows[:, 0])
    return windows, counts


def run_error(counts, rates, grid, truth):
    ll = csr_matrix(counts, dtype=float)@np.log(rates)-.25*rates.sum(axis=0)
    mean = softmax(ll, axis=1)@grid
    return np.linalg.norm(mean-truth, axis=1)


def posterior(counts, rates):
    return softmax(csr_matrix(counts, dtype=float)@np.log(rates)-.02*rates.sum(axis=0), axis=1)


def hashed(path, expected, inputs):
    value = file_sha256(path)
    if value is None or (expected is not None and value != expected):
        raise ValueError(f"missing or changed input: {path}")
    inputs[str(path)] = value
    return Path(path)


def prepare_session(rec, args, number, home_row):
    session, animal = rec["session"], rec["animal"]
    folder = args.output_dir/session.replace("/", "_")
    folder.mkdir()
    inputs = {}
    source = rec["source"]
    with np.load(hashed(source["source_cache_path"], source["source_cache_sha256"], inputs)) as a:
        keys = ["position", "spikes", "supported_run_intervals", "cell_ids", "unit_qc_mask", "rates_first_half_hz", "occupancy_first_half_s", "bin_centers_cm"]
        cache = {k: a[k] for k in keys}
    with np.load(hashed(rec["input_arrays_path"], rec["input_arrays_sha256"], inputs)) as a:
        support, ids = a["support"], a["cell_ids"]
    mask = cache["unit_qc_mask"].astype(bool)
    if not np.array_equal(cache["cell_ids"][mask], ids):
        raise ValueError("cell identity mismatch")
    raw_path = args.dataset_root/session
    hashed(raw_path/"Spike_Data.mat", None, inputs)
    raw = load_replay_session(raw_path)
    raw = replace(raw, position=cache["position"], spikes=cache["spikes"], run_times=cache["supported_run_intervals"])
    electrodes = dict((int(cell), int(tt)) for tt, cell in raw.tetrode_cell_ids)
    if len(electrodes) != len(raw.tetrode_cell_ids) or not set(ids).issubset(electrodes):
        raise ValueError("missing or duplicate tetrode metadata")
    lo, hi = raw.run_times[:, 0].min(), raw.run_times[:, 1].max()
    q = np.linspace(lo, hi, 5)
    cfg = EncodingConfig(bin_size_cm=8., smoothing_sigma_bins=1.5, min_speed_cm_s=10.,
        min_occupancy_s=.05, rate_floor_hz=1e-4, arena_padding_cm=0., use_excitatory=False, exclude_ripple_intervals=False)
    fits = []
    for a, b in [(q[0], q[1]), (q[1], q[2])]:
        intervals = np.array([(max(a, s), min(b, e)) for s, e in raw.run_times if min(b, e)>max(a, s)])
        fit = fit_place_field_encoding(replace(raw, run_times=intervals), cfg)
        if not np.array_equal(fit.cell_ids, cache["cell_ids"]) or not np.allclose(fit.bin_centers, cache["bin_centers_cm"]):
            raise ValueError("quarter fit does not match source grid/units")
        fits.append(fit)
    common = (fits[0].occupancy_s >= .05) & (fits[1].occupancy_s >= .05)
    stability = []
    for a, b in zip(fits[0].rates_hz[mask][:, common], fits[1].rates_hz[mask][:, common], strict=True):
        stability.append(float(np.corrcoef(a, b)[0, 1]) if len(a)>2 and min(a.std(), b.std())>0 else np.nan)
    grid = cache["bin_centers_cm"][support]
    early = cache["rates_first_half_hz"][mask][:, support]
    home = np.array([home_row.home_x_cm, home_row.home_y_cm], float)
    near = np.linalg.norm(grid-home, axis=1) <= 20
    if not near.any():
        raise ValueError("Home has no supported states")
    spike_times = raw.spikes[:, 0]
    spike_speed = np.interp(spike_times, raw.position[:, 0], _speed_cm_s(raw.position[:, 0], raw.position[:, 1:3]))
    early_run = (spike_times >= q[0]) & (spike_times < q[2]) & (spike_speed >= 10) & _times_in_intervals(spike_times, raw.run_times)
    observed_ids, observed_counts = np.unique(raw.spikes[early_run, 1].astype(int), return_counts=True)
    count_by_id = dict(zip(observed_ids, observed_counts, strict=True))
    rate = np.array([count_by_id.get(int(cell), 0) for cell in ids])/cache["occupancy_first_half_s"].sum()
    units = pd.DataFrame(dict(cell_id=ids, tetrode=[electrodes[int(i)] for i in ids],
        mean_rate_hz=rate, stability=stability,
        area_cm2=64*(early >= .5*early.max(axis=1)[:, None]).sum(axis=1),
        coverage_score=early[:, near].mean(axis=1)/early.mean(axis=1)))
    eligible = np.flatnonzero(np.isfinite(units.stability) & (units.mean_rate_hz > 0))
    if len(eligible)//2 < 20:
        raise ValueError("too few early-eligible units")
    units["eligible"] = np.isin(np.arange(len(units)), eligible)
    units.to_csv(folder/"early_unit_descriptors.csv", index=False)
    match_w, match_c = run_windows(raw.position, raw.run_times, raw.spikes, ids, q[2], q[3])
    confirm_w, confirm_c = run_windows(raw.position, raw.run_times, raw.spikes, ids, q[3], q[4])
    pops = population_candidates(eligible, units, args.seed+number)
    candidate_rows = []
    for p in pops:
        ix = p["indices"]
        u = units.loc[ix]
        error = run_error(match_c[:, ix], early[ix], grid, match_w[:, 2:])
        candidate_rows.append(dict(candidate_id=p["candidate_id"], kind=p["kind"], n_cells=len(ix),
            mean_rate_hz=u.mean_rate_hz.mean(), median_stability=u.stability.median(),
            median_area_cm2=u.area_cm2.median(), coverage_score=u.coverage_score.mean(),
            match_median_cm=np.median(error), match_p75_cm=np.quantile(error,.75)))
    candidates = pd.DataFrame(candidate_rows)
    candidates.to_csv(folder/"matching_candidates.csv", index=False)
    chosen, statuses = [], []
    for family in ["targeted", "random", "whole_tetrode"]:
        pair, n_match = choose_pair(candidates, family)
        if pair is None:
            statuses.append(dict(animal=animal, session=session, family=family,
                status="no_matching_pair", matching_pairs=n_match, confirmation_pass=False))
            continue
        high, low = pair
        selected = dict(animal=animal, session=session, family=family,
            high={**high, **pops[high["candidate_id"]]}, low={**low, **pops[low["candidate_id"]]}, matching_pairs=n_match)
        chosen.append(selected)
    # This immutable file precedes all confirmation decoding and all replay access.
    frozen = folder/"selected_before_confirmation.json"
    frozen.write_text(json.dumps(dict(created_at_utc=datetime.now(UTC).isoformat(), selected=chosen), indent=2))
    frozen_hash = file_sha256(frozen)
    for pair in chosen:
        for side in ["high", "low"]:
            pop = pair[side]
            ix = pop["indices"]
            error = run_error(confirm_c[:, ix], early[ix], grid, confirm_w[:, 2:])
            home_windows = np.linalg.norm(confirm_w[:, 2:]-home, axis=1) <= 30
            pop.update(confirm_median_cm=float(np.median(error)), confirm_p75_cm=float(np.quantile(error,.75)),
                confirm_home_windows=int(home_windows.sum()),
                confirm_home_median_cm=float(np.median(error[home_windows])) if home_windows.any() else None,
                confirm_nonhome_median_cm=float(np.median(error[~home_windows])) if (~home_windows).any() else None)
        passed = quality_match(pair["high"], pair["low"], "confirm")
        pair["confirmation_pass"] = passed
        a, b = pair["high"], pair["low"]
        statuses.append(dict(animal=animal, session=session, family=pair["family"],
            status="confirmed_match" if passed else "confirmation_failed_no_reselection",
            matching_pairs=pair["matching_pairs"], confirmation_pass=passed,
            high_candidate_id=a["candidate_id"], low_candidate_id=b["candidate_id"], n_cells=a["n_cells"],
            coverage_ratio=a["coverage_score"]/b["coverage_score"],
            shared_cells=len(set(a["indices"]) & set(b["indices"])),
            **{f"{side}_{key}": p[key] for side,p in [("high",a),("low",b)] for key in ["mean_rate_hz", "median_stability", "median_area_cm2", "match_median_cm", "match_p75_cm", "confirm_median_cm", "confirm_p75_cm", "confirm_home_windows", "confirm_home_median_cm", "confirm_nonhome_median_cm"]}))
    assert file_sha256(frozen) == frozen_hash
    np.savez_compressed(folder/"run_validation.npz", match_windows=match_w, match_counts=match_c,
        confirm_windows=confirm_w, confirm_counts=confirm_c, early_rates=early, grid_cm=grid, cell_ids=ids)
    checkpoint = dict(animal=animal, session=session, inputs=inputs, selected=chosen,
        selection_sha256=frozen_hash, run_validation_sha256=file_sha256(folder/"run_validation.npz"),
        matching_windows=len(match_w), confirmation_windows=len(confirm_w), early_eligible_cells=len(eligible),
        no_replay_used=True, status="frozen")
    (folder/"freeze_checkpoint.json").write_text(json.dumps(checkpoint, indent=2))
    print(session, [(r["family"], r["status"]) for r in statuses], flush=True)
    return statuses, checkpoint


def score_session(rec, checkpoint, args, previous, home_row):
    session = rec["session"]
    folder = args.output_dir/session.replace("/", "_")
    if file_sha256(folder/"selected_before_confirmation.json") != checkpoint["selection_sha256"]:
        raise ValueError("selected populations changed after freezing")
    if file_sha256(folder/"run_validation.npz") != checkpoint["run_validation_sha256"]:
        raise ValueError("RUN cache changed")
    if file_sha256(rec["input_arrays_path"]) != rec["input_arrays_sha256"]:
        raise ValueError("replay input changed")
    with np.load(rec["input_arrays_path"]) as z:
        arrays = {k:z[k] for k in z.files}
    with np.load(folder/"run_validation.npz") as z:
        early = z["early_rates"]
    events = previous[previous.session.eq(session) & previous.population_replicate.eq(0)].copy()
    lookup = {k:i for i,k in enumerate(arrays["window_uids"])}
    full_offsets = arrays["frame_offsets"]
    rows, grid = [], arrays["grid_cm"]
    home = np.array([home_row.home_x_cm, home_row.home_y_cm], float)
    near = np.linalg.norm(grid-home, axis=1) <= 20
    for pair in checkpoint["selected"]:
        if not pair["confirmation_pass"]:
            continue
        ix = {side:pair[side]["indices"] for side in ["high","low"]}
        for encoding, rates in [("full_run", arrays["rates_hz"][:, arrays["support"]]), ("early_run", early)]:
            for row in events.itertuples():
                i = lookup[row.window_uid]
                a,b = full_offsets[2*i:2*i+2]
                for cohort, endpoint in [("all_fixed_candidates",row.fixed_endpoint_frame),
                                        ("full_accepted_segment", row.full_segment_endpoint_frame)]:
                    if cohort == "full_accepted_segment" and not row.accepted_full:
                        continue
                    c = arrays["frame_counts"][a+int(endpoint):a+int(endpoint)+1]
                    p = {side:posterior(c[:,idx], rates[idx])[0] for side,idx in ix.items()}
                    high, low = p["high"], p["low"]
                    rows.append(dict(animal=rec["animal"],session=session,window_uid=row.window_uid,
                        family=pair["family"],encoding=encoding,cohort=cohort,n_cells=len(ix["high"]),
                        fixed_endpoint_frame=int(endpoint),home_mass_high=high[near].sum(),
                        home_mass_low=low[near].sum(),home_mass_delta=high[near].sum()-low[near].sum(),
                        home_map_high=bool(near[high.argmax()]),home_map_low=bool(near[low.argmax()]),
                        endpoint_mean_separation_cm=np.linalg.norm(high@grid-low@grid),
                        **{f"{side}_endpoint_spikes":int(c[:,idx].sum()) for side,idx in ix.items()},
                        **{f"{side}_endpoint_active_cells":int((c[:,idx]>0).sum()) for side,idx in ix.items()}))
    frame = pd.DataFrame(rows)
    frame.to_csv(folder/"replay_content.csv.gz", index=False)
    print(session, "scored", len(frame), "endpoint contrasts", flush=True)
    return frame


def summarize(events, output, seed):
    if events.empty:
        return pd.DataFrame()
    keys=["family","encoding","cohort"]
    metrics=["home_mass_high","home_mass_low","home_mass_delta","home_map_high","home_map_low",
             "endpoint_mean_separation_cm","high_endpoint_spikes","low_endpoint_spikes"]
    session=events.groupby(["animal","session"]+keys)[metrics].mean().reset_index()
    counts=events.groupby(["animal","session"]+keys).size().rename("events").reset_index()
    session=session.merge(counts,on=["animal","session"]+keys,validate="one_to_one")
    animal=session.groupby(["animal"]+keys)[metrics].mean().reset_index()
    rng,rows=np.random.default_rng(seed),[]
    for names,g in animal.groupby(keys):
        for metric in metrics:
            v=g[metric].to_numpy(float)
            means=v[rng.integers(0,len(v),(5000,len(v)))].mean(axis=1)
            low,high=np.quantile(means,[.025,.975])
            rows.append(dict(zip(keys,names),metric=metric,equal_rat_mean=v.mean(),
                ci_low=low,ci_high=high,rats=len(v),rats_positive=int((v>0).sum())))
    summary=pd.DataFrame(rows)
    session.to_csv(output/"session_content_summary.csv",index=False)
    animal.to_csv(output/"animal_content_summary.csv",index=False)
    summary.to_csv(output/"content_summary.csv",index=False)
    return summary


def write_report(status, summary, output):
    lines=["# PF Matched Recording Populations", "", "RUN-only population selection; no reselection after RUN confirmation.", "",
           "Home metadata are inferred. Targeted samples demonstrate possibility, not routine recording bias.", "", "## Matching Feasibility", ""]
    for family,g in status.groupby("family"):
        passed=g[g.confirmation_pass]
        lines.append(f"- {family}: {len(passed)}/{len(g)} sessions confirmed, {passed.animal.nunique()} rats.")
    lines += ["", "No confirmation failure contributes replay content to the primary comparison.", "", "## Confirmed Matches: High Minus Low Home Tuning", ""]
    if summary.empty:
        lines.append("No confirmed population pairs. This experiment cannot adjudicate content bias under matched RUN quality.")
    else:
        for r in summary[summary.metric.eq("home_mass_delta") & summary.encoding.eq("full_run")].itertuples():
            lines.append(f"- {r.family}, {r.cohort}: Home mass change {100*r.equal_rat_mean:+.2f} pp; descriptive 95% rat-bootstrap CI [{100*r.ci_low:+.2f}, {100*r.ci_high:+.2f}], {r.rats} rats.")
    lines += ["", "## Boundaries", "",
        "- Same cell count does not mean the same neural observations; actual replay spike support is reported, not matched.",
        "- Global RUN matching does not match local Home decoding quality or noise correlations.",
        "- Frozen original unit eligibility used full RUN; new maps, descriptors and subset matching use early/matching RUN only.",
        "- A Home posterior-mass contrast is not evidence that published planning effects are artifacts.",
        "- Targeted pairs are chosen for a large spatial tuning contrast. Whole-tetrode pairs are also contrast-selected, not an estimate of typical tetrode sampling.",
        "- The random comparison uses the first quality-matched pair, oriented by early Home tuning only.",
        "- Four or fewer rats limit generalization. A failed confirmation is a matching failure, not a negative biological result."]
    (output/"report.md").write_text("\n".join(lines)+"\n")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-manifest",type=Path,required=True)
    parser.add_argument("--previous-result-dir",type=Path,required=True)
    parser.add_argument("--dataset-root",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--seed",type=int,default=20260913)
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=False)
    protocol=ROOT/"docs/pf_matched_population_content_protocol.md"
    inputs=dict(benchmark=args.benchmark_manifest,home=args.previous_result_dir/"home_metadata_qc.csv",
        prior_endpoints=args.previous_result_dir/"event_content.csv.gz",protocol=protocol,script=Path(__file__),
        encoding=ROOT/"src/hipporeplayimm/encoding.py",data_loader=ROOT/"src/hipporeplayimm/data.py")
    manifest=build_script_provenance(input_paths=inputs)
    manifest.update(status="preparing_run_only",created_at_utc=datetime.now(UTC).isoformat(),
        thresholds=GATES,parameters={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
    path=args.output_dir/"manifest.json"
    path.write_text(json.dumps(manifest,indent=2))
    (args.output_dir/"protocol.md").write_text(protocol.read_text())
    started=time.monotonic()
    try:
        benchmark=json.loads(args.benchmark_manifest.read_text())
        records=[r for r in benchmark["results"] if r["dataset"]=="pfeiffer_foster"]
        homes=pd.read_csv(args.previous_result_dir/"home_metadata_qc.csv").set_index("session")
        statuses,checkpoints=[],[]
        for number,rec in enumerate(records):
            st,cp=prepare_session(rec,args,number,homes.loc[rec["session"]])
            statuses.extend(st)
            checkpoints.append(cp)
        status=pd.DataFrame(statuses)
        status.to_csv(args.output_dir/"population_match_status.csv",index=False)
        freeze=args.output_dir/"frozen_populations.json"
        freeze.write_text(json.dumps(dict(frozen_at_utc=datetime.now(UTC).isoformat(),sessions=checkpoints),indent=2))
        manifest.update(status="run_selection_frozen",frozen_populations_sha256=file_sha256(freeze))
        path.write_text(json.dumps(manifest,indent=2))
        # No replay decoding or candidate-content outputs were used to choose populations.
        previous=pd.read_csv(args.previous_result_dir/"event_content.csv.gz")
        if file_sha256(args.previous_result_dir/"event_content.csv.gz")!=manifest["input_file_sha256"]["prior_endpoints"]:
            raise ValueError("prior fixed endpoints changed")
        frames=[]
        for rec,cp in zip(records,checkpoints,strict=True):
            frames.append(score_session(rec,cp,args,previous,homes.loc[rec["session"]]))
        populated=[f for f in frames if not f.empty]
        events=pd.concat(populated,ignore_index=True) if populated else pd.DataFrame()
        events.to_csv(args.output_dir/"matched_event_content.csv.gz",index=False)
        summary=summarize(events,args.output_dir,args.seed)
        write_report(status,summary,args.output_dir)
        manifest.update(status="complete",runtime_s=time.monotonic()-started,
                        completed_at_utc=datetime.now(UTC).isoformat())
    except Exception as exc:
        manifest.update(status="failed",error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        path.write_text(json.dumps(manifest,indent=2))


if __name__=="__main__":
    main()
