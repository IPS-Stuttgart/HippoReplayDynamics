#!/usr/bin/env python3
"""Independent RUN/error reconstruction and fixed-endpoint audit; no reselection."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def dense_posterior(counts,rates,duration):
    logp=np.asarray(counts)@np.log(rates)-duration*rates.sum(axis=0)
    weight=np.exp(logp-logp.max(axis=-1,keepdims=True))
    return weight/weight.sum(axis=-1,keepdims=True)


def run_gate(a,b):
    med=np.array([a[0],b[0]])
    p75=np.array([a[1],b[1]])
    return bool(np.isfinite(np.r_[med,p75]).all() and med.max()<=20 and p75.max()<=40
                and np.ptp(med)<=2 and np.ptp(p75)<=5)


def technical_audit(root):
    manifest=json.loads((root/"manifest.json").read_text())
    assert manifest["status"]=="complete"
    frozen_path=root/"frozen_populations.json"
    assert sha(frozen_path)==manifest["frozen_populations_sha256"]
    for name,path in manifest["input_file_paths"].items():
        assert sha(path)==manifest["input_file_sha256"][name],path
    frozen=json.loads(frozen_path.read_text())
    benchmark=json.loads(Path(manifest["input_file_paths"]["benchmark"]).read_text())
    records={r["session"]:r for r in benchmark["results"] if r["dataset"]=="pfeiffer_foster"}
    prior=pd.read_csv(manifest["input_file_paths"]["prior_endpoints"])
    prior=prior[prior.population_replicate.eq(0)].set_index("window_uid",verify_integrity=True)
    home=pd.read_csv(manifest["input_file_paths"]["home"]).set_index("session")
    status=pd.read_csv(root/"population_match_status.csv")
    output_path=root/"matched_event_content.csv.gz"
    try:
        events=pd.read_csv(output_path)
    except pd.errors.EmptyDataError:
        events=pd.DataFrame()
    checks,run_rows,score_rows=[],[],[]
    for cp in frozen["sessions"]:
        session=cp["session"]
        rec=records[session]
        folder=root/session.replace("/","_")
        fpath=folder/"selected_before_confirmation.json"
        assert sha(fpath)==cp["selection_sha256"]
        pre=json.loads(fpath.read_text())
        assert pre["created_at_utc"] < frozen["frozen_at_utc"]
        assert sha(folder/"run_validation.npz")==cp["run_validation_sha256"]
        for path,value in cp["inputs"].items():
            assert sha(path)==value,path
        with np.load(folder/"run_validation.npz") as z:
            run={k:z[k] for k in z.files}
        assert run["match_windows"][:,1].max() < run["confirm_windows"][:,0].min()
        units=pd.read_csv(folder/"early_unit_descriptors.csv")
        with np.load(rec["input_arrays_path"]) as z:
            arrays={k:z[k] for k in z.files}
        grid=arrays["grid_cm"]
        ids=run["cell_ids"]
        assert np.array_equal(ids,arrays["cell_ids"])
        lookup={uid:i for i,uid in enumerate(arrays["window_uids"])}
        near=np.linalg.norm(grid-home.loc[session,["home_x_cm","home_y_cm"]].to_numpy(float),axis=1)<=20
        for pair in cp["selected"]:
            family=pair["family"]
            initial=next(p for p in pre["selected"] if p["family"]==family)
            computed={}
            for side in ["high","low"]:
                pop=pair[side]
                assert "confirm_median_cm" not in initial[side]
                assert pop["indices"]==initial[side]["indices"]
                assert pop["candidate_id"]==initial[side]["candidate_id"]
                ix=pop["indices"]
                u=units.loc[ix]
                assert len(ix)==pop["n_cells"] and len(set(ix))==len(ix)
                assert u.eligible.all()
                for name,value in dict(mean_rate_hz=u.mean_rate_hz.mean(),median_stability=u.stability.median(),
                        median_area_cm2=u.area_cm2.median(),coverage_score=u.coverage_score.mean()).items():
                    assert np.isclose(value,pop[name])
                if family=="whole_tetrode":
                    for tt in u.tetrode.unique():
                        all_ids=set(units.index[units.tetrode.eq(tt)&units.eligible])
                        assert all_ids.issubset(ix)
                for block in ["match","confirm"]:
                    post=dense_posterior(run[f"{block}_counts"][:,ix],run["early_rates"][ix],.25)
                    error=np.linalg.norm(post@grid-run[f"{block}_windows"][:,2:],axis=1)
                    stats=[float(np.median(error)),float(np.quantile(error,.75))]
                    computed[(side,block)]=stats
                    deviation=max(abs(stats[0]-pop[f"{block}_median_cm"]),abs(stats[1]-pop[f"{block}_p75_cm"]))
                    assert deviation<1e-9
                    run_rows.append(dict(session=session,family=family,side=side,block=block,max_error_cm=deviation))
            a,b=pair["high"],pair["low"]
            assert a["n_cells"]==b["n_cells"]>=20
            assert abs(a["mean_rate_hz"]-b["mean_rate_hz"])/((a["mean_rate_hz"]+b["mean_rate_hz"])/2)<=.1+1e-12
            assert abs(a["median_stability"]-b["median_stability"])<=.05+1e-12
            assert abs(a["median_area_cm2"]-b["median_area_cm2"])/((a["median_area_cm2"]+b["median_area_cm2"])/2)<=.2+1e-12
            assert run_gate(computed[("high","match")],computed[("low","match")])
            confirmed=run_gate(computed[("high","confirm")],computed[("low","confirm")])
            assert confirmed==pair["confirmation_pass"]
            sub=events[events.session.eq(session)&events.family.eq(family)] if not events.empty else events
            if not confirmed:
                assert sub.empty
                continue
            local_prior=prior[prior.session.eq(session)]
            for cohort,count in [("all_fixed_candidates",len(local_prior)),("full_accepted_segment",int(local_prior.accepted_full.sum()))]:
                for encoding in ["full_run","early_run"]:
                    selected=sub[sub.cohort.eq(cohort)&sub.encoding.eq(encoding)]
                    assert len(selected)==count
                    assert not selected.window_uid.duplicated().any()
                    rates=arrays["rates_hz"][:,arrays["support"]] if encoding=="full_run" else run["early_rates"]
                    for row in selected.iloc[::max(1,len(selected)//10)].itertuples():
                        expected=prior.loc[row.window_uid]
                        anchor=expected.fixed_endpoint_frame if cohort=="all_fixed_candidates" else expected.full_segment_endpoint_frame
                        assert anchor==row.fixed_endpoint_frame
                        start=arrays["frame_offsets"][2*lookup[row.window_uid]]
                        c=arrays["frame_counts"][start+int(anchor)]
                        p={side:dense_posterior(c[pair[side]["indices"]],rates[pair[side]["indices"]],.02) for side in ["high","low"]}
                        dh,dl=p["high"][near].sum(),p["low"][near].sum()
                        deviation=max(abs(dh-row.home_mass_high),abs(dl-row.home_mass_low),abs(dh-dl-row.home_mass_delta))
                        assert deviation<1e-10
                        score_rows.append(dict(session=session,family=family,window_uid=row.window_uid,
                            cohort=cohort,encoding=encoding,max_home_mass_error=deviation))
        checks.append(dict(gate=session+":frozen_ids_RUN_confirmation_and_replay",passed=True))
    if not events.empty:
        calculated=events.groupby(["animal","session","family","encoding","cohort"]).home_mass_delta.mean().groupby(["animal","family","encoding","cohort"]).mean().groupby(["family","encoding","cohort"]).mean()
        summary=pd.read_csv(root/"content_summary.csv")
        actual=summary[summary.metric.eq("home_mass_delta")].set_index(["family","encoding","cohort"]).equal_rat_mean
        assert np.allclose(calculated.sort_index(),actual.sort_index())
    checks.append(dict(gate="equal_rat_aggregation",passed=True))
    check=pd.DataFrame(checks)
    check.to_csv(root/"independent_gate_summary.csv",index=False)
    pd.DataFrame(run_rows).to_csv(root/"independent_RUN_reconstruction.csv",index=False)
    pd.DataFrame(score_rows).to_csv(root/"independent_replay_reconstruction.csv",index=False)
    result=dict(status="pass",RUN_reconstructions=len(run_rows),replay_reconstructions=len(score_rows),
        auditor_sha256=sha(__file__),producer_manifest_sha256=sha(root/"manifest.json"),
        confirmed_pairs=int(status.confirmation_pass.sum()),n_selected_pairs=len(run_rows)//4,
        all_failures_excluded_without_reselection=True)
    (root/"independent_audit.json").write_text(json.dumps(result,indent=2)+"\n")
    return result


def figure(root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    status=pd.read_csv(root/"population_match_status.csv")
    if not (root/"session_content_summary.csv").exists():
        return
    sessions=pd.read_csv(root/"session_content_summary.csv")
    animal=pd.read_csv(root/"animal_content_summary.csv")
    colors=dict(targeted="#167c9c",random="#757575",whole_tetrode="#ac4b65")
    fig,axes=plt.subplots(1,3,figsize=(12,4.1))
    for family,color in colors.items():
        g=status[status.family.eq(family)&status.confirmation_pass]
        axes[0].scatter(g.low_confirm_median_cm,g.high_confirm_median_cm,color=color,label=family.replace("_"," "))
        h=sessions[sessions.family.eq(family)&sessions.cohort.eq("all_fixed_candidates")&sessions.encoding.eq("full_run")]
        axes[1].scatter(100*h.home_mass_low,100*h.home_mass_high,color=color)
        a=animal[animal.family.eq(family)&animal.cohort.eq("all_fixed_candidates")&animal.encoding.eq("full_run")]
        x=list(colors).index(family)
        axes[2].scatter(x+np.linspace(-.1,.1,len(a)),100*a.home_mass_delta,color=color)
    axes[0].plot([0,20],[0,20],color="black",lw=1)
    axes[0].set(xlim=(0,20),ylim=(0,20),xlabel="Low-Home population: RUN error (cm)",ylabel="High-Home population: RUN error (cm)",title="Untouched RUN confirmation")
    axes[0].legend(fontsize=8,frameon=False)
    limit=max(20.,float(sessions[["home_mass_low","home_mass_high"]].max().max()*110))
    axes[1].plot([0,limit],[0,limit],color="black",lw=1)
    axes[1].set(xlim=(0,limit),ylim=(0,limit),xlabel="Low-Home: replay Home mass (%)",ylabel="High-Home: replay Home mass (%)",title="Same candidates and endpoints")
    axes[2].axhline(0,color="black",lw=1)
    axes[2].set(xlim=(-.4,2.4),xticks=np.arange(3),xticklabels=["Targeted","Random","Whole\ntetrodes"],ylabel="High minus low Home mass (pp)",title="Per-rat effects; confirmed pairs only")
    fig.tight_layout()
    fig.savefig(root/"matched_content_comparison.png",dpi=180)
    plt.close(fig)


def paired_random_reference(root):
    if not (root/"session_content_summary.csv").exists():
        return
    session=pd.read_csv(root/"session_content_summary.csv")
    keys=["animal","session","encoding","cohort"]
    random=session[session.family.eq("random")][keys+["home_mass_delta"]]
    rng=np.random.default_rng(20260913)
    rows=[]
    for family in ["targeted","whole_tetrode"]:
        paired=session[session.family.eq(family)][keys+["home_mass_delta"]].merge(
            random,on=keys,suffixes=("_selected","_random"),validate="one_to_one")
        paired.to_csv(root/f"{family}_same_session_random_comparison.csv",index=False)
        for (encoding,cohort),group in paired.groupby(["encoding","cohort"]):
            means=group.groupby("animal")[["home_mass_delta_selected","home_mass_delta_random"]].mean()
            excess=(means.home_mass_delta_selected-means.home_mass_delta_random).to_numpy()
            low,high=np.nan,np.nan
            if len(excess)>1:
                samples=excess[rng.integers(0,len(excess),(5000,len(excess)))].mean(axis=1)
                low,high=np.quantile(samples,[.025,.975])
            rows.append(dict(family=family,encoding=encoding,cohort=cohort,sessions=len(group),rats=len(means),
                selected_delta=means.home_mass_delta_selected.mean(),random_delta=means.home_mass_delta_random.mean(),
                paired_excess=excess.mean(),ci_low=low,ci_high=high))
    pd.DataFrame(rows).to_csv(root/"same_session_random_reference_summary.csv",index=False)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--result-dir",type=Path,required=True)
    args=p.parse_args()
    print(json.dumps(technical_audit(args.result_dir),indent=2))
    figure(args.result_dir)
    paired_random_reference(args.result_dir)
