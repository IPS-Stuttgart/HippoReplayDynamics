import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("matched", ROOT / "scripts/analyze_pf_matched_population_content.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def row(idx, kind="random", coverage=1., count=40):
    return dict(candidate_id=idx,kind=kind,n_cells=count,mean_rate_hz=1.,median_stability=.7,
        median_area_cm2=200.,coverage_score=coverage,match_median_cm=10.,match_p75_cm=20.,
        confirm_median_cm=10.,confirm_p75_cm=20.)


def test_targeted_choice_uses_only_run_contrast():
    f=pd.DataFrame([row(0,"home_high",3.),row(1,"home_low",1.),row(2,"home_high",4.)])
    selected,n=m.choose_pair(f,"targeted")
    assert (selected[0]["candidate_id"],selected[1]["candidate_id"]) == (2,1)
    assert n==2
    f["replay_score"]=[1e9,0,-1e9]
    assert m.choose_pair(f,"targeted")[0][0]["candidate_id"]==2


def test_no_match_when_counts_differ():
    f=pd.DataFrame([row(0,"home_high",3.,40),row(1,"home_low",1.,41)])
    assert m.choose_pair(f,"targeted")[0] is None


def test_random_selects_first_not_maximum_coverage():
    f=pd.DataFrame([row(0,coverage=1.),row(1,coverage=1.1),row(2,coverage=5.)])
    a,b=m.choose_pair(f,"random")[0]
    assert {a["candidate_id"],b["candidate_id"]}=={0,1}


@pytest.mark.parametrize("field,value", [("mean_rate_hz",2.),("median_stability",.9),("median_area_cm2",400.)])
def test_descriptor_mismatch_fails(field,value):
    a,b=row(0),row(1)
    b[field]=value
    assert not m.descriptors_match(a,b)


def test_confirmation_can_fail_without_reselecting():
    a,b=row(0),row(1)
    b["confirm_median_cm"]=18
    assert m.quality_match(a,b)
    assert not m.quality_match(a,b,"confirm")


def test_finite_good_quality_required():
    a,b=row(0),row(1)
    a["match_median_cm"]=np.nan
    assert not m.quality_match(a,b)
    a["match_median_cm"]=b["match_median_cm"]=30
    assert not m.quality_match(a,b)


def test_candidates_deterministic_and_whole_tetrodes_intact():
    eligible=np.arange(80)
    units=pd.DataFrame(dict(coverage_score=np.linspace(.1,3,80),tetrode=np.repeat(np.arange(20),4)))
    a=m.population_candidates(eligible,units,5,per_kind=4,tetrode_attempts=20)
    b=m.population_candidates(eligible,units,5,per_kind=4,tetrode_attempts=20)
    assert a==b
    for p in a:
        if p["kind"]=="whole_tetrode":
            for tt in units.tetrode.unique():
                group=set(units.index[units.tetrode==tt])
                intersection=group & set(p["indices"])
                assert not intersection or intersection==group
        else:
            assert len(p["indices"])==40


def test_run_truth_clock_and_spike_counts():
    t=np.arange(0,60,.02)
    pos=np.column_stack([t,20*t,np.zeros(len(t))])
    spikes=np.column_stack([np.arange(0,60,.01),np.ones(6000)])
    w,c=m.run_windows(pos,np.array([[0,60]]),spikes,np.array([1]),10,50,cap=100)
    assert len(w)==100
    assert np.allclose(w[:,2],20*(w[:,0]+.125))
    assert np.all(c==25)
    assert np.all(w[:,0]>=10) and np.all(w[:,1]<=50)


def test_replay_posterior_has_no_temporal_prior():
    rng=np.random.default_rng(41)
    c,r=rng.poisson(1,(30,10)),rng.uniform(.1,10,(10,20))
    order=rng.permutation(30)
    assert np.allclose(m.posterior(c,r)[order],m.posterior(c[order],r))


def test_empty_matching_frame():
    f=pd.DataFrame([row(0,"home_high",1.),row(1,"home_low",1.1)])
    assert m.choose_pair(f,"targeted")[0] is None
