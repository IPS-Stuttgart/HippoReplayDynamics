from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from scripts.analyze_pf_independent_population_content import (
    balanced_partitions, decode, endpoint_readout, event_windows, longest_run,
    profile, profile_gap, reflected_path, regions, simulate_counts, support_stratum,
    truth_frames, summarize_disagreement,
)


def fixture():
    grid = np.array([[x,y] for x in range(0,81,8) for y in range(0,81,8)],float)
    centers = grid[np.random.default_rng(2).choice(len(grid),41)]
    rates = .01+10*np.exp(-np.square(centers[:,None,:]-grid).sum(axis=2)/(2*12**2))
    return grid,rates


def test_partition_disjoint_equal_reproducible():
    grid,rates=fixture()
    a=balanced_partitions(np.arange(41),rates,grid,7,12)
    b=balanced_partitions(np.arange(41),rates,grid,7,12)
    assert a == b
    for p in a:
        assert len(p['a']) == len(p['b']) == 20
        assert not set(p['a']) & set(p['b'])
    assert a != balanced_partitions(np.arange(41),rates,grid,8,12)


def test_profile_normalized_and_symmetric_gap():
    grid,rates=fixture()
    p=profile(rates[:20],regions(grid))
    assert np.isclose(p.sum(),1)
    assert profile_gap(p,p)==0
    q=profile(rates[20:],regions(grid))
    assert profile_gap(p,q)==profile_gap(q,p)
    assert np.array_equal(regions(np.array([[1000,-1000.]]),grid),[6])


def test_simulator_preserves_base_and_overlapping_frame_spikes():
    grid,rates=fixture()
    totals=np.array([0,1,5,2,3,0,7,2,0])
    duration=np.r_[np.full(8,.005),.003]
    states=np.arange(len(totals))
    a=simulate_counts(totals,rates,states,np.random.default_rng(4))
    assert np.array_equal(a.sum(axis=1),totals)
    frames=event_windows(a,duration)
    expected=np.convolve(totals[:8],np.ones(4,dtype=int),'valid')
    assert np.array_equal(frames.sum(axis=1),expected)
    assert np.isfinite(decode(frames,rates,grid)['p']).all()


def test_truth_frames_no_integer_truncation():
    xy=np.column_stack([np.arange(6)+.3,np.arange(6)+.8])
    actual=truth_frames(xy,np.r_[np.full(5,.005),.002])
    assert np.allclose(actual,[xy[:4].mean(axis=0),xy[1:5].mean(axis=0)])


def test_single_path_draw_in_bounds():
    grid,_=fixture()
    xy,speed=reflected_path(grid,np.full(200,.005),np.random.default_rng(11))
    assert (xy>=grid.min(axis=0)).all() and (xy<=grid.max(axis=0)).all()
    assert speed in (0,100,300,600,1200)


def test_width_entropy_and_training_only_features():
    grid,rates=fixture()
    counts=np.full((5,20),1)
    a=decode(counts,rates[:20],grid)
    b=decode(counts,rates[20:40],grid)
    tree=cKDTree(grid)
    args=(a,b,counts,counts,grid,4,np.ones(len(grid))*10,tree,rates[:20])
    r=endpoint_readout(*args)
    altered=endpoint_readout(a,decode(counts*7,rates[20:40],grid),counts,counts*7,
        grid,4,np.ones(len(grid))*10,tree,rates[:20])
    for key in r:
        if key.startswith('a_'):
            assert np.allclose(r[key],altered[key],equal_nan=True)
    assert 0<=r['b_mass20']<=r['b_mass40']<=1+1e-12
    assert 0<=r['posterior_overlap']<=1+1e-12


def test_population_identity_not_ground_truth():
    grid,rates=fixture()
    counts=np.zeros((5,20),int)
    counts[:,0]=10
    a=decode(counts,rates[:20],grid)
    truth=np.tile(grid[np.argmax(np.linalg.norm(grid-a['mean'][-1],axis=1))],(5,1))
    row=endpoint_readout(a,a,counts,counts,grid,4,np.ones(len(grid)),cKDTree(grid),rates[:20],truth,truth)
    assert row['endpoint_separation_cm']==0
    assert row['a_truth_error_cm']>40


def test_temporal_conflict_and_strata():
    assert longest_run([0,1,1,0,1])==2
    assert longest_run([0,0])==0
    df=pd.DataFrame(dict(a_spikes=[0,3,6],b_spikes=[1,4,7],a_active=[0,2,4],b_active=[1,3,5]))
    assert support_stratum(df).tolist()==[0,4,8]


def summary_fixture(real_spikes=4):
    rows=[]
    for source,gen in [('real','observed')]+[(s,g) for g in ['matched_map','drift_gain']
            for s in ['sim_calibration','sim_test','sim_conflict']]:
        for i in range(40):
            sep=i/40 if source=='sim_calibration' else (100 if source=='sim_conflict' and gen=='matched_map' else 0)
            rows.append(dict(animal='RatA',session='RatA/Open1',window_uid=str(i),split=0,
                cohort='all_fixed_candidates',source=source,generator=gen,a_spikes=real_spikes if source=='real' else 4,
                b_spikes=real_spikes if source=='real' else 4,a_active=2,b_active=2,endpoint_separation_cm=sep,
                a_truth_error_cm=1,b_truth_error_cm=1,truth_separation_cm=80,persistent_conflict_ms=0))
    status=pd.DataFrame([dict(split=0,confirmed=True,local_match_all=True)])
    return pd.DataFrame(rows),status


def test_summary_requires_real_calibration_coverage(tmp_path):
    events,status=summary_fixture(real_spikes=8)
    summarize_disagreement(events,status,tmp_path)
    g=pd.read_csv(tmp_path/'disagreement_gate_summary.csv')
    assert not g[g.gate.eq('calibration_coverage_and_denominators')].passed.any()
    assert not g[g.generator.eq('overall')].passed.any()


def test_summary_cross_generator_and_primary_spike_strata(tmp_path):
    events,status=summary_fixture()
    summarize_disagreement(events,status,tmp_path)
    g=pd.read_csv(tmp_path/'disagreement_gate_summary.csv')
    axis=g[g.gate.eq('biological_conflict_interpretable')].set_index('generator').passed
    assert axis['matched_map']
    assert not axis['drift_gain'] and not axis['overall']
    c=pd.read_csv(tmp_path/'null_calibration_by_session.csv')
    assert c.calibration_stratification.eq('minimum_spike_count_only').all()


def test_summary_checks_both_population_recovery(tmp_path):
    events,status=summary_fixture()
    mask=events.source.eq('sim_test')
    events.loc[mask,'a_truth_error_cm']=39
    summarize_disagreement(events,status,tmp_path)
    d=pd.read_csv(tmp_path/'disagreement_by_session.csv')
    assert not d.calibration_resolution_pass.any()
    assert d.a_known_path_median_error_cm.eq(39).all()
