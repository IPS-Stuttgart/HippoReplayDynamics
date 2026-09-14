#!/usr/bin/env python3
"""Reconstruct frozen RUN/error and real/simulated endpoint outputs independently."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.spatial import cKDTree

from scripts._provenance import file_sha256, build_script_provenance
from scripts.analyze_pf_independent_population_content import load_checked, reflected_path, summarize_disagreement
from scripts.analyze_replay_coverage_shuffle_baseline import stable_seed


def independent_posterior(counts,rates,dt):
    logp=np.asarray(counts,dtype=float)@np.log(rates)-dt*rates.sum(axis=0)
    return np.exp(logp-logsumexp(logp,axis=-1,keepdims=True))


def frames(counts,durations):
    n=int(np.isclose(durations,.005,atol=1e-9,rtol=0).sum())
    return np.array([counts[i:i+4].sum(axis=0) for i in range(max(n-3,0))])


def sample_counts(totals,rates,states,rng,gain):
    weights=rates[:,states].T.copy()
    if gain is not None:
        weights*=gain
    p=weights/weights.sum(axis=1)[:,None]
    result=np.array([rng.multinomial(int(n),q) for n,q in zip(totals,p,strict=True)])
    np.testing.assert_array_equal(result.sum(axis=1),totals)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result-dir',type=Path,required=True)
    parser.add_argument('--refresh-summaries',action='store_true')
    args=parser.parse_args()
    root=args.result_dir
    manifest=json.loads((root/'manifest.json').read_text())
    if manifest['status']!='complete':
        raise ValueError('producer not complete')
    np.testing.assert_equal(file_sha256(root/'event_readouts.csv.gz'),manifest['event_readouts_sha256'])
    np.testing.assert_equal(file_sha256(root/'frozen_populations.json'),manifest['frozen_sha256'])
    params=manifest['parameters']
    benchmark=json.loads(Path(params['benchmark_manifest']).read_text())
    records={r['session']:r for r in benchmark['results'] if r['dataset']=='pfeiffer_foster'}
    states=json.loads((root/'frozen_populations.json').read_text())
    events=pd.read_csv(root/'event_readouts.csv.gz')
    previous=pd.read_csv(Path(params['previous_result_dir'])/'event_content.csv.gz')
    previous=previous[previous.population_replicate.eq(0)]
    run_audit,event_audit=[],[]
    for state in states:
        session=state['session']
        folder=root/session.replace('/','_')
        chosen_path=folder/'selected_before_confirmation.json'
        np.testing.assert_equal(file_sha256(chosen_path),state['selection_sha256'])
        chosen=json.loads(chosen_path.read_text())['selected']
        cache_path=Path(params['matched_dir'])/session.replace('/','_')/'run_validation.npz'
        run=load_checked(cache_path,state['run_cache_sha256'])
        for split,part in enumerate(state['confirmation']):
            np.testing.assert_equal(part['a'],chosen[split]['a'])
            np.testing.assert_equal(part['b'],chosen[split]['b'])
            assert len(part['a'])==len(part['b']) and not set(part['a'])&set(part['b'])
            for side in ('a','b'):
                indices=part[side]
                for block,label in [('match','match'),('confirm','confirm')]:
                    p=independent_posterior(run[f'{block}_counts'][:,indices],run['early_rates'][indices],.25)
                    error=np.linalg.norm(p@run['grid_cm']-run[f'{block}_windows'][:,2:],axis=1)
                    for metric,value in [('median',np.median(error)),('p75',np.quantile(error,.75))]:
                        expected=part['descriptors'][side][f'{label}_{metric}_cm']
                        np.testing.assert_allclose(value,expected,atol=1e-8)
                    run_audit.append(dict(session=session,split=split,side=side,block=block,n=len(error),median_cm=np.median(error),passed=True))
            subset=events[events.session.eq(session)&events.split.eq(split)]
            if not part['confirmed']:
                assert subset.empty
                continue
            real=subset[subset.source.eq('real')&subset.cohort.eq('all_fixed_candidates')]
            expected=previous[previous.session.eq(session)]
            assert set(real.window_uid)==set(expected.window_uid)
            assert len(real)==len(expected)
        if not any(p['confirmed'] for p in state['confirmation']):
            assert events[events.session.eq(session)].empty
            continue
        rec=records[session]
        arrays=load_checked(rec['input_arrays_path'],rec['input_arrays_sha256'])
        source=load_checked(rec['source']['source_cache_path'],rec['source']['source_cache_sha256'])
        second=source['rates_second_half_hz'][source['unit_qc_mask'].astype(bool)][:,arrays['support']]
        early,grid=run['early_rates'],run['grid_cm']
        lookup={u:i for i,u in enumerate(arrays['window_uids'])}
        tree=cKDTree(grid)
        sampled=events[events.session.eq(session)].groupby(['split','source','generator','cohort'],group_keys=False).head(3)
        for row in sampled.itertuples():
            part=state['confirmation'][row.split]
            ia,ib=part['a'],part['b']
            w=lookup[row.window_uid]
            a,b=arrays['base_offsets'][w:w+2]
            base,duration=arrays['base_counts'][a:b],arrays['base_durations_s'][a:b]
            if row.source=='real':
                ac,bc=frames(base[:,ia],duration),frames(base[:,ib],duration)
                truth_a=truth_b=None
            else:
                rng=np.random.default_rng(stable_seed(int(params['seed']),f'{session}/{row.split}/{row.window_uid}/{row.generator}/{row.draw}'))
                xy,_=reflected_path(grid,duration,rng)
                _,states_=tree.query(xy)
                generating=early if row.generator=='matched_map' else second
                ga=rng.lognormal(-.045,.3,len(ia)) if row.generator=='drift_gain' else None
                gb=rng.lognormal(-.045,.3,len(ib)) if row.generator=='drift_gain' else None
                sa=sample_counts(base[:,ia].sum(axis=1),generating[ia],states_,rng,ga)
                sb=sample_counts(base[:,ib].sum(axis=1),generating[ib],states_,rng,gb)
                truth_a=frames(grid[states_],duration)/4
                truth_b=truth_a
                if row.source=='sim_conflict':
                    _,alt=tree.query(grid.max(axis=0)+grid.min(axis=0)-xy)
                    sb=sample_counts(base[:,ib].sum(axis=1),generating[ib],alt,rng,gb)
                    truth_b=frames(grid[alt],duration)/4
                ac,bc=frames(sa,duration),frames(sb,duration)
                np.testing.assert_array_equal(ac.sum(axis=1),frames(base[:,ia],duration).sum(axis=1))
                np.testing.assert_array_equal(bc.sum(axis=1),frames(base[:,ib],duration).sum(axis=1))
            i=int(row.endpoint_frame)
            pa,pb=independent_posterior(ac[i],early[ia],.02),independent_posterior(bc[i],early[ib],.02)
            ma,mb=pa@grid,pb@grid
            distance=np.linalg.norm(grid-ma,axis=1)
            check=dict(a_spikes=int(ac[i].sum()),b_spikes=int(bc[i].sum()),
                a_active=int((ac[i]>0).sum()),b_active=int((bc[i]>0).sum()),
                a_x_cm=ma[0],a_y_cm=ma[1],b_x_cm=mb[0],b_y_cm=mb[1],
                b_mass20=pb[distance<=20].sum(),b_mass40=pb[distance<=40].sum(),
                a_width_cm=np.sqrt(max(pa@np.square(grid).sum(axis=1)-ma@ma,0)),
                endpoint_separation_cm=np.linalg.norm(ma-mb))
            if truth_a is not None:
                check.update(a_truth_error_cm=np.linalg.norm(ma-truth_a[i]),b_truth_error_cm=np.linalg.norm(mb-truth_b[i]))
            for key,value in check.items():
                np.testing.assert_allclose(value,getattr(row,key),atol=1e-7)
            event_audit.append(dict(session=session,split=row.split,window_uid=row.window_uid,
                source=row.source,generator=row.generator,cohort=row.cohort,checks=len(check),passed=True))
    pd.DataFrame(run_audit).to_csv(root/'independent_run_reconstruction.csv',index=False)
    pd.DataFrame(event_audit).to_csv(root/'independent_endpoint_reconstruction.csv',index=False)
    result=dict(passed=bool(run_audit and event_audit),run_blocks=len(run_audit),endpoints=len(event_audit),
        actual_source_hashes_verified=True,frozen_ids_verified=True,failed_partitions_excluded=True,
        base_and_overlap_totals_reconstructed=True,independent_dense_posterior_verified=True,
        producer_manifest_sha256=file_sha256(root/'manifest.json'),auditor_sha256=file_sha256(Path(__file__)))
    (root/'independent_audit.json').write_text(json.dumps(result,indent=2))
    if args.refresh_summaries:
        status=pd.read_csv(root/'population_confirmation.csv')
        summarize_disagreement(events,status,root)
        provenance=build_script_provenance(input_paths=dict(readouts=root/'event_readouts.csv.gz',
            producer_manifest=root/'manifest.json',confirmation=root/'population_confirmation.csv',
            summarizer=ROOT/'scripts/analyze_pf_independent_population_content.py',auditor=Path(__file__)))
        provenance.update(non_rescoring=True,changes=['spike-only primary calibration per frozen protocol',
            'per-population accuracy and active-support checks','nonvacuous coverage/denominators',
            'cross-generator overall AND'])
        (root/'summary_revision.json').write_text(json.dumps(provenance,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
