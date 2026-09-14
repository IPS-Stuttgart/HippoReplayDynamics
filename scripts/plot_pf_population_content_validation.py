#!/usr/bin/env python3
"""Plot existing disjoint-population and reliability tables without rescoring."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result-dir',type=Path,required=True)
    args=parser.parse_args()
    root=args.result_dir
    d=pd.read_csv(root/'disagreement_by_session.csv')
    d=d[d.subset.eq('spike_active_supported')]
    fig,axes=plt.subplots(1,2,figsize=(11,4.7),constrained_layout=True)
    palette={'matched_map':'#147d92','drift_gain':'#b94f5b'}
    marks={'Rat1':'o','Rat2':'s','Rat4':'^'}
    for gen,g in d.groupby('generator'):
        for animal,a in g.groupby('animal'):
            axes[0].scatter(a.matched_null_mean_separation_cm,a.matched_real_mean_separation_cm,
                color=palette[gen],marker=marks.get(animal,'o'),s=65,label=f'{gen}, {animal}')
    limits=[0,max(d.matched_null_mean_separation_cm.max(),d.matched_real_mean_separation_cm.max())*1.1]
    axes[0].plot(limits,limits,color='.5',linestyle='--',linewidth=1)
    axes[0].set(xlim=limits,ylim=limits,xlabel='Single-path null: mean separation (cm)',
        ylabel='Real events: mean separation (cm)',title='Support-stratified population disagreement')
    sessions=sorted(d.session.unique())
    for i,gen in enumerate(palette):
        g=d[d.generator.eq(gen)].set_index('session').reindex(sessions)
        error=np.maximum(g.a_known_path_median_error_cm,g.b_known_path_median_error_cm)
        axes[1].bar(np.arange(len(sessions))+(i-.5)*.36,error,
            width=.35,color=palette[gen],label=gen)
    axes[1].axhline(20,color='.4',linestyle='--',linewidth=1,label='20 cm recovery gate')
    axes[1].set(xticks=np.arange(len(sessions)),xticklabels=[s.replace('/','\n') for s in sessions],
        ylabel='Larger A/B median known-path error (cm)',title='Simulation recovery remains limited')
    axes[1].legend(fontsize=8,frameon=False)
    axes[0].legend(fontsize=7,frameon=False,ncol=2,loc='upper left')
    fig.suptitle('PF disjoint-population pilot: conditional diagnostics, not multiplexed replay evidence',fontsize=12)
    fig.savefig(root/'disjoint_population_validation.png',dpi=180)
    plt.close(fig)
    path=root/'reliability'/'metrics_equal_rat.csv'
    if not path.exists():
        return
    metrics=pd.read_csv(path)
    real=metrics[metrics.source.eq('real') & metrics.cohort.eq('all_fixed_candidates') & metrics.split.eq(0)]
    models=['prevalence','spikes','entropy','spikes_entropy','full']
    fig,axes=plt.subplots(1,2,figsize=(11,4.7),constrained_layout=True)
    for i,(radius,color) in enumerate([(20,'#147d92'),(40,'#bb7630')]):
        g=real[real.radius_cm.eq(radius)].set_index('model').reindex(models)
        axes[0].bar(np.arange(len(models))+(i-.5)*.36,g.log_loss,width=.35,color=color,label=f'{radius} cm B support')
    axes[0].set(xticks=np.arange(len(models)),xticklabels=['Prevalence','Spikes','Entropy','Spikes +\nentropy','Full'],
        ylabel='Leave-one-rat-out log loss (lower is better)',title='Does regional information add predictive value?')
    axes[0].legend(frameon=False,fontsize=8)
    policies=['full','spikes_entropy','threshold_spikes','threshold_entropy']
    for model,color in zip(policies,['#147d92','#8b5c91','#bb7630','#556b2f'],strict=True):
        g=real[real.radius_cm.eq(20)&real.model.eq(model)]
        if len(g) and g.claim_status.eq('ok').all():
            axes[1].scatter(100*g.claim_coverage,100*g.b_false_agreement,s=65,color=color,label=model)
    full=real[real.radius_cm.eq(20)&real.model.eq('full')]
    if len(full) and not full.claim_status.eq('ok').all():
        axes[1].text(.03,.05,f"Full model: {100*full.claim_coverage.iloc[0]:.1f}% coverage;\nclaim error undefined for at least one rat",
            transform=axes[1].transAxes,fontsize=8)
    axes[1].set(xlabel='Events receiving fine/coarse support claim (%)',ylabel='Claims not supported by population B (%)',
        title='Policies with evaluable claims in every rat',xlim=(0,100),ylim=(0,100))
    axes[1].legend(frameon=False,fontsize=8)
    fig.suptitle('A-only forecasts tested on excluded rats; three-rat pilot, equal-rat summaries',fontsize=12)
    fig.savefig(root/'population_content_reliability.png',dpi=180)
    plt.close(fig)


if __name__=='__main__':
    main()
