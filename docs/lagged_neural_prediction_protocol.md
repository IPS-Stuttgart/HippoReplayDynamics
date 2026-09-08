# Lagged Neural Prediction Beyond State Persistence

Frozen before new real scores, 2026-09-09. Exploratory on previously examined
data; not a preregistered independent replication or a new discovery by design.

## Question And Novelty Boundary

Can observed population activity forecast later held-out neurons without
seeing intervening or contemporaneous training spikes? Does knowing preferred
state destinations help beyond knowing how long each state persists?

Previous held-out tests condition on other neurons at the target time, often
also future training observations. Those are valid conditional predictions but
do not isolate forecasting. Burst-time recruitment did not explain the stronger
spatial predictions, but its shuffle sensitivity illustrated why order tests
alone cannot establish predictive value. This is a new distinction, not a
parameter rescue of that negative test.

Forecasting neural activity and temporal hippocampal codes are established
ideas. Close precedents include associative versus sequential hippocampal codes
(https://pmc.ncbi.nlm.nih.gov/articles/PMC10894649/), predictive sequence learning
(https://www.sciencedirect.com/science/article/pii/S0896627324003714), and
independent replay evaluation (https://elifesciences.org/articles/85635).
Any eventual novelty claim must concern a verified mechanistic dissociation,
not simply a positive forecast score or absence of identical search wording.

## Frozen Inputs

- All 9,225 original immobile MUA candidates: 4,001 PF / 8 sessions / 4 rats;
  5,224 Tanni / 25 recordings / 5 rats. No continuity or replay-winner selection.
- Count cache/selection parent SHA256:
  d16542d0bf935929a05a335f84cd2060167c76dd8e255378beb202fb84e97386.
- Spatial calibration source SHA256:
  84418e7db33df939b9609dae312eed19d72b7dc6e8d382d7cbdc91e4c1744943.
- Learned K50 HMM source SHA256:
  a8507a6db14d07e7e6f8d5d709a571da949c9394a6c31e2ce8f8e54c88f7efa6.
- Require passing matching source audits; hash every used cache, selection,
  gain table and learned fit/fold manifest. Reuse all five event folds and five
  neural partitions. Do not refit RUN maps, gains, HMM parameters or events.
- Spatial per-cell gain alpha=100, as in the frozen parent. Learned HMM K=50.
- Only complete 20-ms bins enter this experiment. A final partial bin is
  explicitly counted and discarded, never silently treated as 20 ms.

## Causal Forecast

At origin bin t, filter the latent state using training cells in bins 0..t
only. No smoothing. Propagate the state distribution forward h transitions
without conditioning on observations in bins t+1..t+h. Score the held-out
population vector in bin t+h with a normalized mixture of multinomials,
conditional on that held-out bin's spike count. Never update the forecast with
held-out observations.

Horizon h=2 is primary (40 ms between centers, a 20-ms unobserved gap between
origin-window end and target-window start). h=1 and h=4 are fixed sensitivities
(20/80 ms between centers; 0/60 ms intervening gaps). Use every eligible
origin-target pair. All event/model/split/horizon rows are retained: too-short
events have explicit insufficient_full_bins status and no score, not a zero
score. Empty held-out target counts have zero raw score and undefined per-spike
ratios. Coverage is reported at each horizon, not inferred from row totals.

Models: spatial first-order IMM and diffusion, each with the real map and the
frozen joint population-code position permutation; learned neural K50 HMM.
Spatial constants exactly follow the parent: stationary Gaussian sigma=2 cm,
diffusion sigma=60*sqrt(0.02) cm, cutoff=3 sigma, three IMM modes, mode
self-transition exp(-0.02/0.06), uniform initial position/mode. HMM parameters
are fixed separate-event calibration fits. No inference temperature.

For each original filtered origin distribution, compare:

1. dynamic: propagate the frozen original transition operator h times.
2. dwell_only: retain state dwell probabilities but remove preferred destinations.
3. frozen: predict from the origin distribution with no further evolution.
4. no_history: propagate the model initial distribution to target time without
   conditioning on any event spikes.
5. global: separate-event calibration mean cell composition from the frozen
   learned HMM's global_probability (100 uniform-population pseudospikes).
6. same_time: filter through target training spikes, as a clearly labeled
   descriptive contemporaneous-information comparator, NOT a forecast.

For the learned HMM with row transition A and calibration occupancy w,
B_ii=A_ii and B_ij=(1-A_ii)*w_j/(1-w_i) for j!=i. This preserves each state's
geometric dwell distribution, but not necessarily its stationary occupancy.
For spatial transitions, preserve each source-position self-transition and
distribute the remaining mass uniformly over other valid positions. In IMM,
preserve the entire mode-transition matrix and apply this reset separately
to each destination mode's spatial kernel. This preserves mode switching and
within-mode position-stay probabilities, not every full-joint recurrence.
Both controls start from the same ORIGINAL filtered origin posterior; this
isolates forward propagation, not an optimally refitted competing model.

## Endpoints And Aggregation

At each horizon, sum target-bin scores within event/split, then form paired
contrasts and divide by the same total held-out target spikes. Median across
the five neural splits within event; equal events per session, equal sessions
per animal, equal animals per dataset. Report raw nats/event as sensitivity.
Exact animal bootstrap (4^4 / 5^5 resamples), conditional on fits/folds/events.
No pooled nine-animal significance or outcome-dependent endpoint changes.

Primary per-spike contrasts at h=2 for real spatial IMM, real diffusion and
learned HMM: dynamic minus dwell_only, frozen, no_history, and global.
Spatial map-specific increment: real-minus-permuted dynamic score AND
(dynamic-dwell_only)_real minus (dynamic-dwell_only)_permuted.
Supportive: dynamic minus same_time, spatial IMM minus learned HMM, and
IMM minus diffusion. All reported, none chosen as the headline after scoring.

A replicated route-prediction lead requires the same model to beat dwell_only,
frozen, no_history and global with positive animal means in every animal and
positive lower CI in both datasets at the PRIMARY horizon. A spatial-route
claim additionally requires both map contrasts on those criteria. Passing
does not establish unique IMM, causal sequence generation, behavioral
planning, biological replay prevalence or high-importance novelty.

## Verification

Synthetic directed sequences versus dwell-matched random transitions; sparse
versus dense IMM operators; exact tiny-state sequence enumeration; normalized
held-out multinomial mixture; forward-only suffix/held-out mutation tests;
full-bin/horizon eligibility; no-vacuous coverage; paired aggregation and null
score cases. Cross-check forward filters against existing prefix smoothers at
the prefix endpoint. Independent real-data audit reconstructs all score-table
contrasts and coverage, hashes and folds, plus a deterministic event per fold
and all its splits/horizons using a separately written filtering/forecast path.

The detector used all cells, maps may use later RUN, and calibration folds are
retrospective. Only the within-event forecast is causal. This does not become
an online causal detector, an intervention or an end-to-end held-out recording.
