# Frozen control: position-free co-firing versus spatial prediction

## Question

Does the PF conditional held-out predictive result exceed an explicitly
position-free co-firing model learned from other candidate events? Apply the
same diagnostic to the already frozen hc-11 PRE/POST cohort, without reopening
its unsupported spatial temporal-replication claim.

This is a comparator study, not a newly invented replay mechanism. Maboudi et
al. (2018), https://pmc.ncbi.nlm.nih.gov/articles/PMC6013258/, already warns that
cross-validated HMM fit can arise from correlated groups without sequential
dynamics. Position-free latent models can inherit spatial structure from
spikes: they are NOT guaranteed non-replay nulls. Beating or losing to one
does not uniquely establish or exclude spatial replay.

## Frozen observations

PF: 160 original RUN ripple events, eight sessions/four rats, parent
pf-count-conditioned-prediction-all160x5-20260908/manifest.json, SHA256
5904fd3741ee61c5c8006a8962a36be34dacc02a5f2e33a43c378a4718d980f6.
Use the exact native 4 ms counts, cell IDs and five 70/30 splits. Reuse proper
T=1 count-conditioned spatial scores; do not rescore or retune spatial models.

hc-11: all 320 original PRE/POST events, eight sessions/four rats. Parent
hc11-conditional-cross-cell-prediction-320x5-20260908, manifest SHA256
bc846d25e27b9400d2870b2cad534c781d1fdee12f27496e5a8d96db153d3d07.
Keep 20 ms bins, native units/splits and both direction encodings. Position-
free scores are identical across direction variants; they are not independent
replicates. Compare within dataset, not raw score magnitudes across datasets.

Within each session/phase, split the 20 events chronologically into first ten
and last ten. Fit on the opposite half, excluding calibration events within
one second of a test interval. No test event is dropped. Require nonempty
calibration and positive total count. Record every exclusion.

## Position-free models

All parameters see only calibration-event spikes. No rate map, position,
direction or geometric distance enters these models. All units, including
neurons held out for a future test event, may provide spikes in independent
calibration events, never the scored event. This is cross-event plus cross-
neuron prediction, not identical training-data exposure to the RUN spatial
decoder. This asymmetry limits mechanistic model-selection claims.

Fit a global cell-composition vector using 100 population pseudospikes toward
uniform cells. Fit a mixture of normalized multinomial cell-composition
vectors: K=3 primary, K=8 capacity sensitivity. Each component has ten
pseudospikes toward the calibration global vector; mixture weights have one
pseudocount total toward equal weights. Fit standard MAP-EM with three fixed
seeded initializations, at most 1000 updates and relative penalized-objective
change below 1e-8. Choose only by calibration objective, never test scores.
Check monotonicity; if the highest-objective restart fails to converge, fail
the technical run rather than interpreting a deliberately weak comparator.

Calibration spike bins are nonoverlapping 20 ms bins (sum successive five
4 ms PF bins, retaining the final partial bin). Never pool across events.
Model fitting uses no temporal order. Keep zero-count calibration bins: they
carry no cell-identity information. Do not filter test bins for spike support.

Use fitted components in three frozen temporal variants:
- independent assembly assignment each bin;
- one fixed assembly for the entire event;
- persistent assembly: A = exp(-dt/0.06) I + (1-exp(-dt/0.06)) w 1^T,
  with fitted mixture weights w as the initial and stationary distribution.

The 60 ms retention scale is declared in advance, not a learned switching
time or a geometric-motion claim. Evaluate all variants; do not optimize it.
Infer each event's assignment posterior from training-cell identities given
their observed bin totals. Score held-out identities with proper normalized
multinomial probabilities conditional on their bin totals. Held-out spikes
never re-infer assignments. Also score the calibrated global composition.

## Endpoints and gates

Primary: PF T=1 spatial IMM minus K=3 persistent position-free assembly.
Report all spatial IMM/diffusion/independent/static models against global
and all six assembly comparators; temporal assembly versus independent and
fixed assembly; K=8 capacity sensitivity. For hc-11 primary direction mixture,
report PRE and POST separately and retain pooled sensitivity. Do not claim
state or dataset interactions by contrasting significance patterns.

Pair split scores, median across five splits per event, mean within session,
equal-session mean per animal, equal-animal mean. Report per-heldout-spike
sensitivity and 5000 animal-cluster-only bootstrap intervals, seed 20260908.
Four animals per dataset give limited population resolution; maps, calibration
fits and observed within-animal data remain fixed in these exploratory CIs.

A spatial advantage is a useful bounded lead only if positive against both
K=3 and K=8 persistent comparators in all four PF animals with lower intervals
above zero. It is not proof against every position-free model. If not, report
the temporal coordination as insufficiently spatially specific under this
comparator, not as false replay. No positive-only subset or new K/tau tuning.

Before real interpretation, verify synthetic known-assembly recovery and
null behavior, EM convergence, raw count/split agreement with parent sources,
cross-event nonleakage, separate exact predictive calculations, and aggregate
reconstruction. Synthetic validation is an operating check, not proof of
adequate power for arbitrary real spatial or assembly alternatives.
