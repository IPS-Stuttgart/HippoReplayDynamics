# Frozen diagnostic: cross-event sleep relative-rate recalibration

## Question and interpretation boundary

Does globally reweighting RUN place maps using distinct sleep events improve
proper cross-cell prediction, and does it reveal a temporal advantage beyond
independent spatial snapshots? This follows an inconclusive frozen hc-11 sleep
result and a positive primary actual-MAZE cross-time control. It is exploratory,
not an independent replication of a pre-existing sleep claim.

Marginal cell-frequency differences may reflect changed replayed occupancy or
sampling, not physiological gain. This diagnostic does not identify gain,
remapping, experience dependence, or a uniquely switching brain mechanism.
Global rate changes, learned tuning during sleep, and replay rate information
already have precedent (Tirole et al., eLife 2022, 79031; Maboudi et al., Nature
2024, doi:10.1038/s41586-024-07397-x). The purpose is to discriminate a simple
observation-transfer explanation before testing richer mechanisms.

## Frozen source and observations

- Parent: hc11-conditional-cross-cell-prediction-320x5-20260908.
- Manifest SHA256: bc846d25e27b9400d2870b2cad534c781d1fdee12f27496e5a8d96db153d3d07.
- All 320 already-selected events, 20 PRE plus 20 POST per session, eight
  sessions, four rats. No outcome-based event or animal exclusions.
- Native 20 ms bins, CA1 unit identities, RUN maps and five 70/30 neural splits
  remain exactly frozen. Both direction-mixture and pooled maps; T=1 only.
- Four models: independent positions, static location, diffusion, first-order
  IMM. Kernels and the single population-code permutation remain frozen.

## Cross-event calibration, never the scored event

Within each session and sleep phase, sort events by start time then ID. The
first ten and last ten define two contiguous folds. Calibrate each test fold
using only the opposite fold. Exclude calibration intervals within one second
of any test interval, and record every exclusion. Require at least one
calibration event and positive calibration count; otherwise fail the run,
without a silent fallback or cohort change. All test events are retained.

Calibration uses counts from all frozen units, including cells later held out
for a test event, but NEVER spikes from that event or its guarded fold. This
is analogous to separately learning held-out neurons' encoding parameters.
It is cross-event as well as cross-neuron validation, not an untouched sleep
encoder. Temporal neuron splits remain identical across all conditions.

Let p0_i be the normalized spatial mean of the pooled RUN firing-rate map.
This is a declared uniform-space reference, not measured sleep occupancy.
Let c_i be each cell's total count in calibration events, C=sum(c_i). Set:

    p_cal_i = (c_i + alpha * p0_i) / (C + alpha)
    gain_i = p_cal_i / p0_i
    adapted_rate_i(x,direction) = gain_i * RUN_rate_i(x,direction)

Conditions: original RUN map; alpha=100 (primary recalibration); alpha=1000
(stronger-shrinkage sensitivity). Alpha denotes population pseudospikes.
Do not tune alpha against held-out outcomes, add a zero-prior fit, or choose
the condition that yields a preferred winner. The same gain multiplies all
positions and directions of a cell; there is no spatial retuning.

For each event/split, infer latent posteriors using training-cell identities
conditional on their bin totals. Held-out scores use normalized multinomial
probabilities conditional on held-out bin totals. Held-out event spikes never
update latent inference. Retain zero-heldout-count observations as zero scores;
per-spike ratios are undefined there. Report an adapted nonspatial baseline
using p_cal restricted and renormalized to held-out cells. Original uses p0.

## Contrasts and inference

Primary: POST, direction mixture, alpha=100. All variants/states retained.
Report IMM-independent, IMM-static, diffusion-independent, IMM-global,
independent-global, real-permuted IMM, recalibration improvement of IMM and
global scores, and the interaction:

    (adapted IMM - adapted independent) - (original IMM - original independent)

Form paired split contrasts, median across five splits per event, mean within
session, equal-session mean per animal, equal-animal mean. Report four animal
point estimates, per-heldout-spike sensitivity, and 5000 animal-cluster-only
bootstrap intervals (seed 20260908). These condition on maps, calibration
observations and within-animal data; they are exploratory, not full nuisance-
estimation uncertainty or familywise inference. With four rats, an exact
one-sided animal sign-flip test cannot attain p<0.05.

## Decision before looking

Recalibration alone improving absolute scores is not replay evidence.
Advance only if primary adapted temporal-versus-independent and static
contrasts are positive with lower cluster intervals above zero and all four
animal means positive, and the real-map predictive advantage is positive.
Otherwise retain the unsupported external-temporal-advantage verdict and
report whether the improvement was rate-only. Never redefine the null or
select only successful animals to rescue it. Even a pass is a diagnostic
lead requiring new events/replication, not a high-importance discovery.

## Verification

Before interpretation: check parent and cache hashes, complete event/model/
split factors, disjoint calibration/test intervals and neuron IDs, proper
nonpositive log scores, unchanged posterior hashes, agreement with original
unadapted parent scores, independent rate/predictive recomputation, and
per-animal aggregate reconstruction. Tests must include injected count/rate
changes, changing held-out spikes without changing inference, guard handling,
empty calibration, zero counts, normalization, and non-vacuous completeness.
