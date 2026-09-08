# Frozen conditional-prediction order by map factorial

Question: does the temporal-versus-independent predictive benefit in the
common-pipeline PF/Tanni experiment require time order, and is that dependence
stronger with correct spatial adjacency? This is a new predictive-endpoint
control, not reuse of old all-cell log evidence or temperature-scaled scores.

## Fixed Inputs

Use all 9,225 previously frozen high-MUA events, all 33 sessions/nine animals,
the same RUN maps, spatial masks, five 70/30 cell splits, multinomial observation
model and exact physical diffusion/IMM parameters. Parent manifest SHA256:
`d16542d0bf935929a05a335f84cd2060167c76dd8e255378beb202fb84e97386`.
Require the passing independent parent audit. Reuse its original-order scores.
Do not select trajectory-positive, ripple-positive or favorable-animal subsets.

## Factorial

For every event generate 20 independently sampled whole-bin permutations,
seed 20260908 hashed with dataset/animal/session/event/shuffle identities.
Use the same permutation for every cell, both maps and all five cell splits.
Carry the bin width with its complete population count vector; this preserves
partial-bin exposure, every cell's total, total duration and population snapshots.
Rebuild time centers from the permuted widths. Include identity permutations
and duplicate random draws; do not condition the null on being different.
Record permutation indices and unique/identity counts, including short events.

Evaluate all four cells: real/original, real/shuffled, permuted-map/original,
permuted-map/shuffled. The wrong map is the SAME single shared population-code
permutation from the parent. It changes spatial adjacency, not population-code
content. It is not a genuine alternate-context map or a universal no-replay null.

Infer from training cells alone. Only after inference, score held-out cell
identities conditional on their totals using frozen smoothed marginal
posteriors. No likelihood tempering and no held-out posterior update. Score
diffusion and first-order IMM; independently/static-position models are
invariant-score checks under both manipulations. Global composition baselines
are unchanged and are not recalibrated on shuffled test events.

## Endpoints

Within each event/cell split, use the MEAN of the 20 shuffled predictive scores:

    real_order_advantage = real_original - mean(real_shuffled)
    wrong_order_advantage = wrong_original - mean(wrong_shuffled)
    interaction = real_order_advantage - wrong_order_advantage

For each dynamic model also report original-minus-independent and mean
shuffle-minus-independent under each map, and median-shuffle order-advantage
sensitivities. Compute paired differences first, then five-split medians per
event, session means, equal-session animal means and equal-animal dataset means.
Do not subtract separately aggregated medians to reconstruct an interaction.

Primary new endpoints are the IMM real-order advantage and order-by-map
interaction in Tanni. Both require positive lower pointwise hierarchical
bootstrap intervals and positive means in all five animals for a bounded
order-and-adjacency result. PF is a same-pipeline supporting comparison, not
independent novelty. Diffusion contrasts and per-held-out-spike normalization
are descriptive sensitivities, not replacement gates.

Use the parent 5,000-draw animal/session/event hierarchical bootstrap and seed.
Maps, neuron splits, permutations and fitted composition baselines stay fixed.
Animal count remains four/five, not thousands of independent biological units.
Individual shuffle p-values, if reported, are descriptive Monte Carlo ranks
with minimum 1/21; they are not multiple-testing-corrected replay labels.

The parent Tanni advantage over other-event composition did not pass. This
factorial cannot retrospectively rescue that gate. Even a positive interaction
does not identify a unique IMM mechanism or exclude temporally structured
co-firing. Time-order controls and cross-validated sequence models have direct
precedent in Maboudi et al. (2018), eLife 34467. Do not claim novelty for shuffling.

## Verification

Before scientific interpretation require synthetic exact-inference/shuffle
tests, partial-bin/count preservation, held-out nonleakage, full factorial and
nonvacuous gates, original-score invariances, exact source hashes, independent
dynamic score reconstruction across both datasets/all sessions, and independent
reconstruction of paired summaries and intervals. Preserve failures, zero-held-
out cases and the complete parent cohort; do not tune settings to force a pass.
