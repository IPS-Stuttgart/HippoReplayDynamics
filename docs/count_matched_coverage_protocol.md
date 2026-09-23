# Exact-count recording-coverage control

Frozen before outcome inspection, 2026-09-23.

## Question and cohort

Does observing a broader set of neurons change geometric trajectory acceptance
or future-neuron prediction after matching the number and timing of observed
spikes? Use every candidate in the complete independent-detector PF/Tanni cache:
33 sessions, nine animals, 12,696 events. No new event selection or thresholds.

## Paired intervention

Use all five frozen inference/evaluation partitions. The detection population
remains separate. For each event and split compare:

1. Full inference population, unmodified (reference).
2. Frozen half-neuron population, retaining all its observed spikes.
3. Five deterministic samples without replacement from the full inference
   population, each matching arm 2's total count in EVERY original 5-ms bin.

Zero-count bins remain zero. Whole 20-ms bins and overlapping windows therefore
have exactly matched population counts, durations, and edge-support criteria.
The matched sample can still contain spikes from the half-neuron population;
it is not a disjoint-cell comparison. Five draws are technical repeats, not N.

## Decoding and prediction

Decode each overlapping 20-ms window, shifted 5 ms, independently with a flat
spatial prior. In all arms use the SAME count-conditioned emission family:
log L(x) = sum_i n_i log[r_i(x) / sum_j r_j(x)], omitting location-independent
multinomial constants. The sum is over that arm's observable neurons. This
deliberately removes the total-rate/no-spike evidence used by ordinary Poisson
decoding. The full/half ordinary Poisson geometric labels are reported ONLY as
legacy regression checks, not mixed with the primary count-conditioned labels.

Use the existing longest MAP run with adjacent jumps <20 cm, >=10 frames and
>=40 cm displacement, with >=2-spike trimmed edges. This is the geometric
stage only, not full shuffle-validated replay identification.

Reuse the frozen out-of-event neural codebooks, chronological folds and matched
occupancy/self-transition null. Infer causally from each arm's inference spikes
and predict the same untouched evaluation neurons 40 ms ahead, over an
unobserved 20-ms gap. Score their identities conditional on evaluation count.
No future/evaluation spikes enter test-event filtering. This is a fixed-codebook
recording intervention, not a claim that a codebook can be learned after dropout.

The exact-count manipulation is an empirical finite-population subsample.
Because target counts come from the particular half-cell subset, count matching
does not create an exact new Poisson sampling model. Count-conditioned emissions
are a common working decoder, not a claim of a fully specified generative model
for this intervention. Matching counts does not match active-cell count, tuning,
or spatial coverage: those are the information differences being tested.

## Frozen endpoints and aggregation

Primary contrast: distributed-count-matched minus half-neuron, separately for
(a) geometric acceptance, and (b) conditional future log score per evaluation
spike. Also report dynamic-minus-matched-null advantage in each arm. Average
matched draws within event/split; average splits within event; average events
within session, sessions within animal, and animals within dataset. Report
individual animals, leave-one-animal-out results and descriptive exact animal
bootstrap intervals. Four/five animals are not large-sample confirmation.

Secondary: conditional-full-pass but half-fail cases, explicitly a training-only
selected diagnostic. Main comparison uses ALL frozen events. Zero evaluation
spike events count for geometry but are unavailable for per-spike prediction.
No retuning from the result. A null difference is not equivalence; a positive
result supports information beyond total spike count, not a biological mechanism.

## Technical gates

Pinned source hashes; all 33 sessions complete; five splits/five matched draws
per event; exact fine/coarse/overlap count equality; disjoint inference/evaluation;
identical targets; no scoring failures; finite geometry and supported scores;
legacy full/half Poisson labels unchanged. Independently verify the manipulation,
conditional MAP and predictive scores before interpreting results. Archive only
compact summaries, protocol, provenance and verification in the paper repository.
