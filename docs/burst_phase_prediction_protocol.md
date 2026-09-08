# Burst-Time Recruitment Versus Spatial Dynamics

Frozen before new scores. Exploratory on reused data, not independent discovery.

## Question
Can a stable early-to-late pattern of cell recruitment within population bursts
account for predictive information attributed to temporal sequence models?
This is normalized event time, NOT theta or ripple oscillation phase.

The preceding cross-context transfer test failed its positive-control ladder.
It also showed that shuffled-order sensitivity can coexist with failure to beat
independent-state prediction. Here test a specific physiological alternative:
stereotyped recruitment tied to the burst envelope rather than its spatial path.

Stereotyped neural cascades and limitations of replay shuffles are not new:
see https://pmc.ncbi.nlm.nih.gov/articles/PMC10983782/ and
https://elifesciences.org/articles/85635. This test does not itself establish
a novel mechanism. Its contribution, if substantial, would be the cross-event,
held-out-neuron dissociation between burst-time recruitment and spatial evidence.

## Fixed Inputs
All 9,225 existing immobile MUA candidates: PF 4,001 / eight sessions / four
animals; Tanni 5,224 / 25 recordings / five animals. Retain sparse recordings.
No continuity, replay-model, ripple, decoder-quality or outcome selection.
Reuse 20-ms count caches including the recorded final partial bin, five
chronological folds with one-second guards, and five fixed 70/30 neural splits.

Parent manifest SHA256:
`d16542d0bf935929a05a335f84cd2060167c76dd8e255378beb202fb84e97386`.
Rate-adapted spatial comparator manifest:
`84418e7db33df939b9609dae312eed19d72b7dc6e8d382d7cbdc91e4c1744943`.
Learned-assembly comparator manifest:
`a8507a6db14d07e7e6f8d5d709a571da949c9394a6c31e2ce8f8e54c88f7efa6`.
Require passing matching audits for both comparator sources and hash all used
inputs. No spatial model or HMM is refitted or rescored in this experiment.

## New Comparator
Let u=(count-bin center - event start)/(event end - event start). Learn cell
proportions as a smooth function of u from OTHER calibration events only.
Primary five uniformly spaced knot centers in (0,1), at (j+0.5)/5. Linearly
interpolate between neighboring knots, clamping to endpoint knots outside
their span. Fit each knot by linearly weighted calibration cell counts plus
100 pseudospikes toward the calibration global cell composition. That global
composition itself has 100 uniform population pseudospikes, matching the
learned-HMM global baseline. Normalize each knot across cells.

This is a fixed kernel estimate, not optimized latent-state inference. No
target spike, spatial coordinate, rate map or target model score enters its
fit. Primary five knots; three and ten knots are prespecified sensitivities.
Target bin probabilities are the convex interpolation of whole-population
knot proportions, then restricted and renormalized to held-out cells.
Score exact multinomial identities conditional on each held-out bin's total.
The new comparator does not use target training spikes at all.

Candidate boundaries were detected using all cells in the frozen parent.
All models share those windows. Prediction is conditional on that selection,
duration and held-out count totals; it is not an end-to-end held-out detector.
Event time uses seconds, not index/T, so partial-bin centers remain correct.

Score original target observations at all three knot settings. For the primary
five-knot model also score 20 frozen parent whole-bin permutations, keeping the
time coordinates fixed and moving each population vector together. This
preserves spike counts, windows and bin-vector contents. Existing source HMM
and spatial original/shuffle scores are reused without changes.

## Paired Endpoints
Primary five-knot differences per held-out spike:
1. Burst-time model minus calibration global (does recruitment transfer across events?).
2. Spatial IMM minus burst-time model (does spatial prediction add information?).
3. Learned K50 HMM minus burst-time model (does latent temporal modeling add information?).

Supportive: burst-time original minus mean shuffle; spatial IID minus burst-time;
same-emission IID minus burst-time; spatial/HMM global and order contrasts for
context. Raw nats/event and three/ten knots are sensitivities, not substitutes.
Do not label ratios of non-nested log-score improvements as causal mediation
or a fraction of biological replay explained.

Compute paired differences within neural split, median over the five splits
within event, equal-event means within session, equal-session means within
animal, and equal-animal means separately for PF and Tanni. Enumerate all n^n
animal-bootstrap draws (n=4 or 5). Intervals condition on events, folds and
fits and are not adjusted for the wider historical hypothesis search.

A transferable recruitment lead requires positive burst-time-minus-global
per-spike means in every animal of both datasets and positive lower bootstrap
limits in both. It must ALSO beat its own phase-averaged profile and shuffled
order on those same criteria. The phase-averaged predictor averages the learned
whole-population probabilities over the target event's known time coordinates,
then predicts all bins from that fixed composition. No target spikes enter this
average. This matched baseline prevents different global shrinkage from being
mistaken for a temporal recruitment effect. This safeguard was added before
any real phase-model fitting/scoring. An additional spatial signal requires the same pattern for
IMM-minus-burst-time. A failure of either is reported, not repaired by selecting
knots, animals, event subsets or a favorable normalization.

## Verification
Synthetic stationary and time-varying cell-composition fixtures; normalized
multinomial probabilities; exact phase interpolation and exposure clock;
zero held-out counts; target changes cannot refit calibration; missing model
rows and shuffles must fail. Independent reconstruction checks all calibration
counts/knot estimates and original/first-shuffle scores with separate formulas,
all reused comparisons, folds, support, normalization, event medians and
animal-level summaries. Native NWB/RUN-map fitting is not repeated.

No high-importance discovery, novelty or causal mechanism is declared solely
from a passing numerical test. Interpret alongside close published precedents.
