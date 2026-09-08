# Frozen training-only geometry versus predictive structure

Freeze before computing geometry labels or stratified predictive results.
This is a measurement-interpretation test, not a new replay definition.

## Question and Novelty Boundary

Do candidates rejected by geometric trajectory screening retain temporal-order
and spatial-adjacency information that predicts held-out neurons? Conventional
and HMM-based replay labels can disagree (Maboudi et al., 2018,
https://elifesciences.org/articles/34467), so disagreement alone is not novel.
The possible contribution is linking recording-dependent geometric rejection
to independently predictive structure on the same frozen 2D candidates, while
retaining nonspatial composition comparators and failures.

## Frozen Inputs

Use the audited 9,225-event common PF/Tanni cohort: 4,001 PF events/eight
sessions/four animals and 5,224 Tanni events/25 sessions/five animals. Reuse
RUN maps, unit QC, five cell partitions and ALL original/order-shuffled proper
predictive scores. Factorial manifest SHA256:
`40f4b123142ddd47b22e7db1311602b849a4d996d943076d26c07e0485c524ae`.
No new event selection, model scoring, map fitting or dynamics tuning.

## Geometry Labels

For each event and split, use ONLY that split's training cells. Pool complete
5 ms source count bins into overlapping 20 ms frames at 5 ms strides. Do not
use a partial final source bin for a complete frame. Decode each frame with
an independent uniform-prior Poisson observation model and the same 8 cm RUN
maps. No temporal prior, path smoothing or held-out event counts.

Primary screen: trim edges to frames containing at least two training spikes;
do not remove weak interior bins. Find the earliest longest MAP run whose
adjacent jumps are strictly <20 cm. Require >=10 decoded frames and >=40 cm
start-to-end displacement. Use the coverage benchmark's 1e-9 cm tolerance at
the exact 20/40 cm boundaries. This is the GEOMETRIC component of the transferred
PF-style criterion, not its 5,000+5,000 shuffle validation or an exact original
author pipeline. Never call geometric pass/reject true replay/non-replay.

Fixed sensitivities: >=11 frames, and a per-frame >=2 active training cells/
>=3 spikes filter (with both frame thresholds). Never bridge unsupported bins.
Record reasons for rejection, support, jump fraction and training entropy.

## No Selection Leakage

Primary: the already-frozen split 0, which was not chosen from results. Label
with its training cells; evaluate ONLY its held-out predictions. Report splits
1-4 separately as sensitivities. Do not combine labels across splits: a neuron
held out in one partition can be a training neuron in another. In particular,
do not define "all-splits rejected" and then present its held-out result as
independent validation. No repeated-split score pooling for the primary test.

Candidate ascertainment originally used all cells. This analysis is conditional
on that frozen detection, not training-only candidate detection. Label selection
uses original-order training activity; the predictive test measures its
generalization, not an unbiased estimate for all possible neural events.

## Endpoints and Aggregation

Report all, geometrically accepted and rejected strata, without claiming they
are quality-matched or that their difference is causal. The primary stratum is
geometric rejection under split0/edge-only/10-frame screening. Require positive
animal means and positive lower hierarchical intervals for:

1. IMM original minus independent-position prediction.
2. IMM original minus mean time-shuffled prediction under the real map.
3. IMM order by map interaction.

Only their conjunction supports the bounded phrase "predictive order and
adjacency survive geometric rejection". Additionally require IMM original minus
other-event composition to claim benefit beyond that nonspatial comparator.
Keep that fourth gate separate and never rescue it using the first three.
Diffusion's corresponding first three contrasts are fixed descriptive controls.

Reuse paired split-level contrasts, never differences of previously aggregated
medians. Fixed split0 gives one score per event. Average events within session,
sessions within animal, animals within dataset. Primary intervals use 5,000
animal/session/event hierarchical bootstrap draws, seed20260908, with all seven
contrasts paired in each draw. Also report per-held-out-spike sensitivities;
zero-count ratios are missing, not zero. Do not replace raw primary gates.

Keep zero-stratum sessions/animals in tables with counts zero and means missing.
Report actual and total session/animal coverage. A missing animal prevents an
all-animal gate pass. Five split sensitivities and alternate criteria get point
estimates/animal signs, not a search for whichever interval is favorable.

## Verification and Interpretation

Test exact frame timing/count conservation, boundary/tie/unsupported-bin
geometry, stationary/continuous/jumpy fixtures, and invariance to held-out count
perturbations. Independently reconstruct complete overlapping frame counts from
cached native spike timestamps, all training labels, score joins, strata and
primary intervals. Source artifacts and the parent native-count audit remain
hash-pinned; do not imply that RUN maps are independently refitted.

If rejection retains independent spatial-order prediction, it shows the
geometric screen is not exhaustive for that endpoint. It does not prove missed
continuous trajectories, estimate a biological false-negative rate, identify an
IMM circuit, or recover physical replay speed. If it fails, retain the failure;
do not loosen the geometry, predictor, event definition or dataset to make it
positive. High-importance novelty remains to be established beyond precedent.
