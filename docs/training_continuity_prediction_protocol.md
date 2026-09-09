# Training-Only Continuity And Independent Prediction

Frozen before computing the new classification or stratified predictive results,
2026-09-10. This is a retrospective follow-up of extensively inspected datasets,
not a prospectively unseen biological confirmation.

## Question

Do candidates rejected by a transferred geometric continuity screen nevertheless
contain information that predicts separate neurons, beyond static, independent
position and nonspatial composition alternatives? Does the lost-under-thinning
subset have such independent support? This links the recording-coverage
benchmark to predictive validation without treating either as replay truth.

No new clock model, replay-speed claim, biological event selection or scoring
parameter tuning is involved. Candidate ascertainment used all neurons before
the historical split; inference is conditional on that frozen ascertainment.

## Fixed Inputs

Use all 9,225 high-MUA candidates from 33 sessions/nine animals: 4,001 PF and
5,224 Tanni. Keep the five existing 70/30 neuron partitions and RUN-only maps.
Primary score manifest SHA256:
`d16542d0bf935929a05a335f84cd2060167c76dd8e255378beb202fb84e97386`.
Order-by-map manifest SHA256:
`40f4b123142ddd47b22e7db1311602b849a4d996d943076d26c07e0485c524ae`.
Require both linked passing independent audits and verify the used files.

Original and shuffled predictive scores are reused without rescoring. They
infer from training cells alone, then score held-out cell identities conditional
on each held-out bin total. Held-out event spikes never update the inferred
posterior. Scores sum marginal predictive log probabilities, not joint event
evidence, future-time forecasts or evidence for an anatomical mechanism.

## New Training-Only Screen

Reconstruct overlapping 20 ms windows at 5 ms strides from native 5 ms counts.
Do not span an event edge or a partial final bin. Independently decode each
window using a uniform spatial prior and the untempered Poisson likelihood,
including the position-dependent silence term. This classifier is distinct
from the parent's count-conditioned temporal predictive models.

For each split, use only its training neurons. Trim unsupported edges to the
first and last windows with >=2 training spikes; do not remove internal windows
in the primary setting. Retain the earliest longest MAP run with adjacent
distances <20 cm and require >=10 frames and >=40 cm endpoint displacement.
Retain the audited 1e-9 cm tolerance for exact grid-boundary roundoff. Sensitivities:
11 frames, and internal windows requiring >=2 active training cells/>=3 spikes.

This is only the necessary geometric stage of the transferred PF-style screen.
Geometrically rejected candidates cannot pass its complete two-shuffle rule;
geometrically passing candidates are NOT labeled replay or shuffle-significant.
We do not redo the two 5,000-shuffle tests or claim exact author-pipeline
reproduction. The common RUN maps differ from the original authors' encoding.

Within each training split also hide half of the training cells, using a fixed
hash seed (`training_continuity_v1|20260910|dataset|animal|session|split`), without
using the held-out neurons or scores. Freeze the nested subset for the session.
Record retained/lost/gained/rejected geometric status, actual neuron indices,
MAP paths, longest-run lengths, displacements, support and failure reasons.
Predictions remain those from the original 70% training population: independent
support for an event lost by thinning does not mean it was recoverable from the
smaller training population. No such claim is made.

Primary classification uses the parent's valid-bin mask for consistent frozen
inputs. Sensitivity removes bin centres outside the recorded arena bounds.
Occupied grid cells can straddle boundaries, so clipping is not assumed to be
biological ground truth. It changes the classifier only, not the existing
predictive scores. Report counts of excluded centres and changed decisions.

## Endpoints And Aggregation

Primary group: geometric failures with at least ten edge-supported decoding
frames in the full training population. These candidates have an opportunity
to pass the duration requirement, but fail continuity/displacement. Keep short
and unsupported failures in separate descriptive groups and in the full cohort.
Secondary group: full-training geometric pass lost with nested-half training.

For each group evaluate all five paired held-out contrasts:

1. IMM minus independent positions.
2. IMM minus one static location.
3. IMM minus other-event global cell composition.
4. IMM original-order minus mean of 20 whole-bin-shuffled scores, real map.
5. The original-minus-shuffled contrast with real adjacency minus that with the
   parent's shared population-code permutation.

The five scores are not interchangeable: a shuffle effect alone does not rescue
failure against an unshuffled comparator. Diffusion and per-held-out-spike
contrasts are descriptive sensitivities, not replacement primary endpoints.
No new real-event replay labels are based on individual predictive signs.

First pair differences within split. Apply membership using that split's
training-only screen. Take one median across qualifying splits per event/group,
then session means, equal-session animal means, and equal-animal dataset means.
Report qualifying split counts. Events can enter different groups in different
splits, so group results are not independent samples; no unpaired group-difference
test is allowed. All retained events have equal event weight, not spike/split
weights. Keep zero-held-out events in raw endpoints; their per-spike value is
undefined, not zero.

Use 5,000 animal/session/event hierarchical bootstrap draws, seed 20260910.
Maps, candidate definitions, partitions and order permutations remain fixed;
their uncertainty is not covered. Only four/five animals underpin inference.
Compute intervals for the two declared groups at the primary 10-frame,
edge-only, parent-mask setting. Other groups and settings are descriptive.
Also report split-0 sensitivity without outcome-driven selection of a split.

A bounded primary predictive-support result requires every contrast's lower
pointwise interval >0 and positive animal means in all four/five source animals,
with a nonempty group in every animal. Empty groups fail availability, not pass
vacuously. This joint rule is not a unique-mechanism claim or a blanket
multiple-hypothesis correction across the project's historical analyses.

## Verification

Freeze and test code before classification. Check source hashes, exact event/
split/settings coverage, unchanged predictive files, and absence of held-out
cells from both classification populations. Reconstruct every MAP and label
independently from native counts; compare against the existing audited geometric
implementation. Allow different argmax indices only at independently verified
numerical ties and report them; no non-tied disagreement is acceptable.
Reconstruct all joins, group medians, session/animal summaries and primary
intervals independently. Tests include held-out count mutation, partial bins,
strict jump/displacement boundaries, longest-run ties, zero support, schema
failure and missing-event/split rejection.

## Novelty And Limits

Replay validation without ground truth and independent validation measures are
already established by [Takigawa et al.](https://elifesciences.org/articles/85635).
Comparing sequence detectors is not itself new; [Huh et al.](https://www.nature.com/articles/s41467-026-74822-2)
compare pairwise firing-order likelihood with decoded sequence metrics. Their
trial-type ordering model is not silently assumed to transfer to arbitrary 2D
routes. The possible addition here is independent neural support for events
rejected by a recording-sensitive geometric screen, linked to the same-event
population perturbation. Priority and biological interpretation remain open.
Do not call rejection a false negative without replay ground truth, equate
predictive success with accurate physical speed, or use this subgroup analysis
to rewrite failed whole-cohort replication gates.
