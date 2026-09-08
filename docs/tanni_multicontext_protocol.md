# Frozen cross-context prediction diagnostic

2026-09-08. Exploratory test in existing Tanni data, not independent confirmation
and not a new replay-definition or IMM-positive rescue.

## Scientific question

Can allowing another familiar arena explain high-MUA activity that is poorly
predicted by the current-arena map? The possible additional contribution is a
proper held-out-cell dissociation of current-map mismatch, contextual activity
and within-map position content. Remote replay itself is not new (Karlsson and
Frank2009, and later work). A positive context result alone is reactivation,
not evidence of temporal replay, continuous motion or memory consolidation.

[Tanni et al.](https://discovery.ucl.ac.uk/id/eprint/10152694/1/tanni.pdf), Cell
identification, states that all five recordings per animal were jointly spike
sorted from concatenated waveforms. Shared tetrode/cluster IDs can therefore
be aligned across arenas. They are not waveform-verified anew by this study.
Four familiar contexts A/B/C/D were visited; A was repeated. We do not treat
unvisited-today contexts as novel or unavailable memories.

## Inputs and first-half-only encoding bank

Coverage source manifest:
`2ed1c4d910a3ddc4027eefb398d48a55bae6829c3bf16e8f57955beb9ee55613`.
Frozen common-event prediction parent:
`d16542d0bf935929a05a335f84cd2060167c76dd8e255378beb202fb84e97386`.
Use all5,224 already-frozen high-MUA candidates,25recordings/fiveanimals.
Never select events by context score, trajectory geometry or held-out outcome.

Each of the five recording templates uses its cached first-half RUN rate map
only (8cm,1.5bin smoothing, RUN>=10cm/s). First-half occupancy>=0.05s determines
spatial support; second-half occupancy and full-session unit-QC masks are NOT
used. Use unit IDs available in every recording of that animal. Select units
using ONLY first-half maps: summed occupancy-weighted expected spikes>=30,
pooled occupancy-weighted mean rate<=4Hz, maximum rate in any template>=2Hz.
These are fitted-map-derived activity criteria, not literal counted spike QC,
and do not certify waveform cell type or cross-half firing stability.

Five templates represent four contexts. Prior context probabilities are1/4;
the two A templates each receive1/8. Current-context comparisons retain both
A templates, preventing an extra same-arena template from masquerading as a
remote-context improvement. Spatial priors are uniform within each template's
training-supported bins, NOT uniform across the concatenated state space.

Five deterministic70/30 cell partitions per animal, seeded20260908, are shared
across all its recordings. Every cell's encoding is learned from first-half RUN.
Inference uses replay training cells only; held-out cells never update context
weights or position posteriors. This is a new matched comparison, not directly
comparable to the old full-RUN/current-session unit populations or log scores.

## RUN validation before candidate interpretation

Take up to100 deterministically evenly distributed non-overlapping200ms windows
per recording from second-half RUN, starting>=1s after the source midpoint,
with tracking gaps<=0.1s and speed>=10cm/s throughout. Bin20ms, no trajectory
prior. Keep low-spike/zero-spike windows and report their support.
Validate context identification against known physical arena (A/A-return share
label). Report balanced accuracy by physical context then animal; posterior
correct-context probabilities; and current-context position-based held-out
prediction versus current-context global-composition prediction.

RUN interpretation gate: at least50 windows in every recording; every animal
has context-balanced accuracy>0.5 (four-class chance0.25) and positive
current-position-minus-current-global held-out score. No weak animal is removed.
If it fails, the candidate analysis may be retained as technical/descriptive
output only; it cannot establish remote spatial reactivation.

## Candidate models, no temporal prior

For each template compute independent-bin conditional multinomial likelihoods
and uniform-prior position posteriors using training cells. The template is
constant across the event; its weight is proportional to its prior times the
product of training-bin marginal likelihoods. Sum weights of the two A templates
for reported physical-context probability. Then, with weights/posteriors frozen,
score each held-out population vector conditioned on its total spike count.

- mix_iid: uncertain context plus independent positions within context.
- current_iid: same inference restricted to known current physical context.
- mix_global/current_global: context-dependent RUN composition, no spatial
  latent position; context weights again learned only from training neurons.
- blind_iid/blind_global: fixed prior context weights for comparison.
- event_global: current-recording other-event composition, five chronological
  folds with1s interval guard;100pseudo-spikes toward that recording's first-half
  RUN composition. Never use the target fold's candidate spikes to calibrate it.

Same bins and common cell population for every score. Partial final event bins
retain their observed counts. No powered/tempered likelihood; the target scores
are proper normalized multinomial marginal log scores, not joint log evidence.

## Primary and secondary outputs

Primary all-candidate paired contrasts: mix_iid-current_iid,
mix_iid-mix_global, and mix_iid-event_global. Calculate per-split differences
before taking event medians across the five splits. Mean events within a
recording, mean the two A recordings, then average four contexts equally per
animal, then animals equally. All raw and per-held-out-spike contrasts reported;
zero-count ratios remain missing. Raw differences remain primary.

Bootstrap animals and events within their fixed recording/context structure
5,000times, seed20260908. Five animals limit generality; frozen maps and cell
partitions are not resampled as independent biological subjects. A potential
cross-context spatial-content lead requires all three mean contrasts positive,
95%lower bounds>0 and all five animals positive, plus the RUN gate.
Failure of the nonspatial composition comparison stays a failure.

Report training-only physical-context probabilities, remote posterior mass and
remote mass>=0.8 counts descriptively. These are not a validated remote-replay
prevalence estimate. Do not select a favorable remote subset and call its
held-out result confirmatory. Context weights do not validate time order.

## Stop and novelty boundary

Retain failure, do not tune context priors, unit QC or events to rescue it.
A positive result would justify independent temporal/context-mismatch controls
before a memory claim. The high-importance-paper goal is not achieved by this
diagnostic, a successful software test, or a known remote-replay phenomenon.
