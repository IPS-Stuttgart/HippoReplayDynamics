# Neuron coverage, trajectory classification and future prediction

Frozen 2026-09-18 before new scores. This tests, rather than assumes, the
statement that trajectory classification can change when fewer neurons are
observed while predictive temporal information remains detectable. Already
explored recordings: not a preregistered independent biological replication.

## Cohort and fixed quantities

Reuse all 33 independently detected recordings of 2026-09-17: eight PF
sessions/four rats, 25 Tanni sessions/five rats. Verify the producer manifest
and all session input hashes. Do not redetect, replace, select new windows or
drop weak animals. Detection uses its reserved 20% RUN-QC population, disjoint
from inference/evaluation. Retain the original five 70/30 partitions of the
remaining cells, RUN maps, five chronological calibration folds and one-second
exclusion guards. Evaluation cells and target windows stay identical at every
coverage level. Count percentages relative to the FULL INFERENCE population,
not all recorded neurons; report absolute cell counts too.

Within every partition create ten deterministic nested cell orderings.
Fractions: 1, .75, .50, .25 (floor, minimum two). Repeat zero reproduces the
earlier full/half populations exactly; nine new orderings use seed 20260918.
Fractions are nested within each repeat, not between repeats. Full inference
is computed once, never counted ten times as independent evidence.

## Classification

Freeze the existing PF-style necessary geometric screen: independent flat-prior
Poisson MAP, 20-ms windows/5-ms strides, edge windows >=2 spikes; longest run
with adjacent jumps <20 cm must have >=10 frames and >=40 cm displacement.
No dynamical prior or temporal posterior smoothing enters classification.
The original two 5000-shuffle tests are NOT included; use the term geometric
trajectory screen, never shuffle-validated replay. All-candidate acceptance
curves and failure reasons are primary descriptive endpoints. Repeat with a
fixed full-population-supported event denominator to expose changes caused by
loss of decodable duration. Count both losses and gains, not just losses.

## Predictive endpoints

Use actual future held-out cell-identity scoring conditional on target spike
totals: 20-ms bins, target center 40 ms ahead, 20-ms unobserved gap. Never use
intervening/future inference spikes or target evaluation spikes to construct a
forecast. Primary comparator is the maximum-entropy occupancy/self-transition
matched null with its OWN causal filter. Global composition, no-history,
frozen-origin and shared-origin controls remain visible.

Arm A (fixed codebook): use frozen K50 HMMs trained on OTHER calibration events
from the prior run. Recompute inference and forecast from each subset. This
isolates test-event observation coverage, not loss of training information.
Arm B (restricted calibration): for repeat zero at every reduced level, refit
the SAME K50/2-restart/500-iteration model using only retained inference cells
plus the fixed evaluation cells from OTHER calibration events. Omitted cells
are absent from fitting and inference. Nonconvergence is failure, not grounds
to change K, prune a session, or substitute the fixed-codebook result. This
tests loss of calibration information too. Full-population fit is the same
source fit for both arms. Any missing arm invalidates a full-coverage claim.

Prespecified primary reduction: 50%. Other levels characterize the dose curve,
not choices for rescuing a failed primary. Groups: all, full-baseline pass,
lost (full passes, reduced fails), lost with reduced >=10 supported frames,
gained, retained, and full-baseline supported. Membership uses inference
neurons only. Compare reduced and full predictive scores on EXACTLY the same
event/partition/repeat/evaluation targets, computing differences BEFORE
aggregation. Report absolute advantages and paired decrease separately:
detectable information is not information equivalence or unchanged accuracy.

## Aggregation and interpretation

First median over qualifying repeats within partition/event; then median over
partitions within event, equal-event session means, equal-session animal means,
equal-animal dataset means. Classification probabilities use means (not
majority votes) at the two resampling levels before session/animal means.
No-target per-spike scores are undefined and remain explicit denominators.
Short events remain classification denominators but cannot receive forecasts.
Subsamples are technical repetitions, not additional biological replicates.

Report exact animal-bootstrap descriptive intervals, leave-one-animal-out
estimates and per-animal values; PF has only four rats. No arbitrary magnitude
threshold defines 'strong': quantify absolute percentage-point changes and
relative changes with uncertainty. Retention support at 50% requires complete
technical validation, positive reduced matched-null/global/no-history group
means and positive matched-null means in every represented animal, with the
descriptive interval above zero. Show both lost and lost-supported groups and
both arms. No-positive result is not evidence of no information. No universal
claim unless both datasets meet the same requirements. An unresolved or
negative Tanni result is retained, not tuned away.

## Validation and artifacts

Test deterministic nested subsets, disjoint evaluation/detection cells,
invariance to unused spikes, true future-gap scoring, matched-null constraints,
paired aggregation, non-vacuous failure gates, and full/half repeat-zero
reproduction of prior scores. Independently reconstruct sampled subsets,
geometry and forecast values plus ALL aggregate estimates. Verify hashes.
Save protocol, commits, cell IDs, per-row status/scores, event/session/animal
tables, failure reasons, denominators, technical gates, figures and provenance.
Run on gpuserver6000 detached with logs and terminal status, never restart on
a transient connection failure. Archive compact results and code in the paper
repo. No new uniform-speed, memory-function, IMM-specific or ground-truth-replay
claim follows from this test.
