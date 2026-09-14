# hc-11 content observability preflight

Frozen before the counts audit. This is a feasibility experiment, not a new
successful reinterpretation of the failed PF-to-AutoPI diagnostic. No replay
posterior, model evidence, spatial-content outcome or predictive model enters
this audit. All eight public core hc-11 recordings, four rats, are attempted.

## Native data and denominators

Use spikes.cellinfo, CellClass.cellinfo and position.behavior MAT files. Join
excitatory labels by UID, not row order. Require explicit CA1/lCA1/rCA1 region
and native pE=1, pI=0; retain every source unit in the identity audit. Timestamps
are native seconds. Position metadata explicitly says meters; convert x/y to
cm exactly once. Do not apply a fitted time shift. Native MazeEpoch is RUN and
POSTEpoch is the population-burst detection epoch. These are track recordings,
not a second open-field dataset; linear/circular maze metadata stay visible.

Detect candidates with the existing AutoPI rest-MUA kernel, unchanged scientific
parameters: 1-ms summed activity, Gaussian SD 10 ms, peaks above epoch mean+3 SD,
mean-return boundaries, duration 50-2000 ms, >=5 spikes, >=3 and >=10% of native
CA1 excitatory units active. Boundary-clipped bursts are excluded explicitly.
Detect using ALL native CA1 excitatory units, before the activity screen below.
No immobility, ripple, NREM or replay claim follows from this detector. The
available native NREM-ripple files cover only Achilles/Cicero; they cannot stand
in for four-animal coverage and do not define the primary candidate pool.

## Information availability, not encoding validation

For an optimistic preflight, prescreen units on FIRST HALF of chronological
MazeEpoch only: at least 30 spikes during supported movement >=10 cm/s, mean
rate <=4 Hz during that supported movement. Compute centered unsmoothed speed
from native x/y; gaps >100 ms, nonfinite coordinates and block edges are excluded.
Use adjacent half-open supported frame intervals to count running spikes.
This does NOT test place fields, local coverage, field stability or decoder
accuracy. Subsequent encoding QC may remove more cells. POST spikes do not
influence the RUN screen or population assignment.

Partition activity-screened cells into two and three equal disjoint populations,
seeds 0,1,2 derived from seed 20260914 + dataset/session/population count; split0
is primary. Drop at most k-1 cells in k groups. Require at least five/group.
No favorable seed replaces a failed primary result.

Count spikes/active cells in two separately labeled 20-ms windows:
1. Primary: last complete 20 ms on the candidate's original 5-ms grid, matching
   the endpoint convention of earlier experiments. Do not move the endpoint.
2. Diagnostic only: native MUA-peak-centered 20 ms, only when wholly inside
   the candidate. Missing peak windows stay in the denominator as unavailable.
The peak does not replace the endpoint. Report both without picking by outcomes.

An activity-supported population window has >=3 spikes from >=2 cells. Report
the fraction with ALL groups supported, individual spike/active-cell distributions,
both/all-silent fractions, dropped cell IDs and acquisition alignment checks.
These thresholds are information preflight, not certification of localization.

## Predeclared feasibility decision

An ENDPOINT follow-up is a candidate for further encoding validation only if
all four rats are retained, >=80% of detected candidate endpoints can be measured,
and at least 20% of endpoints have activity support in both halves in >=3/4 rats
(sessions weighted equally within rat). Three-group results are a stricter
secondary check. No bootstrap of cells/windows pretends to add animals.

Any failure means the proposed fixed-endpoint observation design is not ready
for another predictive-content validation. A peak-only positive result would
motivate a NEW readout-design proposal, not validate endpoint-content estimates.
Even a pass requires later encoding, known-position recovery and independent
population-content tests. The active remedy/diagnostic goal remains open.

## Outputs and checks

Write per-session native inputs/counts, unit audit, candidates, window-level
population counts, per-session/per-animal summaries, feasibility gates, markdown
report and exact source/code/protocol hashes. Independently recount saved windows
from native spike timestamps. Unit tests cover UID reordering/missing labels,
native units/clocks, first-half screening, endpoint timing, disjoint partitions,
empty/failing denominators and scalar recount agreement. Use detached systemd
services on gpuserver6000. Do not score replay or fit a predictive model here.
