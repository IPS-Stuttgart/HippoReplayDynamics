# PF Recorded-Cell Intervention: Destination Content

## Status

Completed on gpuserver6000, 2026-09-13. Code producer `4ca45839`.
Detached service `pf-recording-goal-content-20260913`, terminal exit status 0.
User lingering enabled to survive disconnects. No new raw recordings downloaded.

Result directory:
`/mnt/seagate10tb/florianpfaff/pf-recording-goal-content-20260913`

8 sessions, 4 rats, 4,001 unchanged eligible MUA candidates; 816 pass the frozen
full-cell two-shuffle trajectory screen. Three previously frozen half-population
subsets yield 12,003 paired candidate comparisons. The half subsets accept 333,
278 and 311 events respectively. Those are descriptive pooled counts; effects
below weight replicates, sessions and rats equally.

## Main Results

| Readout | Result | 95% rat-bootstrap interval |
|---|---:|---:|
| Fixed-candidate endpoint mean-estimate shift | 20.29 cm | 14.57 to 24.71 cm |
| Fixed-candidate posterior total variation | 0.469 | 0.432 to 0.497 |
| MAP Home/non-Home disagreement | 5.46% | 4.32 to 6.37% |
| Home posterior mass, full -> half | 9.37% -> 7.38% | paired change below |
| Home posterior-mass change | -1.99 pp | -2.64 to -1.32 pp |
| Full-accepted events, fixed-candidate Home mass change | -1.15 pp | -1.74 to -0.50 pp |

Home mass decreases in all four rats. The predeclared 10 and 30 cm neighborhoods
also decrease: -0.68 and -2.61 pp respectively. These are posterior probabilities,
not the percentage of events truly replaying Home.

All-candidate endpoint median shifts by rat: Rat1 11.32, Rat2 20.41, Rat3 24.32,
Rat4 25.10 cm. A shift from full decoding is instability, not absolute error.

## Aggregate Cancellation

For the endpoint of each population's earliest-longest continuous segment, the
full-to-half change in the accepted-event Home-mass summary decomposes into:

| Component | Change | 95% rat-bootstrap interval |
|---|---:|---:|
| Same events, same full-segment timestamp, different cells | -0.84 pp | -1.85 to -0.12 |
| Change in which events enter the summary | +0.98 pp | -0.56 to +2.36 |
| Change in retained segment endpoint timestamp | +0.06 pp | -0.23 to +0.36 |
| Total reselected summary change | +0.20 pp | -2.11 to +1.88 |

The components sum exactly. Selection partly offsets decoding loss in this run;
the composition interval includes zero. An apparently stable aggregate is not
evidence that individual event content is stable. The analysis does NOT show a
large or reliable change in the final reselected Home summary.

## Accepted-Trajectory Endpoint Check

Independent dense reconstruction of all 816 full-accepted events at their fixed
FULL continuous-segment endpoint (not the end of the whole candidate) gives a
20.68 cm mean-estimate shift. This post-run anchor diagnostic agrees with the
primary all-candidate result.

After additionally requiring >=3 spikes and >=2 active cells in that SAME
half-population endpoint bin, 1,442/2,448 event/subset comparisons remain; shift
is 12.83 cm (equal-rat mean of session/replicate medians). This is a descriptive
post-hoc sensitivity, not an independently selected confirmation. Poorly
supported endpoint bins explain some, but not all, of the instability.

## Known-Endpoint Simulations

1,600 straight 200 ms, 400 cm/s paths through the actual maps, same generated
spikes before/after cell removal. True paths and endpoint labels never change.
Exactly half end near Home. All simulations retained, no continuity selection.

- Median endpoint error: full 26.45 cm, half 37.27 cm; change +10.82 cm,
  rat-bootstrap interval +9.16 to +12.47 cm.
- True Home proportion 50%; MAP-classified Home 19.81% full, 12.19% half.
- Home sensitivity: 38.38% full, 22.92% half. Non-Home specificity: 98.75%, 98.54%.
- Classified Home prevalence change -7.63 pp, interval -9.19 to -6.06 pp, all rats negative.

Crucially, full-cell simulation recovery is already poor. These use RUN-rate
Poisson spiking without a replay gain, median 26 spikes per 200 ms (range 6-76),
and are not spike-matched to real events. They demonstrate a possible content
loss under known truth, not a quantitatively calibrated correction for real goal
replay or a valid biological neural simulator. More calibration would be needed
before translating this into a missed-goal count.

## Metadata and Validation

Home is the unique well shared by alternating fills (ID15 Open1, ID29 Open2).
Coordinates inferred independently from behavior; median across 21-43 arrivals,
with p75 spread 3.6-5.6 cm. The release documents fill times, not explicit Home
annotations or complete well coordinates. Rat2/Open1 has a 38.6 cm visit outlier,
retained and visible in the audit. No claim of exact author-analysis replication.

Producer gates 9/9 pass. All original MAP paths and geometric decisions match
the frozen benchmark. Independent audit: 264 sampled candidate/subset dense
reconstructions plus 2,448 accepted-segment comparisons; largest sampled numeric
discrepancy 1.15e-13. Source hashes, unchanged simulation labels, selection
decomposition and equal-rat aggregation verified. Audit uses a separate dense
likelihood calculation, not the producer's decoder.

## Interpretation

The first experiment supports a methodological extension: recording coverage
affects decoded destinations as well as continuity acceptance, and a stable
accepted-event aggregate can conceal event-level changes. It does not establish
that the animal represents the wrong goal, that the original PF planning result
is false, or that all observed Home mass is a sampling artifact.

Before a goal-planning claim, reproduce the appropriate reference readout with
validated well coordinates and behavior/distance-matched alternatives. Before a
missed-content correction, calibrate simulation spike support and extend recovery
across true goal prevalence. No further experiment is automatically launched.
