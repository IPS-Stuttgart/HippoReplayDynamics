# Leave-rat-out learned content screening: failed development test

## Decision

Do not promote this rule to independent-recording validation. It reduces pooled
position/regional disagreement but fails Home-content uniformity, one entropy
guard, and known-position regional accuracy safeguards. The active validated
remedy/diagnostic goal remains OPEN; pooled improvements are not completion.

Protocol frozen at 16a3bd1c, implementation 788cbe6b; metadata repair and reporter
7489c94d. No model settings, retained fraction or numerical science gates changed
after results. All three exported models are exactly identical before and after
the metadata-only repair.

## Design

For each of Rat1, Rat2 and Rat4, train using ONLY the other two rats' earlier RUN
Q3 and separate Poisson/conditional calibration simulations. A fixed 128-leaf
regression tree partitions observable features of both populations. Shared leaf
retention probabilities are optimized subject to class-balanced accuracy and
agreement constraints, then used to rank a deterministic half-cohort.

This is not an A-only forecast of unseen neural activity: the paired input
features include regional probabilities and between-population disagreement.
The test is transfer of an accuracy-aware screening decision to an excluded rat.
The development banks were already used in earlier experiments, so this is not
a newly blinded confirmatory analysis or an external dataset replication.

Same original populations, maps and endpoints: 919/1,836 candidate endpoints
and 257/513 previously accepted-segment endpoints retained. No windows shifted,
cells added or likelihoods modified. Real-time durations remain 20 ms.

## Results

Equal-rat summaries, averaging sessions within rat first:

| Original candidate endpoints, early maps | All | Learned half |
|---|---:|---:|
| Home posterior gap (percentage points) | 9.0003 | 6.9918 |
| Position separation (cm) | 42.1505 | 34.6663 |
| Regional TV | 0.49713 | 0.42148 |
| High-population entropy | 0.69628 | 0.67688 |
| Low-population entropy | 0.73281 | 0.73802 |

Separation improves about17.8%, and regional TV about15.2%, with both improving
in all three rats. Pooled Home gap improves about22.3%, but its per-rat safeguard
fails:

| Rat | All Home gap (pp) | Learned half (pp) |
|---|---:|---:|
| Rat1 | 12.0779 | 12.3957 |
| Rat2 | 6.6206 | 7.8720 |
| Rat4 | 8.3024 | 0.7077 |

The apparent content remedy is carried by Rat4, not a consistent cross-rat
improvement. Full-map candidate gap improves only8.5829->7.2403pp and also fails.
Accepted endpoints improve7.3047->4.5025pp with early maps, but the full-map
sensitivity6.8679->6.5518pp fails its per-rat requirement. Low-population entropy
increases on the primary candidates, another failed guard.

Known-position pooled physical errors improve for all six sources. Nevertheless,
per-rat error guards fail in several sources, and high-population Home Brier
guards fail for ALL six sources. For native Q4, both populations' class-balanced
physical errors improve in every rat, yet Rat4's high-population Home Brier
increases0.32626->0.37289. Both true classes meet the20% minimum-retention gate.

## Failure localization (post-hoc readout, no new scoring)

Rat4 is where the real-data Home gap falls most. Its mean high/low Home posterior
mass changes from0.10738/0.02436 to0.03400/0.02692. But in known-Home native RUN
observations the high-population Home mass falls0.24210->0.16140; Home Brier
worsens0.64995->0.74378 and physical error worsens38.3078->42.0519cm.
225/460 known-Home observations are retained. For non-Home RUN observations,
2,847/5,684 are retained and physical error improves57.3017->50.8779cm.

Thus pooled physical-error improvements can hide worse recovery specifically
near Home. The real replay truth is unknown: this does not prove those events'
true destination, but it disqualifies calling the smaller real Home gap a
validated correction. Classwise tables for ALL rats/control sources are included
so the illustrative Rat4 result is not the only available diagnostic.

The calibration progress t itself reaches only0.589 (hold Rat1 out),0.424
(hold Rat2 out),0.526 (hold Rat4 out), below the full t=1 training target. This
observable partition/shared-source policy is more constrained than the previous
source-specific, outcome-informed oracle. Its training success must not be
inferred from the oracle's feasible bounds.

## Verification and artifacts

Run on gpuserver6000 in detached user service `learned-content-screen-v2-20260915`,
terminal exit0 in17.93s. Result directory:
`/mnt/seagate10tb/florianpfaff/learned-content-screen-v2-20260915`.

The initial run is retained separately at `learned-content-screen-20260915`.
It fit the same models and selected observations but could not summarize them
because legacy source event tables omit `animal`. The repaired reader validates
session identity and derives animal from it. No source file was changed.

Independent audit reconstructed33,910 calibration and109,274 evaluation rows
from original counts/rates, refit all3 trees and checked6 primal/dual certificates,
held-rat exclusion, feature columns, frozen-model timing, deterministic selection,
all summaries and gates. Evaluation rows include truth-bank controls and map
sensitivities, NOT109,274 independent replay events.

Maximum certificate constraint residual5.78e-14, stationarity1.88e-12,
duality gap1.21e-11. Measurement manifest SHA256:
`5a48caa9e429d298a89d67c562e63ed5cff9c8eaf7ae46b5bb4091cff5e13712`.
73 related tests passed and Ruff passed. The optional Matplotlib Axes3D warning
does not affect the inspected2D figure.

Authoritative compact report: `report-final/`, with training, real, truth,
per-animal/session, classwise regional, selected-score, gate and provenance tables.
All outcome interpretation remains development-only. No external run or
biological comparison was launched.
