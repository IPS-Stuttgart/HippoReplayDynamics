# Grid-first regional inference: pilot outcome

2026-09-17. Audited synthetic falsification pilot, not real replay reconciliation.

## Decision

**Neither the five-point truth-error screen nor the same-event population-gap
screen passed.** Do not deploy the proposed correction on the real Home-rich/
Home-poor populations. Grid-first inference improves some average errors, but
does not make the tested populations coverage-invariant at the fitted optimum.

This does not prove every estimator using these encoding maps must fail, nor
that recording design is the only remaining lever. Finite-sample variability,
weak identification, observation-model mismatch and regularization remain
distinct issues; this one-replica pilot cannot separate all of them.

## Frozen experiment

Eight PF sessions, four rats, 1,139 common candidate-count templates, and all
22 frozen population definitions. Target: mean geometric Home occupancy over
the final 40 ms, using two independent 20-ms observation windows. All seven
stationary/moving/jumping scale strata and their equal empirical mixture were
tested at the four frozen binary-label quotas, using validation replica 0 only.
Those quotas are not occupancy quotas, and eight mixtures are not every mixture.

Calibration replica 0 was used solely to choose smoothing by three-fold held-out
window log likelihood. Folds were assigned by original event ID, keeping both
windows and all generated copies/strata of the same template together. No Home
mask or geometric truth entered the optimizer, smoothing graph or CV choice.

One global penalty was selected for both members of every pair and all sessions:

| Spatial penalty | Equal-animal held-out log score per window |
| --- | ---: |
| 0 | -17.323570 |
| 0.001 | -17.320333 |
| 0.01 | -17.315434 |
| 0.1 | -17.313142 |

The strongest tested penalty, 0.1, was selected. It is not a proven optimum over
all penalties, and the small CV differences are not equivalence/confidence tests.
No penalty was retuned after seeing regional truth or population gaps.

## First EM versus fitted grid weights

The first EM iterate from the flat grid prior exactly equals the mean flat-prior
posterior, as proposed. This is an algebraic identity, not evidence that every
first-step discrepancy is shrinkage or that subsequent regional estimates agree.
Unpenalized grid weights were initialized with 20 EM iterations and optimized to
a checked convex-simplex dual gap; the regularized fit uses the same grid-level
likelihood plus the frozen neighbor-smoothing penalty.

Full-population mean absolute occupancy errors, in percentage points:

| Observation likelihood | First EM | Unpenalized NPMLE | CV-regularized | Worst regularized error |
| --- | ---: | ---: | ---: | ---: |
| Conditional multinomial | 11.82 | 3.53 | 5.84 | 21.39 |
| Poisson | 11.63 | 5.03 | 6.15 | 23.23 |
| Fixed-total censored multinomial | 11.81 | 3.49 | 5.90 | 21.43 |

Unpenalized conditional inference improves the mean substantially, but its worst
full-population error is still 27.73 pp. Regularization reduces that extreme while
raising average regional error. Better held-out predictive likelihood does not
guarantee better accuracy for a particular regional functional.

The stationary conditional stratum alone has 3.23-pp mean / 14.13-pp maximum
unpenalized error, or 7.26-pp mean / 21.39-pp maximum regularized error. Thus the
five-point failure is not confined to within-bin moving trajectories.

## Same-event population discrepancy

Mean absolute high/low gaps, in percentage points, on identical simulated events:

| Population family / observation | First EM | Unpenalized | Regularized | Worst regularized gap |
| --- | ---: | ---: | ---: | ---: |
| Targeted / conditional | 6.39 | 13.82 | 11.17 | 23.85 |
| Whole-tetrode / conditional | 2.00 | 7.49 | 3.80 | 11.52 |
| Targeted / Poisson | 9.97 | 24.24 | 18.25 | 35.86 |
| Whole-tetrode / Poisson | 3.67 | 15.54 | 8.41 | 25.96 |
| Targeted / censored-total | 4.85 | 5.20 | 3.38 | 10.58 |
| Whole-tetrode / censored-total | 2.11 | 5.01 | 1.89 | 7.53 |

These are equal-comparison/equal-session means within each family (four targeted
session pairs and three whole-tetrode pairs), not animal-level confidence
intervals. Separate per-animal and equal-animal summaries are supplied. Reused
simulation templates and mixture panels are not independent animals.

Under conditional multinomial, targeted-high mean occupancy error is 3.97 pp
unpenalized / 4.51 pp regularized, while targeted-low error is 12.27 / 14.19 pp.
The proposed reconciliation therefore fails on the population contrast it was
intended to address; improvement in the full-population average is insufficient.
No comparison with the old real 11.38%/2.80% masses was made, since the cohort,
estimand and likelihood are not identical.

## Observation-model finding

The bank fixes whole-recording spike counts. Even the frozen full decoder uses
only 67-214 cells out of 80-263 recorded cells. Subset counts depend on position:
conditioning on a subset total does not generally leave a common location prior
independent of that total. The conditional-multinomial likelihood removes this
participation information along with the Poisson silence term.

The censored-total diagnostic retains N_total-N_subset as a single 'other cells'
category with the excluded cells' summed training rates. This is coherent with
fixed-total allocation before selection when position is constant within a bin.
It uses extra whole-recording information, not just the isolated subset's spikes.
Its improvement in targeted agreement is an informative model diagnostic, not a
validated remedy: its worst regularized gap is 10.58 pp and the targeted-low
regularized truth error remains 8.98 pp on average.

Native active-cell acceptance remains a formal conditioning caveat, although
only 10 spike-support rejections occurred across these 39,865 accepted simulated
event draws. Most proposal rejection was geometric label conditioning. Both
moving and jumping can change position within a 20-ms likelihood window.

## Spatial support preflight

The first attempt stopped before penalty selection because legacy matched maps
have more columns than the simulation grid. No arrays were truncated or matched
by index. Each population's native coordinates were reconstructed from the
hashed original matched archive and training encoder; every frozen rate entry
and Home mask was checked against that reconstruction. All generator grid
coordinates occur in the corresponding decoder grids; the legacy grids have
additional bins. The shared smoothing rule operates on each native grid.

The initial output folder is an aborted preflight, not a result. The authoritative
experiment folder is the v2 run listed below. No events, likelihood settings,
scientific thresholds or penalty-selection rules changed during this correction.

## Verification

- Final focused regression suite: 126 tests passed; Ruff clean for all six new
  Python source/test files.
- All 264 event-fold CV fits and 4,224 validation fits converged under the frozen
  dual-gap requirement; maximum independently recomputed gap was 9.996e-7.
- An independent audit reconstructed every first-iterate grid distribution,
  regional sum, objective, dual gap, truth error and paired-gap row.
- 1,680 raw time-window count vectors were separately reconstructed from saved
  spike identities and native time boundaries. Truth arrays matched the audited
  source bank; inputs and output hashes were verified.
- Synthetic optimizer fixtures agree with a separate SLSQP solver. Tests include
  the first-iterate identity, flat likelihoods, subset-count information loss,
  censored totals, smoothing, and non-vacuous event-CV penalty selection.

Numerical convergence is not regional identification. No real-data reconciliation,
real event-bootstrap intervals, nonspatial perturbation panel or own-population
MUA detection experiment was run after the primary synthetic screens failed.

## Artifacts

Repository: `/home/florianpfaff/HippoReplayDynamics-content-stability-20260914`

Base commit `8b8473018da6ab5c293b13600399be41686e6c15`, branch
`test-cross-dataset-content-stability`, with uncommitted research additions.
Run manifests record code/input hashes, commands and dirty state.

Under `/mnt/seagate10tb/florianpfaff/`:

- `regional-grid-mixture-pf-v2-20260917/`: authoritative CV, grid weights,
  likelihood settings, truth/gap tables and gates.
- `regional-grid-mixture-pf-audit-20260917/`: independent audit.
- `regional-grid-mixture-pf-report-20260917/`: non-rescoring report and figure.
- `regional-grid-mixture-pf-20260917/`: aborted coordinate preflight; do not use.

The defensible result is that grid-first inference improved mean full-population
accuracy but did not reconcile the weak-coverage populations under the proposed
likelihood and regularization scheme. Aggregate-after-inference is not by itself
a validated solution to regional content uncertainty.

