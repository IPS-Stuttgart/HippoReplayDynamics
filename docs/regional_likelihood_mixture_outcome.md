# Regional likelihood-mixture prevalence: failed remedy screen

Completed on gpuserver6000, 2026-09-15. Protocol commit `8682e91c` preceded
new estimates; numerical producer/auditor commit `04db2bee`. Reporting-only
dependency fix `6d1bb022` did not change or rerun measurement or the audit.

## What was tested

Estimate regional prevalence from each population's complete distribution of
Home-versus-outside likelihood ratios, instead of averaging its uniform-prior
posterior mass. This applies established mixture/prior-adjustment estimation,
not a new statistical principle (Saerens et al., 2002,
doi:10.1162/089976602753284446).

The test preserves the four original confirmed PF matched pairs, 1,836 fixed
candidates, 513 previously accepted fixed segment endpoints, cell subsets,
Home region and 20-ms bins. Each population fits its own mixture without access
to the other population's estimate or any target truth. The regional mixture
proportion is a DIFFERENT estimand from mean posterior mass; known-truth
prevalence recovery is therefore essential, not an optional extra.

## Primary Poisson results

Errors below average the five predeclared prevalence panels and both populations
within session, sessions within rat, then equally weight the three rats.
The panels reuse each source bank; they are not independent replicates or
calibrated estimates of performance on arbitrary real replay events.

| Known-truth source | Mean-posterior absolute error (pp) | Mixture absolute error (pp) |
| --- | ---: | ---: |
| Native RUN, reweighted prevalence panels | 27.59 | 5.71 |
| Native RUN, natural observed prevalence | 3.85 | 3.94 |
| Matched Poisson simulation | 27.68 | 0.69 |
| Poisson simulation with gain x4 | 21.74 | 21.17 |
| Event-total conditional counts | 25.27 | 15.05 |
| Late-map drift | 25.67 | 15.44 |
| 80% shared nonspatial assembly | 25.57 | 43.32 |

Matched Poisson recovery passed the frozen <=5 pp and >=20% improvement gate,
including the no-rat-worsened safeguard. Native reweighted RUN improved but
missed the <=5 pp requirement; natural RUN did not improve. Gain and shared-
assembly no-harm gates failed (the gain pooled mean hides a rat-level failure).

The central real-data target worsened, not merely by a marginal amount:

| Same-event content contrast | Original absolute gap (pp) | Mixture absolute gap (pp) |
| --- | ---: | ---: |
| Early map, fixed candidates | 9.00 | 26.84 |
| Original full map, fixed candidates | 8.58 | 26.27 |
| Early map, accepted segment endpoints | 7.30 | 19.47 |
| Original full map, accepted segment endpoints | 6.87 | 18.33 |

All four original-content gates fail. Fourteen of 32 real Poisson estimates
(four sessions x two populations x two maps x two cohorts) are boundary-zero
fits. These are explicitly retained and cannot be called calibrated absence
of Home replay. In the full-map candidate analysis, the low-Home population
fits zero in Rat1/Open1, Rat2/Open1 and Rat4/Open2, while the paired high-Home
estimates are 26.07%, 28.59% and 21.07% respectively.

The exploratory conditional-on-count likelihood does not rescue the result:
its full-map candidate gap increases from 6.09 to 19.94 pp, and its shared-
assembly truth error increases from 24.66 to 50.90 pp. It is not promoted over
the frozen Poisson primary after examining these outcomes.

## Interpretation and decision

The numerical estimator works under its matched generative assumptions, but
does not correct the observed population sensitivity. Excellent recovery in a
matched Poisson simulation is not a sufficient transfer test for real regional
content. Shared assemblies provide one explicit, known-truth counterexample;
they do not prove that such an assembly caused the real PF discrepancy.

This result rules out simple model-implied prior adjustment as the current
remedy. It does not establish biological Home prevalence or refute the original
PF planning result. It also does not prove all calibration approaches fail.
No threshold, likelihood, region, event time or population was changed to turn
the scientific failure into a pass. No independent dataset was run because the
development screen failed. The full independent-data remedy goal remains open.

## Verification and artifacts

Server root:
`/mnt/seagate10tb/florianpfaff/regional-likelihood-mixture-20260915`

- `source-audit/reconstruction.json`: a fresh reconstruction of all four cached
  source sessions, native spike counts and synthetic banks; snapshots verified
  unchanged before and after the campaign.
- `measurement/`: 560 estimates, cached likelihood ratios, native/simulation
  summaries, original-content summaries, gates and code/input manifest.
- `independent_audit.json`: separately reconstructs 437,096 regional likelihoods
  from counts/rates including full probability constants, verifies estimates
  with an independent scalar optimizer, and rebuilds all gate-relevant tables.
- `report-v2/`: authoritative final non-rescoring report, six compact CSVs,
  two-panel figure and hash manifest. `report/` is a preserved incomplete output
  from an optional `tabulate` formatting dependency failure, NOT a valid report.
- Numerical manifest SHA256:
  `74c05786572a5075b37aac746f61f16fa38fd81d8c94a858a81b699d2964c5d4`.
- 36 related tests passed. Tests include known mixture truth, nonidentification,
  boundary fits, count conditioning, equal-rat weighting, missing/duplicate rows,
  single-rat truth harm, independent likelihood reconstruction, full reporting,
  and deliberately tampered scientific summaries with recomputed checksums.
- Ruff passed. The rendered figure was visually checked for legibility.

The detached campaign completed all measurement/audit work, then terminated
at the report-only dependency error. The amended reporter completed separately
with exit code zero. No measurement was repeated and no live job remains.
