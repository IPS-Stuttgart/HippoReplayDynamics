# Fresh-Path Clock Recovery: Audited Results

Date: 2026-09-10. Status: `fresh_path_primary_recovery_incomplete`.

The experiment completed on gpuserver6000. Technical validation passed, but
none of the four recording/stream groups passed all primary recovery gates.
No real replay events were rescored. This is calibration evidence, not a
biological constant-speed result or validation of fuzzy continuity criteria.

## Fixed Experiment

The protocol was frozen with producer commit
`8e88e0fab97d93362601737d17cab2e09147180c` before generation. It uses one
Pfeiffer/Foster recording (Rat1/Open1, 177 cells, 552 spatial bins) and one
Tanni recording (R2470, 2018-08-10_18-11-37, largest arena, 71 cells,
1,397 spatial bins). These two recordings do not support a dataset-wide
comparison or isolate cell count from environment size and map differences.

Every coherent event draws a fresh path from the complete finite scoring
prior, rather than a persistent 256-path source library. Stationary and
independent-bin reset generators remain included. Native event durations and
per-bin spike totals are preserved; new cell identities are simulated.

The scorer integrates over all 1,986,472 admissible paths without access to
generator identities or sampled paths. All five mixture weights are fitted.
The target is phi = neural / (neural + physical) among coherent events, not
the coherent fraction of all events and not a speed in cm/s. The neural clock
is constant arc-speed in the recorded population's normalized cell-probability
space, not an anatomical propagation speed across the hippocampus.

Total: 76,800 simulated event observations, 600 primary fits (128 events
each), and 12 predeclared secondary pooled fits (6,400 events each). The pooled
fits reuse the primary observations; they are not new independent evidence.

## Primary Results

Each row summarizes 50 repetitions at each of three true fractions, 0.25,
0.50 and 0.75. Gates were unchanged: absolute bias <=0.10, interval coverage
>=0.90, correct-direction power >=0.80, and false direction at phi=0.50 <=0.05.

| Recording | Stream | Worst absolute bias | Minimum coverage | Minimum directional power | False direction at 0.50 | All gates |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| PF Rat1/Open1 | 0 | 0.027 | 86% | 20% | 6% | Fail |
| PF Rat1/Open1 | 1 | 0.098 | 94% | 12% | 6% | Fail |
| Tanni largest arena | 0 | 0.198 | 98% | 0% | 2% | Fail |
| Tanni largest arena | 1 | 0.152 | 96% | 2% | 0% | Fail |

Across individual non-50% conditions, power is 12-26% for PF and 0-4% for
Tanni. Low power is the clearest failure. With only 50 repetitions, a 6% false
direction rate is three calls; it is not by itself strong evidence of a
miscalibrated nominal 5% rate. Wilson simulation intervals are retained in the
summary CSV.

The uncertainty cannot be inferred from mean estimates alone:

| Recording | Stream | True phi | Median interval width | Fraction with full [0, 1] interval | Median fitted coherent weight |
| --- | ---: | ---: | ---: | ---: | ---: |
| PF | 0 | 0.25 | 0.676 | 0% | 0.602 |
| PF | 0 | 0.50 | 0.800 | 2% | 0.602 |
| PF | 0 | 0.75 | 0.641 | 2% | 0.616 |
| PF | 1 | 0.25 | 0.758 | 2% | 0.597 |
| PF | 1 | 0.50 | 0.772 | 2% | 0.606 |
| PF | 1 | 0.75 | 0.789 | 8% | 0.609 |
| Tanni | 0 | 0.25 | 1.000 | 74% | 0.601 |
| Tanni | 0 | 0.50 | 1.000 | 72% | 0.587 |
| Tanni | 0 | 0.75 | 1.000 | 70% | 0.602 |
| Tanni | 1 | 0.25 | 1.000 | 62% | 0.592 |
| Tanni | 1 | 0.50 | 1.000 | 80% | 0.592 |
| Tanni | 1 | 0.75 | 1.000 | 68% | 0.598 |

These post-run descriptive diagnostics use `phi_high - phi_low` and exact
profile endpoints in `fresh_path_clock_fits.csv`; they do not change gates.
All 600 fits have coherent weight >=0.05. Thus the full-width intervals are
not simply produced by fitting no coherent events: the split between two
coherent clocks is poorly identified despite a fitted total coherent weight
near its generating value of 0.60. This observation does not validate recovery
over other coherent fractions, which were not varied here.

## Secondary Pooled Diagnostic

At the true 50% mixture, pooling improves precision and reduces the earlier
restricted-library bias in the more affected conditions:

| Recording | Stream | Previous pooled estimate | Fresh pooled estimate | Fresh conditional profile interval |
| --- | ---: | ---: | ---: | --- |
| PF | 0 | 0.597 | 0.489 | [0.421, 0.556] |
| PF | 1 | 0.599 | 0.540 | [0.473, 0.605] |
| Tanni | 0 | 0.215 | 0.465 | [0.295, 0.636] |
| Tanni | 1 | 0.503 | 0.444 | [0.273, 0.617] |

Across all 12 pooled conditions, 11 intervals contain the generating fraction.
This is not a validated 95% coverage estimate: there is only one pooled fit
per condition, two encoders, and these fits reuse the primary simulations.
One PF stream at true phi=0.75 estimates 0.676 with interval [0.608, 0.742].
Tanni stream 0 intervals still cross 0.50 even at true phi=0.25 and 0.75.
Pooled success therefore does not replace failed primary recovery gates.

## What Changed, And What Remains Open

Exhaustive integration removes Monte Carlo approximation over the specified
path prior. Fresh-path generation additionally removes reuse of a small path
library. These are different issues. The new pooled comparison suggests that
source-route sampling contributed to the earlier bias, while finite spike
information and mixture identifiability remain substantial limitations.

The restricted-library result should not simply be discarded as a coding
artifact. A restricted route distribution is also a legitimate mismatch
between the routes generating activity and the assumed scoring prior; real
replay may favor particular routes. Matching generator and scorer priors is
a best-case calibration, not a solution to unknown biological route use.
The old and new simulations use different sampled paths and spikes, so the
contrast does not isolate a paired causal effect of library size.

The methodological lead is that inference of a temporal law can depend on
latent route assumptions as well as recording information. This two-encoder
study does not establish novelty, a publication-ready result, uniform physical
replay speed, or a specific neural mechanism. The conceptual connection between
field spacing and sequence speed already has close precedents; see
`replay_clock_publication_scope.md`.

Before real-data clock claims, priorities are useful recovery power, route-prior
robustness, independent encoding maps, and broader recording coverage. Further
tuning merely to pass matched-prior gates would not resolve these limits.

## Verification And Provenance

Production runtime: 2,249.9 s. Independent audit: pass.

- All 76,800 event observations were independently regenerated, comprising
  669,170 time-bin population count vectors across 100 generation batches.
- Six predeclared first/middle/last score blocks were independently recomputed:
  59,785 paths and 139,176 partial log-likelihood sums.
- All 384,000 merged event/model likelihoods were checked.
- All 612 optimum and likelihood-profile fits were independently certified.
- Maximum numerical score error was 7.21e-10.
- All 235 production output hashes and eight report output hashes matched.
- Full regression: 230 tests passed; Ruff check/format passed on 27 files.
- Both PNGs were visually inspected and checked for nonblank pixels.

The audit does not independently rescore every path/event likelihood. It
independently regenerates all observations, checks all merges/fits, and samples
the predeclared numerical blocks.

Remote root: `/mnt/seagate10tb/florianpfaff/`.

| Artifact directory | Manifest | SHA256 |
| --- | --- | --- |
| `fresh-path-clock-recovery-pf-tanni-large-20260910` | `fresh_path_clock_manifest.json` | `a271a90ee8f323a6cbdd398013780fb1280776bab5b3b0cabc41c3f9bb16e1b9` |
| `fresh-path-clock-recovery-pf-tanni-large-20260910-audit` | `fresh_path_clock_audit.json` | `bc5beb9c16f2b4fb1445918e06f77ba3e3a667ab5270342dedc5afded8869f53` |
| `fresh-path-clock-recovery-pf-tanni-large-20260910-report` | `fresh_path_clock_report_manifest.json` | `8d70dcd8cad776a1c11bafa7c5384b3c5eb48b75a6618f932bf4f7a11962d977` |

Reporter commit: `67cea7e819be3e940fe107f8b6d4f4239b958eb6`.
Publication-scope note: `1db1b356f2e742cc21f859d85082125f713c2738`.
Validation artifacts: `fresh-path-clock-recovery-pf-tanni-large-20260910-validation-final/`.
No push or biological scaling was performed.
