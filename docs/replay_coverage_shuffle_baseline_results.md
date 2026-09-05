# Coverage Under a Transferred PF-Style Two-Shuffle Criterion

Status: full-budget benchmark completed and independently reconstructed on
gpuserver6000. This establishes recording sensitivity of the specified test,
not latent replay truth, biological prevalence or speed uniformity.

## Provenance and Verification

- Artifact: `/mnt/seagate10tb/florianpfaff/replay-coverage-shuffle-baseline-all33-20260905/`.
- Scorer/auditor commit: `8bc0f9c543478a9b54aac4120f8b5496ed756289`, clean.
- Scoring manifest SHA256:
  `4221be8c927177561f7081da754e3e883dd895b837c987e531ad17c5a68c8bc3`.
- All 33 sessions, nine animals: PF eight/four, Tanni 25/five. Tanni ripple
  availability remains 23/25; unavailable sessions are not zero-ripple controls.
- 14,441 eligible core windows across detector cohorts. MUA/ripple windows can
  overlap, so this is not a count of distinct independent biological events.
- Full RUN-QC populations plus the exact same three half-cell subsets used in
  the original real-cell-removal experiment. Cell IDs verified, not only seeds.
- K=5,000 per map-shuffle family; seed 20260905, eight workers and batch size
  eight, BLAS/OMP single-threaded. Longest session processed in 1,260.42 s.
- All technical/full-budget gates pass; all 462,112 decision rows verified.
  The audit independently recounted 493,198 raw 5 ms bins and checked 610,096
  original/sample-null geometric decisions. It recounted 246,640,000 saved
  null acceptance bits to reconstruct p-values and every acceptance decision.
- Independent dense likelihood checks covered 324,050 sampled MAP frames,
  with zero tied-bin disagreements. This is not independent rescoring of
  every MAP frame in all 5,000 shuffles. First three null paths per family
  are saved and independently geometrically checked.
- Original report is retained at `report/`. `report-v2/` adds the explicitly
  post-inspection descriptive paired order contrast and common within-dataset
  figure axes. No observations, decoder, threshold, bank or scoring output
  changes in this reporting update.
- All 188 relevant coverage/identifiability tests pass, including 17 targeted
  shuffle/report tests. Targeted Ruff and whitespace checks pass. The test
  suite includes missing controls, non-vacuous empty cohorts, raw timestamps,
  shuffled-path reconstruction, numerical boundaries and paired animal
  weighting; it is not validation of the biological null itself.

The initial K=40 runtime fixture exposed floating-point disagreement at exactly
40 cm displacement. The independent audit failed it. A declared 1e-9 cm boundary
tolerance and dense-MAP roundoff audit rule were committed before any K=5,000
production run. The repeated K=40 fixture passed; both timing artifacts are
retained, and neither provides primary significance evidence. See the numerical
addendum; do not silently attribute differences from older geometric rates to
shuffle testing alone.

## Transferred Test, Not Exact Author Reproduction

Independent flat-prior Poisson MAP decoding, 20 ms windows at 5 ms strides,
edge trimming to >=2 spikes, longest run with <20 cm adjacent jumps, >=10
frames and >=40 cm displacement. Primary significance requires both
cell-identity and per-cell spatial-map shuffle p-values below 0.02, using
`(successes+1)/(5000+1)` and the same geometric criterion on shuffled decodes.

The common encoding stays at 8 cm and uses the frozen RUN/unit QC. This differs
from PF's original 2 cm maps and other encoding/candidate choices. Spatial
shuffles are declared independent nonzero circular x/y map translations, not
an assumption that this matches unpublished author code exactly. The 11-frame
and internal >=2 cells/3 spikes filters are separate sensitivities. Geometric
failures stay in the denominator with uncomputed p-values explicitly missing.

## Original Observations

Equal-animal means: first average the three half subsets within a session,
then sessions within animals, then animals equally. CIs resample animals
5,000 times. They are limited by four/five animals and condition on the frozen
event cohorts, maps and shuffle banks.

| Dataset / detector | Eligible windows | Geometric full/half % | Two-shuffle full/half % | Half-minus-full pp [95% CI] | Animals negative |
|---|---:|---:|---:|---:|---:|
| PF MUA | 4,001 | 30.04 / 12.42 | 22.19 / 8.53 | -13.67 [-19.13, -8.02] | 4/4 |
| PF native ripple | 2,581 | 36.06 / 16.94 | 24.86 / 10.94 | -13.92 [-20.82, -8.70] | 4/4 |
| Tanni MUA | 5,224 | 10.55 / 2.63 | 5.77 / 1.70 | -4.07 [-4.42, -3.73] | 5/5 |
| Tanni LFP ripple | 2,635 | 4.76 / 1.07 | 2.21 / 0.64 | -1.57 [-2.43, -0.82] | 5/5 |

Thus the primary cell-removal effect is not confined to a geometric-only
criterion. The significance filter changes the yield, but not the sign of
the primary recording sensitivity in any animal.

Every cohort's mean half-minus-full interval remains negative across the
declared alpha 0.01/0.02/0.05, ten/eleven frames and internal-support options.
Animal uniformity is not universal across sensitivities: Tanni ripple,
edge-only/eleven frames at alpha 0.02 and 0.05, has four rather than five
negative animals. Changing the geometric criterion also changes its shuffle
reference distribution, so shuffle-significant event sets need not nest as
one geometric threshold becomes stricter. No sensitivity was promoted to
optimize acceptance.

## Order-Randomized Observations and Their Limits

One whole-5-ms-population-vector permutation per event is generated once and
reused for full/half decoding, with a genuinely partial last bin fixed. This
preserves population snapshots and total spikes, but not local overlapping
20 ms totals. It is not guaranteed to destroy every coincidental sequence or
to turn every event into a known biological non-replay.

| Dataset / detector | Randomized two-shuffle full/half % | Original-minus-randomized, full pp [95% CI] | Change in order excess under half cells, pp [95% CI] |
|---|---:|---:|---:|
| PF MUA | 2.44 / 1.40 | +19.75 [11.78, 27.73] | -12.62 [-18.23, -6.88] |
| PF native ripple | 3.39 / 2.09 | +21.47 [11.56, 31.39] | -12.62 [-20.24, -6.66] |
| Tanni MUA | 0.96 / 0.35 | +4.81 [3.96, 5.66] | -3.46 [-3.84, -3.09] |
| Tanni LFP ripple | 1.78 / 0.07 | +0.43 [-1.55, 2.17] | +0.14 [-1.47, 1.75] |

The paired contrasts were added descriptively after viewing the frozen
benchmark, without changing scoring or selection. They do not estimate
bias-corrected biological prevalence, and their animal-bootstrap intervals
do not integrate repeated observation-order permutations.

PF and Tanni MUA retain a clear original-order excess at the cohort level;
that excess decreases under cell removal in every animal. Tanni ripple does
not establish a full-population original-order excess or a decrease in that
excess. Therefore the nine-animal reduction in raw accepted fractions must
not be repackaged as nine-animal replication of a loss of order-specific
replay. Both are reported, rather than hiding the weak ripple control.

## Paper Implication and Next Work

This closes the transferred established-criterion benchmark. The recording
effect survives the two-shuffle filter, while control acceptance demonstrates
why successful continuity/significance screening is not a substitute for
recovery and specificity checks. The experiment does not measure true speed.

Direct precedent already exists for degrading a decoder or removing recorded
units before reassessing replay. The additional paper target must connect
this measurement sensitivity to speed-gradient recovery and inference limits;
see `replay_coverage_novelty_scope.md` and `replay_coverage_paper_prospect.md`.

Remaining required work is held-out-population calibration-transfer validation
and the integrated methods/results/limitations pack. No uniform-speed,
neural-mechanism or complete paper-ready claim follows from this benchmark.
