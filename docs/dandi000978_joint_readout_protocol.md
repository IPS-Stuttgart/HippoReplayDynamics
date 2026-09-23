# DANDI000978 joint-readout calibration and finite-group sensitivity

Frozen 2026-09-23 before computing the joint diagnostic. Seed 20260924.
Inputs are the completed sparse-count RUN calibration and the unchanged frozen
sleep pilot. This is conditional measurement calibration, not a replay rescore.

## Question

Separately above-chance CA1/PFC RUN readouts do not establish that PFC can validate
an uncertain CA1 route target in a group of three or eleven sleep events. Test
that exact scalar readout against a known shared RUN-route positive control.

## Bank and Eligibility

Reuse archived training-only, whole-epoch-held-out RUN maps, raw-verified count
vectors and ten fixed-seed thinning repetitions. CA1 and PFC anchors were sampled
independently within the same known directed route: they are not simultaneous
windows or necessarily the same trial/location. Their shared route is known
behavior, not a claim of coordination, abstract identity or sleep ground truth.

For each sleep-count target, repetition and scope, form a four-route block.
Require exact count matching in both regions for all four routes in a block.
No clipping, upsampling, replacement search or outcome-based exclusion. Report
block coverage and target coverage, and preserve unavailable primary targets.
Native-count and sleep-count comparisons use the same eligible anchor blocks.
250 ms windows are primary calibration; whole trials are an optimistic control.

CA1 reference = argmax of its count-conditioned four-route posterior, as in the
sleep pilot. PFC scalar = per-spike score for that CA1 reference minus the mean
score over four routes. Also evaluate the known route as an oracle reference.
The oracle is explicitly a positive control, never a sleep target estimator.

## Conditional Sensitivity, Not Biological Inference

For each animal, reference, scope and native/sleep-count regime:

- The original primary design contains exactly its frozen lost-label targets
  (JS14 three, ZT2 eleven), not a replacement selected from convenient counts.
  If any target lacks an eligible block, report unsupported; do not simulate a
  smaller group and call it complete.
- Additional design sizes are 10, 25 and 50 unique supported count targets,
  sampled without replacement. These are sensitivity curves, not authorization
  to scale replay scoring or an optimization of a biological threshold.
- Generate 500 virtual studies. Within each target sample one eligible block
  and a uniform known route. The positive control pairs PFC from that route
  with CA1; the null pairs a uniform PFC donor route, including the same route
  with probability 1/4. Counts are identical across donor routes.
- For each virtual study draw 199 analogous randomized pairings and compute
  one-sided (1 + number null >= observed)/(1 + 199). Apply alpha 0.05 unchanged.
- Report positive-control rejection probability and null false-positive rate.
  Monte Carlo error is simulation precision only, not animal-level uncertainty.
  Repetitions, targets and virtual studies are not new biological replicates.

Equal-route mixtures, particular held-out RUN templates and independent donors
are assumptions, not known sleep generators. Sparse readouts and mismatched
locations can limit this control. Strong joint RUN performance still cannot
validate RUN-to-NREM transfer. Weak performance is not absence of sleep content.
No new replay threshold, replay event set, or biological claim is introduced.

## Decision Boundary

This diagnostic is technically complete only when all expected banks are checked
and every requested design has an explicit supported/unsupported result. Record
whether the exact primary group is representable and whether its positive-control
rejection probability reaches 0.80 in both animals while null rejection stays at
or below 0.075. Those are planning diagnostics, not new replay pass criteria.
Even a pass leaves `sleep_content_validated = false` and `paper_ready = false`.

## Literature and Cohort Boundary

Shin et al., eLife 2026 (doi:10.7554/eLife.110795.3), already report sleep-state
differences in hippocampal-prefrontal activity and assembly reactivation using
DANDI000978. The accompanying public REMHFOs code (commit
3b4245accfa27af9ca9be6bf2538791fd948ab96) reads precomputed regional MAT products;
the inspected repository does not contain a per-unit NWB regional crosswalk.
It does not resolve the six remaining animals' anatomical ambiguity. Mere
cross-area reactivation or a REM/NREM contrast is not our novel contribution.
