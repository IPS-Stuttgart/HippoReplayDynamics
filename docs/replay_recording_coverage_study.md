# Recording Coverage, Replay Speed, and Continuity

Status: active investigation; no biological uniformity claim.

## Question

How much can finite recording coverage distort decoded replay speed and the
probability that a genuine trajectory passes a fixed continuity criterion?
Can calibrated inference distinguish uniform physical speed from meaningful
spatial speed modulation in Pfeiffer/Foster and Tanni recordings?

## Completion Requirements

1. Freeze and hash source code, simulation settings, real-data inputs, animal
   and session inclusion, and event definitions. Keep an audit of exclusions.
2. Validate the observation model against its generator. Fixed-total count
   simulations require a conditional multinomial decoder; unconditional
   Poisson decoding is a separately labeled misspecification control.
3. Isolate cell count, field width/density, arena area/aspect, spike support,
   temporal bin/stride, and spatial discretization. Use paired latent paths
   and population replicates, with latent-path eligibility checked separately.
4. Evaluate MAP and posterior-mean speeds before selection, after bin-support
   filtering, and within the selected continuous run. Do not bridge missing
   bins or treat overlapping steps as independent experimental replicates.
5. Verify simulation recovery using uniform speed AND positive/negative spatial
   gradients, along with stationary, discontinuous, and shuffled controls.
   Report false retention and false rejection, spatial distortion, and recovery
   uncertainty. A continuity heuristic alone is not a replay significance test.
6. Calibrate against actual recording populations and event support from BOTH
   datasets without selecting only already-continuous events. Test controlled
   cell subsampling of identical observed events. Synthetic results do not
   identify the unobserved replay prevalence of rejected candidates.
7. Carry animal/session uncertainty and sensitivity to event definition, decoder
   settings, and rate-map estimation into the real-data conclusions. Avoid
   interpreting a simulation-population bootstrap as animal replication.
8. Evaluate a prespecified meaningful speed-gradient equivalence range only
   where injected gradients are recoverable. Failure to reject zero is not
   equivalence. Failed recovery leads to an identifiability limit, not uniformity.
9. Produce auditable tables, inspected figures, executable tests, a methods and
   limitations document, and a claim-by-evidence matrix. Distinguish findings
   established here from published prior work; do not tune toward a positive
   biological or methodological result.

## Initial Artifact Audit (2026-09-05)

Existing artifacts are on gpuserver6000 in
`/home/florianpfaff/HippoReplayIMM-replay-geometry-hypotheses/results/`:

- `ratinabox-place-field-coverage-continuity-full-v1/`
- `matched-place-field-coverage-continuity-full-v1/`

The original scripts are untracked in that working copy and are not modified by
this study. The existing matched-count generator samples cell identities from
normalized rate vectors with externally imposed population totals. The decoder
instead includes the unconditional Poisson population-rate term. Consequently,
the primary matched-count experiment is not likelihood-matched and needs an
explicit conditional-likelihood reanalysis.

The existing Poisson sensitivity is not affected by that specific fixed-count
mismatch. Its effect changes with field width; broad Gaussian fields can reverse
the arena-area difference. The thresholded-field stress test is not an empirical
estimate. The old arena contrast also changes aspect ratio, and its learned-map
benchmark uses support profiles from already-selected continuous events.

These observations motivate new controlled tests; they do not establish that
coverage explains the observed difference in trajectory-event rates.

## Staged Work

- [x] Locate the actual completed RatInABox runs and audit the generator/decoder.
- [x] Reproduce old inputs with conditional and unconditional likelihoods.
- [ ] Run a matched mechanistic coverage and speed-recovery factorial.
- [x] Audit and subsample real populations/candidates from both datasets.
- [x] Validate held-out RUN decoding with training-only unit/grid/map fitting.
- [ ] Validate calibration and gradient identifiability on held-out populations.
- [ ] Finish the paper-facing evidence and limitations pack.

The first reanalysis is a diagnosis of an existing simulation, not completion
of the full study.

## Completed Real-Population Perturbation

See `replay_coverage_real_input_protocol.md`,
`replay_coverage_subsampling_protocol.md`, and
`replay_coverage_real_subsampling_results_20260905.md`.
All 12,141 candidates across 33 sessions were retained and cached. The paired
population-subsampling experiment passed technical gates; continuity loss with
half the cells is negative in every animal and remains without a per-bin spike
support filter. Speed readouts change with sampling and selection, not uniformly
across datasets. These observations establish real-data measurement sensitivity,
not ground-truth bias or biological uniformity. Actual-map recovery remains
necessary; the subsequent training-only RUN validation is recorded below.

## Completed Training-Only RUN Validation

See `replay_coverage_run_validation_protocol.md` and
`replay_coverage_run_validation_results.md`. All 165 folds across 33 sessions
completed; the independent audit recounted observations and verified all
165,000 prediction rows. Every Poisson session beats the wrong-cell-map control
descriptively. However, nominal 95% regions cover tracked position only about
51% (PF) and 61% (Tanni) in 250 ms windows. Good localization does not establish
calibrated uncertainty. Short RUN bins have much lower spike support and cannot
be treated as replay error ground truth. The next simulations must explicitly
separate oracle decoding from observation/map mismatch and verify uncertainty
and speed-gradient recovery before any adaptive continuity or equivalence claim.

See `replay_coverage_novelty_scope.md` for direct prior work on calibration,
sampling bias, and replay validation. The paper contribution must go beyond
those established general observations.

## Completed Likelihood Audit

See `replay_recording_coverage_audit_20260905.md` for results and exact artifact
locations. All 32,400 legacy events reproduced; the corrected 30 cm field-width
effect is smaller and its interval crosses zero. The narrower-field effect
persists. This is a mixed result, not a universal coverage explanation.
