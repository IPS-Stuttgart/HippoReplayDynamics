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
- [ ] Reproduce old inputs with conditional and unconditional likelihoods.
- [ ] Run a matched mechanistic coverage and speed-recovery factorial.
- [ ] Audit and subsample real populations/candidates from both datasets.
- [ ] Validate calibration and gradient identifiability on held-out populations.
- [ ] Finish the paper-facing evidence and limitations pack.

The first reanalysis is a diagnosis of an existing simulation, not completion
of the full study.
