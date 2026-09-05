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
- [x] Run a matched mechanistic coverage and speed-recovery factorial.
- [x] Audit and subsample real populations/candidates from both datasets.
- [x] Validate held-out RUN decoding with training-only unit/grid/map fitting.
- [x] Run and reconstruct an empirical-map development recovery benchmark.
- [x] Replicate with data-only pooled-cell controls and paired information doses.
- [x] Test independent RUN-half maps and shared-gain observation mismatch.
- [x] Evaluate frozen speed intervals on new A/B simulated panels and report abstention/transfer.
- [x] Freeze and audit native/LFP-ripple versus MUA inputs, with core/fixed windows and explicit missingness.
- [ ] Inspect LFP/MUA examples and run detector/window paired coverage-effect decoding.
- [ ] Compare with established shuffle-significant replay baselines.
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

## Completed Empirical-Map Development Recovery

See `replay_coverage_recovery_protocol.md` and
`replay_coverage_recovery_results.md`. The frozen 33-session run generated
4,855 known paths using 971 source support profiles; all 349,560 metric rows
passed technical checks, and path/count reconstruction plus direct support
recount verified every row. No source event was selected on decoded success.
Continuity loss under cell removal largely disappears when total counts are
restored using true-position-dependent labels and decoded conditionally. That
intervention can introduce information and does not isolate count loss.
The later data-only pooled control below retains most continuity loss despite
preserving all spikes. Strong injected spatial gradients are
substantially attenuated, particularly in Tanni; selected cores are too sparse
in most Tanni sessions for gradient inference. These are development results,
not independent calibration or biological uniformity. The full factor-isolation
and independent recovery/equivalence requirements above remain incomplete.

## Completed Paired Information-Loss Counterfactual

See `replay_coverage_counterfactual_protocol.md`,
`replay_coverage_information_controls.md`, and
`replay_coverage_counterfactual_results.md`. The 33-session production run uses
three new randomizations and nested exposures, with 17,478 truth-trial records
and 1,957,536 metric rows. Full reconstruction and independent support recount
passed. Pooling removed cells preserves every spike but retains most continuity
loss, correcting a count-only interpretation of oracle restoration. Strong
spatial gradients remain attenuated at roughly source-comparable spike budgets;
much higher exposures partly recover them. These are independent synthetic
draws on the same empirical maps, not held-out-map calibration or biological
replication. The full completion requirements remain active.

## Completed Field-Geometry and Resolution Factorial

See `replay_coverage_geometry_protocol.md` and
`replay_coverage_geometry_results.md`. The frozen RatInABox Gaussian-field
experiment crosses population count, field width, arena area and aspect, with
a targeted spatial-grid/window/stride factorial on identical fine spikes.
All 96 batches completed; all 55,296 observation arrays reconstructed and all
1,400,832 metric rows passed independent support/truth-eligibility checks.
An additional 9,728 rows passed analytic likelihood/position-error checks.
The eight synthetic populations are not biological replication.

Longer windows substantially increase known-path recovery but also null
acceptance. In the illustrated 8.75 m2 slice, 20 to 40 ms changes recovery
from 11.34% to 64.34% and shuffled-path acceptance from 0.52% to 18.23%.
The recovery increase persists on a common truth-eligible path set. Finer
spatial grids barely change gradient recovery; window averaging itself removes
kinematic detail. Field-width/area effects depend on observation conditions.
No setting is promoted as optimal, and no biological uniformity claim follows.
The map/observation mismatch follow-up is recorded below. Independent
calibration, real event-definition sensitivity and established-method
comparisons remain necessary.

## Completed Independent RUN-Map / Shared-Gain Benchmark

See `replay_coverage_map_mismatch_protocol.md` and
`replay_coverage_map_mismatch_results.md`. All 66 directions across 33 sessions
scored 745,728 metric rows. Both decoder maps use the same observations, selected
cells and state support; only the fitted rate values change. The generator uses
the other chronological RUN half, and both directions are evaluated. The audit
reconstructs half maps and observations, checks every row's support/eligibility,
and independently checks 25,344 sampled decoding rows.

Full-cell constant-speed MAP recovery changes from 35.82% to 19.60% for PF and
7.94% to 6.80% for Tanni under independent rather than generator-known rates.
The PF decrease is rat-uniform; the smaller Tanni contrast is not. Position
error and posterior-inclusion loss worsen under independent rates in both.
Known-coordinate speed-gradient response remains attenuated (0.321 PF, 0.121
Tanni for an injected contrast of 1.0). Selected-only Tanni gradients remain
too sparsely available for cohort-wide interpretation. Shared gain is a
declared stress model, not fitted replay covariance. This completes the planned
map-estimation sensitivity within this surrogate, not calibration of real
replay or the remaining event-definition/baseline/equivalence requirements.

## Completed Speed-Interval Calibration and Transfer

See `replay_speed_identifiability_protocol.md` and
`replay_speed_identifiability_results.md`. All 33 sessions completed 247,896
readout rows and 633,600 test interval decisions; all intervals reproduce,
all training maps refit exactly and 6,864 sampled panel rows reconstruct.
The frozen inverse Gaussian/conformal and raw-bootstrap baselines use disjoint
fit/calibration/test simulation draws. B maps and shared gain never enter fitting.

Matching-simulation conformal coverage is near nominal before continuity
selection, but combined map/gain stress lowers coverage to 76.75% PF/89.40%
Tanni in the bin-supported readout. Primary selected-event conformal results
are finite only in Rat1; all Tanni primary calibrated results abstain. Neither
calibrated method makes a +/-0.25 equivalence claim anywhere in the experiment.
The result is an inference limit at the declared panel budget and settings,
not biological uniformity or a universally calibrated method. Real
event-definition sensitivity, published replay-detection comparisons and
new-population validation remain distinct uncompleted requirements.

## Completed Likelihood Audit

See `replay_recording_coverage_audit_20260905.md` for results and exact artifact
locations. All 32,400 legacy events reproduced; the corrected 30 cm field-width
effect is smaller and its interval crosses zero. The narrower-field effect
persists. This is a mixed result, not a universal coverage explanation.

## Prepared Real Event-Definition Sensitivity Inputs

See `replay_coverage_event_definition_protocol.md` and
`replay_coverage_event_definition_results.md`. All 12,141 source MUA events remain
in the frozen 33-session artifact, with 90,132 core/fixed-window rows including
native PF and newly detected Tanni ripple-like sources and explicit exclusions.
The audit verified all source identities, window spike support and 5,665 overlap
edges, rehashed 150 consumed native arrays and refiltered one Tanni recording.

PF native ripple definition is available in all eight sessions; new Tanni LFP
detection is available in 23/25 sessions. Two sessions below the frozen 60 s
immobile-baseline requirement are labeled unavailable, not zero-ripple. Their
MUA windows remain available, and all nine animals have both-detector sessions.
MUA/ripple overlaps differ strongly under these definitions; those are candidate
assay differences, not replay precision/recall or biological differences.

No new replay decoding occurred in preparation. Detector/window cell-removal
effects, representative trace inspection and published replay baselines remain
uncompleted requirements. Use all-intermediate-bin support for the new speed
comparison, not the endpoint-only speed check in the older subsampling helper.
