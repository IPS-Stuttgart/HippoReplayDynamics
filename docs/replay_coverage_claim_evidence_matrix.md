# Claim-by-Evidence and Completion Audit

Current checkpoint: paired information-loss counterfactual completed. This is
not a final paper-ready certification. Scientific computations used
gpuserver6000; artifact paths and commit hashes are in the linked results notes.

## Claims

| Claim | Current evidence | Decision |
|---|---|---|
| Recorded-population size changes continuity classifications of identical real candidates | 12,141 source windows, 33 sessions/nine animals; negative half-population effect in every animal, with/without bin-support filtering | Supported for this fixed detector/decoder/criterion; latent replay truth unknown |
| Total spike loss alone explains the population-removal effect | Oracle restoration nearly rescues recovery, but pooled removed spikes retain most loss | Not supported; oracle restoration can introduce spatial information |
| Individual spatially tuned cell identities matter beyond aggregate counts | Pooled-half recovery 10.50% vs full 29.30% PF; 2.98% vs 7.78% Tanni, with all fine-bin totals preserved | Supported in the empirical-map surrogate; not a complete geometric-coverage decomposition |
| Flat decoded spatial-speed profiles establish uniform true speed | Large imposed gradients yield attenuated responses; primary PF 0.351 and Tanni 0.107 for injected contrast 1.0 | Not supported by this pipeline's current recovery |
| Higher information can improve recovery of known speed gradients | Paired exposure sweep approaches window-mean truth at much higher spike rates | Supported as a stress test, not physiological rate inference or a calibrated correction |
| Posterior credible mass provides a ready-made adaptive continuity threshold | Nominal 95% coverage is about 51% PF/61% Tanni for held-out 250 ms RUN positions | Not supported without calibration; RUN calibration need not transfer to replay |
| Recording coverage explains the biological PF/Tanni difference | Different populations/maps/arenas and event definitions; no ground truth for rejected candidates | Not established |
| The coverage-causes-apparent-jumps hypothesis is new | Prior work explicitly states the hypothesis, including Ji et al. (2026) | Not novel by itself; target quantitative recovery/calibration rather than the generic caveat |

## Full Objective Requirements

Requirement numbering follows `replay_recording_coverage_study.md`, unchanged.

| Requirement | Evidence now | Remaining work / status |
|---|---|---|
| 1. Frozen provenance and inclusion | Clean scorer commits, input/cache/output hashes, snapshots, code archives, fixed sources and explicit smoke exclusions | Completed for runs so far; must also apply to remaining experiments |
| 2. Generator/likelihood validation | Fixed-total conditional-likelihood audit; Poisson/conditional counterfactual tests; exact reconstruction | Observation-family checks complete for these settings; map/observation mismatch validation remains |
| 3. Isolate all named recording and analysis factors | Real nested cell removal; pooled-cell controls; rate sweep; legacy field-width sensitivities | INCOMPLETE: controlled density/width, area/aspect, temporal bin/stride and spatial-grid factorial |
| 4. Estimators, support and selection | MAP/mean, unfiltered/filtered and selected-core metrics; direct no-gap-bridging recount | Implemented for current 20 ms/5 ms/8 cm settings; extend consistently to remaining settings |
| 5. Known-path and null recovery | Positive/negative/zero gradients, stationary, discontinuous and whole-bin-shuffled controls; three new randomizations | Current benchmark complete; not independent held-out-population calibration or false-rejection validation across all factors |
| 6. Both real recording populations | All candidates cached; real perturbations; empirical maps and source durations; simulated/source spike budgets reported | Complete population groundwork; controlled maps are surrogates, not replay ground truth |
| 7. Uncertainty and sensitivity | Equal-animal summaries; repeated synthetic draws; training-only RUN validation; explicit missingness | INCOMPLETE: event-definition, map-estimation and broader decoder sensitivities; only nine animals |
| 8. Meaningful equivalence where identifiable | Large injected gradients often poorly recovered; selected estimates frequently unavailable | INCOMPLETE: prespecified equivalence range, calibrated procedure and independent evaluation; no biological equivalence claim |
| 9. Paper-ready artifacts and novelty | Tested scripts, audited tables, inspected figures, protocols/results and this claim matrix; targeted literature checks | INCOMPLETE: integrated final methods/limitations pack and comparison to established calibration/detection methods |

## Next Required Decision

The new counterfactual resolves the earlier restoration ambiguity; repeating it
with more identical source draws is not the next priority. The next experiments
must separate spatial field geometry and temporal/grid approximation from
information quantity, then evaluate map/observation mismatch and a proposed
calibration on independent draws/populations. No thresholds should be chosen
to make the current real data appear uniform or the simulations pass.

References: `replay_coverage_real_subsampling_results_20260905.md`,
`replay_coverage_run_validation_results.md`, `replay_coverage_recovery_results.md`,
`replay_coverage_counterfactual_results.md`, `replay_coverage_novelty_scope.md`.
