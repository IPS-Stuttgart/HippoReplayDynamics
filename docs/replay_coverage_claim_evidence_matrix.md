# Claim-by-Evidence and Completion Audit

Current checkpoint: independent RUN-half maps and shared-gain observation
benchmark completed and independently reconstructed, following the controlled
field-geometry and decoder-resolution factorial. This is
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
| Longer decoding windows are an unqualified improvement | In the fixed large-arena synthetic slice, 20 to 40 ms raises continuous recovery from 11.34% to 64.34%, but shuffled acceptance from 0.52% to 18.23% | Not supported; recovery and null acceptance must be evaluated jointly |
| Finer spatial grids restore lost speed gradients | 4/8/16 cm grids give nearly identical posterior-mean gradient responses in the paired resolution factorial | Not supported within the frozen range; more grid points do not supply missing information |
| A larger arena universally reduces continuity recovery | Native Poisson recovery declines with area, while the common-total generative contrast crosses zero; field-width and aspect effects depend on observation conditions | Not supported as a universal statement; common-total labels use known truth and are not a count-only intervention |
| Posterior credible mass provides a ready-made adaptive continuity threshold | Nominal 95% coverage is about 51% PF/61% Tanni for held-out 250 ms RUN positions | Not supported without calibration; RUN calibration need not transfer to replay |
| Known generator maps adequately represent independently fitted RUN-map recovery | PF full-cell simulated recovery falls from 35.82% to 19.60%; Tanni changes from 7.94% to 6.80%, while localization worsens in both | Known maps can be optimistic; recovery effect is not uniformly large across datasets |
| Independent maps restore speed-gradient identifiability | Full-cell known-coordinate response is 0.321 PF/0.121 Tanni for injected contrast 1.0; selected Tanni estimates available in only 1/50 half directions | Not supported; still requires calibrated recovery/abstention, not uniformity inference |
| Recording coverage explains the biological PF/Tanni difference | Different populations/maps/arenas and event definitions; no ground truth for rejected candidates | Not established |
| The coverage-causes-apparent-jumps hypothesis is new | Prior work explicitly states the hypothesis, including Ji et al. (2026) | Not novel by itself; target quantitative recovery/calibration rather than the generic caveat |

## Full Objective Requirements

Requirement numbering follows `replay_recording_coverage_study.md`, unchanged.

| Requirement | Evidence now | Remaining work / status |
|---|---|---|
| 1. Frozen provenance and inclusion | Clean scorer commits, input/cache/output hashes, snapshots, code archives, fixed sources and explicit smoke exclusions | Completed for runs so far; must also apply to remaining experiments |
| 2. Generator/likelihood validation | Fixed-total conditional audit; Poisson/conditional tests; independent RUN-half rate maps and shared-gain stress; reconstruction | Completed within these declared surrogate families; not a complete model of replay correlations |
| 3. Isolate all named recording and analysis factors | Real nested cell removal; pooled-cell controls; rate sweep; paired RatInABox count/width/area/aspect factorial with targeted 4/8/16 cm, 10/20/40 ms, 5/10 ms decoder sweep | Controlled factorial complete within the frozen ranges; not a general decomposition of biological dataset differences |
| 4. Estimators, support and selection | MAP/mean, unfiltered/filtered and selected-core metrics across the resolution sweep; independent no-gap-bridging and truth-eligibility recount; common-eligible path contrasts | Implemented for the frozen settings; no post-hoc optimal decoder promoted |
| 5. Known-path and null recovery | Positive/negative/zero gradients, stationary, discontinuous and shuffled controls on empirical maps and eight RatInABox populations; now independent half maps/shared gain too | Current benchmarks complete; independently validated calibration/abstention remains |
| 6. Both real recording populations | All candidates cached; real perturbations; empirical maps and source durations; simulated/source spike budgets reported | Complete population groundwork; controlled maps are surrogates, not replay ground truth |
| 7. Uncertainty and sensitivity | Equal-animal summaries; repeated synthetic draws; training-only RUN validation; independent half maps; explicit missingness | INCOMPLETE: real event-definition and broader decoder sensitivity; only nine animals; half-map drift is not pure estimation noise |
| 8. Meaningful equivalence where identifiable | Large injected gradients often poorly recovered; selected estimates frequently unavailable | INCOMPLETE: prespecified equivalence range, calibrated procedure and independent evaluation; no biological equivalence claim |
| 9. Paper-ready artifacts and novelty | Tested scripts, audited tables, inspected figures, protocols/results and this claim matrix; targeted literature checks | INCOMPLETE: integrated final methods/limitations pack and comparison to established calibration/detection methods |

## Next Required Decision

The counterfactual resolves the earlier restoration ambiguity, and the new
factorial separates field geometry from analysis resolution within a controlled
simulation. Repeating either with more identical draws is not the next priority.
Map-estimation/observation mismatch is now evaluated within a declared surrogate.
Next evaluate a frozen calibration or abstention procedure on independent
populations. Real event-definition
sensitivity and comparison with published baselines remain necessary. No
thresholds should be chosen to make the current real data appear uniform or the
simulations pass.

References: `replay_coverage_real_subsampling_results_20260905.md`,
`replay_coverage_run_validation_results.md`, `replay_coverage_recovery_results.md`,
`replay_coverage_counterfactual_results.md`,
`replay_coverage_geometry_results.md`, `replay_coverage_map_mismatch_results.md`,
`replay_coverage_novelty_scope.md`.
