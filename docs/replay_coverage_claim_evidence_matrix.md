# Claim-by-Evidence and Completion Audit

Current checkpoint: the transferred PF-style two-shuffle benchmark is completed
and independently audited across all33, after detector/window coverage decoding,
trace inspection, speed-interval calibration, independent RUN-half maps and the
geometry/resolution factorial. This is
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
| Matching-simulation interval calibration guarantees transfer | Bin-supported all-data conformal coverage about 95% when matching, 76.75% PF/89.40% Tanni with disjoint maps plus shared gain | Not supported; all nine animals show lower combined-stress coverage |
| High all-panel coverage establishes informative speed inference | Primary selected Tanni calibration has 100% coverage but 0% finite intervals; PF finite conformal results only in Rat1 | Not supported; report abstention, finite-only coverage, width and animal availability |
| Current calibrated intervals establish meaningful uniform speed | No inverse Gaussian/conformal +/-0.25 equivalence claims in any tested condition, including constant-speed truth | Not supported at the declared panel budget and settings; not an impossibility theorem or biological null |
| Recording coverage explains the biological PF/Tanni difference | Different populations/maps/arenas and event definitions; no ground truth for rejected candidates | Not established |
| The coverage-causes-apparent-jumps hypothesis is new | Prior work explicitly states the hypothesis, including Ji et al. (2026) | Not novel by itself; target quantitative recovery/calibration rather than the generic caveat |
| The primary real cell-removal continuity result survives ripple versus MUA event definition | 28,451 eligible core/fixed windows, 910,432 rows, all metrics reconstructed; negative primary half-cell effect in all nine animals for both definitions and windows | Supported for the frozen geometric criterion; not shuffle-significant replay, and not every likelihood/peak sensitivity interval excludes zero |
| Candidate-event speed changes under cell removal have a universal sign | Same-supported-step PF shifts about +143 to +146 cm/s across definitions; Tanni MUA positive, ripple core/fixed -93/+59 with intervals crossing zero | Not established; Tanni ripple estimates uncertain, strongly reduced measurability and no real latent-speed truth |
| Cell removal changes acceptance under a transferred published-budget two-shuffle criterion | 14,441 eligible core windows, 33 sessions; K=5,000 per family; all 462,112 decision rows reconstructed. Primary MUA full/half acceptance PF 22.19%/8.53%, Tanni 5.77%/1.70%; negative in all nine animals, also for ripple cohorts | Supported at common 8 cm encoding and declared shuffle operators; not an exact author-pipeline reproduction or latent replay truth |
| Both map-shuffle tests guarantee an accepted event contains intact original time order | Order-randomized primary acceptance PF MUA 2.44% and native ripple 3.39%; Tanni MUA 0.96% and LFP ripple 1.78%, versus original 5.77%/2.21% for Tanni | Not established; one order surrogate/event, with population snapshots preserved. These are control acceptance rates, not known biological false-positive rates |
| Decreased acceptance under cell removal is always decreased original-order-specific acceptance | Paired original-minus-randomized excess decreases in all PF and Tanni MUA animals; Tanni ripple change +0.14 pp, CI [-1.47, 1.75], with no clear full-population order excess | Not established for every event class; do not equate the primary nine-animal acceptance result with universal loss of order-specific replay |
| Real-cell removal as a replay-quality control is itself new | Silva et al. (2015) degrade decoders; Liu et al. (2023) remove units and recompute replay | Not novel by itself; additional contribution must connect selection, spatial-speed recovery and limits of inference |

## Full Objective Requirements

Requirement numbering follows `replay_recording_coverage_study.md`, unchanged.

| Requirement | Evidence now | Remaining work / status |
|---|---|---|
| 1. Frozen provenance and inclusion | Clean scorer commits, input/cache/output hashes, snapshots, code archives, fixed sources and explicit smoke exclusions | Completed for runs so far; must also apply to remaining experiments |
| 2. Generator/likelihood validation | Fixed-total conditional audit; Poisson/conditional tests; independent RUN-half rate maps and shared-gain stress; reconstruction | Completed within these declared surrogate families; not a complete model of replay correlations |
| 3. Isolate all named recording and analysis factors | Real nested cell removal; pooled-cell controls; rate sweep; paired RatInABox count/width/area/aspect factorial with targeted 4/8/16 cm, 10/20/40 ms, 5/10 ms decoder sweep | Controlled factorial complete within the frozen ranges; not a general decomposition of biological dataset differences |
| 4. Estimators, support and selection | MAP/mean, unfiltered/filtered and selected-core metrics across the resolution sweep; independent no-gap-bridging and truth-eligibility recount; common-eligible path contrasts | Implemented for the frozen settings; no post-hoc optimal decoder promoted |
| 5. Known-path and null recovery | Positive/negative/zero gradients, stationary, discontinuous and shuffled controls; independent maps/gain; new fit/calibration/test panel splits | Current benchmarks complete, including conditional surrogate calibration/abstention; new-population transfer not established |
| 6. Both real recording populations | All candidates cached; real perturbations; empirical maps and source durations; simulated/source spike budgets reported | Complete population groundwork; controlled maps are surrogates, not replay ground truth |
| 7. Uncertainty and sensitivity | Equal-animal summaries; training-only RUN validation; independent half maps; completed detector/core/fixed-window paired decoding with explicit two-session LFP unavailability; 15 preselected traces inspected | Completed within the frozen sensitivity ranges; only nine animals, half-map drift is not pure estimation noise, traces do not prove detector specificity or hardware synchronization |
| 8. Meaningful equivalence where identifiable | Frozen +/-0.25 band and 0.10/0.50 sensitivities; Gaussian/conformal/raw comparison on independent panels; 633,600 decisions verified; no calibrated primary-band claims | Surrogate evaluation completed with an identifiability limit; biological equivalence and transferable calibrated procedure NOT established |
| 9. Paper-ready artifacts and novelty | Tested scripts, audited tables, inspected figures, protocols/results and this claim matrix; transferred PF two-shuffle and standard calibration baselines; targeted direct-prior checks and a paper prospect | INCOMPLETE: held-out-population validation and integrated final methods/limitations pack; no blanket novelty certification |

## Next Required Decision

The counterfactual resolves the earlier restoration ambiguity, and the new
factorial separates field geometry from analysis resolution within a controlled
simulation. Repeating either with more identical draws is not the next priority.
Map-estimation/observation mismatch is now evaluated within a declared surrogate.
The frozen calibration/abstention comparison is now complete on new simulation
panels, with limited availability and failed transfer; do not hide that by
retuning to test outcomes. New biological populations were not held out.
Real detector/window paired decoding and selected trace inspection are now
complete with explicit missingness. The PF-style published-budget two-shuffle
comparison is now complete; the real-cell effect remains, but order-randomized
acceptance limits interpretations of the low Tanni ripple yield. New-population
validation and the integrated methods/limitations pack are next. An exact
author-encoding reproduction or head-to-head with other sequence detectors
would be additional, different comparisons. No
thresholds should be chosen to make the current real data appear uniform or the
simulations pass.

References: `replay_coverage_real_subsampling_results_20260905.md`,
`replay_coverage_run_validation_results.md`, `replay_coverage_recovery_results.md`,
`replay_coverage_counterfactual_results.md`,
`replay_coverage_geometry_results.md`, `replay_coverage_map_mismatch_results.md`,
`replay_coverage_novelty_scope.md`, `replay_speed_identifiability_results.md`,
`replay_coverage_event_definition_results.md`,
`replay_coverage_detector_decoding_results.md`,
`replay_coverage_shuffle_baseline_results.md`, `replay_coverage_paper_prospect.md`.
