# Fresh-Path Clock Calibration Protocol

## Purpose And Fixed Scope

The preceding exhaustive calculation removes scoring-path Monte Carlo error
but retains 256-path generator libraries. Its estimates can remain biased
after pooling 6,400 observations. This experiment removes finite teacher-bank
reuse without changing the scorer, spatial maps, priors or recovery gates.

Use the same two fixed recordings, PF Rat1/Open1 and Tanni R2470
2018-08-10_18-11-37 (largest arena), from the audited exhaustive pilot.
Do not substitute the small Tanni arena, change encoding maps, drop stationary
or reset controls, or select an easier event class.

Every coherent simulated observation independently draws straight versus
curved with probability 0.5, then uniformly draws one admissible descriptor
within that family from the entire audited finite prior. Resets independently
draw a new path in every time bin, including the same 0.5 family weights.
Stationary observations draw one uniform supported location for the event.
Sampling is with replacement; accidental repeated paths are allowed.
No persistent small generator library exists. Scoring integrates all paths
without seeing generator labels or sampled descriptor indices.

Preserve source generator classes, real event bin-count profiles, duration,
population size and mixture assignments from the preceding run. Generate new
cell identities from conditional multinomial emissions. Compute the same
801-knot integrated rates under physical or population-code arc-length clocks.
This is a matched simulator-prior test, not robustness to neural model error.

Use two independent random streams, each with 50 repetitions of 128 events
at each true neural/(physical+neural) fraction 0.25, 0.50 and 0.75.
The source mixture probabilities are 0.6*(1-phi), 0.6*phi, 0.2, 0.1, 0.1
for physical, neural, stationary, physical-reset, neural-reset respectively.
Keep realized mixture-class assignments from the original schedule. Source
stream names do not refer to finite path libraries. Total: 76,800 observations.

The seed prefix is `fresh_path_clocks_v1|20260910`, followed by tag, repeat,
stream, event-in-population and scenario, separated by `|`, SHA256 first eight
bytes interpreted little-endian. The prior and the random sequence are frozen
before generation. Record sampled latent indices for audit only.

## Estimation And Decision

Fit all five mixture weights from counts only. Primary: the same 600 separate
128-event population fits. Preserve the existing thresholds: worst absolute
bias <=0.10, interval coverage >=0.90, directional power >=0.80, false direction
at true phi=0.50 <=0.05. Report Wilson simulation intervals and all conditions.
Passing code or optimizer checks is distinct from passing scientific recovery.

Predeclare 12 secondary pooled fits using all 50 repetitions (6,400 events
per recording, stream and fraction). These diagnose finite-sample bias and
precision; they cannot estimate coverage from one fit or add independent
animals. Report conditional likelihood-profile intervals without calling them
validated biological uncertainty. Do not replace the primary gates with them.

Compare with the preceding reused-library estimates on matching source
schedules. Counts and paths are newly simulated, so these are not identical
spike observations and the contrast is not a paired biological effect.

## Verification

Before interpreting estimates:

- Verify prior manifest and input/output hashes, all metadata and bin totals.
- Independently regenerate all fresh descriptor selections and count vectors
  with explicit barycentric spatial interpolation and merged-knot trapezoidal
  integration, rather than the producer's interpolator and antiderivative.
- Verify all source descriptors and final scored descriptors are identical
  to the audited exhaustive prior. Recompute first/middle/last start blocks
  per recording on the predeclared first two observations per condition.
- Check every weighted likelihood merge and all 600 primary plus 12 pooled
  optimum/profile certificates, then independently recompute summaries/gates.
- Retain corruption/missing-data tests, null generators and finite-count
  uncertainty. No real replay scoring until recovery is understood.

## Interpretation Limits

Improvement would identify a calibration problem with reusing a finite
generator library. It would not establish fixed physical speed in real replay,
new neural-sheet dynamics, a resolution-invariant continuity classifier, or
generality across animals. Two encoders and a restricted path family remain
only a diagnostic. Failure would require investigating model/observation
identifiability rather than silently changing the biological target.
