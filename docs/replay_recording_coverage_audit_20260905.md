# Coverage Calibration: First Audit

Date: 2026-09-05. Status: progress, full study still active.

## Authoritative Run

- Server: `gpuserver6000` (hostname `workstation2`).
- Worktree: `/home/florianpfaff/HippoReplayDynamics-recording-coverage`.
- Branch: `investigate-replay-recording-coverage`.
- Scoring code commit: `9a1cb7a09e50802db2d3dbf2bf2cbbeae5d9f995`.
- Run: `/mnt/seagate10tb/florianpfaff/replay-coverage-likelihood-audit-full-20260905`.
- Manifest reports a clean Git tree and hashes both legacy simulation scripts,
  the reference CSV/manifest, and the new decoder/audit code.
- RatInABox 1.15.3; NumPy 2.4.6; SciPy 1.17.1; pandas 3.0.5.
- Validation: 18 unit tests, Ruff, and staged whitespace check passed.
- All 32,400 original event rows reproduced: identical spike totals and
  continuity decisions, numerical errors within 2.9e-14 cm.
- The same spikes were evaluated with two likelihoods, two point estimators,
  and two support filters: 259,200 event-metric rows, not independent events.

## Why Conditioning Matters

The original primary generator fixes each population count N and samples cell
identities with probabilities p_i(x) = lambda_i(x) / sum_j lambda_j(x).

For that experiment, the corresponding fixed-position likelihood is

    log L_conditional(x) = sum_i n_i log p_i(x).

The previously used unconditional Poisson likelihood is

    log L_Poisson(x) = sum_i n_i log lambda_i(x) - dt * sum_i lambda_i(x).

The latter uses spatial variation in the population firing rate even though
the matched-count generator imposed totals independently. Its extra information
is therefore misspecified in that generator. This diagnosis does NOT establish
that Poisson decoding of the actual recordings is wrong. Nor does it remove
the shared approximation of representing a moving 20 ms window by one position.

## Main Numerical Result

Gaussian fields, 120 cells, fixed population counts, MAP continuity screening.
Negative differences indicate fewer simulated trajectories retained in the
larger arena. Intervals bootstrap the 20 paired synthetic populations.

| Field SD | Original large-minus-small pass | Count-conditioned pass | Corrected 95% CI |
|---|---:|---:|---:|
| 20 cm | -7.67 percentage points | -5.58 percentage points | [-8.08, -3.17] pp |
| 30 cm | -7.17 percentage points | -1.83 percentage points | [-5.33, +1.58] pp |
| 40 cm | +2.42 percentage points | +0.83 percentage points | [-2.83, +4.33] pp |

At 30 cm the paired likelihood-by-arena interaction is +5.33 percentage points,
95% CI [+1.58, +9.25]. Thus the likelihood choice accounts for much of the
previously reported arena difference in this particular synthetic condition.

The independent Poisson-generation sensitivity remains a separate control:
its correctly specified Poisson decoder gives large-minus-small pass differences
of -8.17, -1.75, and +3.50 percentage points at field SDs 20, 30, and 40 cm.
The last has CI [+0.17, +6.58] pp. The coverage effect is not monotonic or uniform
across these field assumptions. Hard-thresholded fields give larger effects but
remain a stress test, not a fitted model of the real populations.

## Speed Readout Warning

For the fixed-count 30 cm condition, true speed was 1000 cm/s. The population
mean of event-median MAP step speeds was around 1600 cm/s in both arena sizes.
This is an audit readout from overlapping 20 ms windows advanced by 5 ms on an
8 cm grid: an 8 cm MAP step divided by 5 ms is 1600 cm/s. It is NOT the original
non-overlapping speed estimand and is not evidence about biological speed.
The full study must retain the distinction between selection stride and speed
measurement stride and compare to the integrated ground-truth path.

## Learned-Map Calibration Audit

The current PF script `analyze_pfeiffer_foster_mmse_speed_injections.py` uses
templates from already-continuous events and can redraw paths/spikes until a
decoded trajectory passes, retaining the largest-displacement draw otherwise.
The saved `pfeiffer-foster-mmse-speed-injections-high-mua-full-base800-stochastic-power-v1`
manifest specifies `maximum_path_draws=20`, `matched_mua_profile`, and 1264 source
events, all already Foster-continuous. It records a dirty tree but does not hash
the running script, so exact historical source identity is not guaranteed.
These outputs cannot be treated as an unbiased all-candidate recovery estimate.

The current Tanni registration injection script differs: its redraw criterion
uses TRUE path displacement before spikes/decoding. That is a conditional latent
eligibility design and must not be conflated with decoded-outcome selection.
Both approaches need their conditioning and rejected draws made explicit.

The inspected PF stochastic-power reporter already correctly abstains:
`technical_inconclusive_injection_recovery_failed`. This study does not promote
that result to uniformity or a biological near-wall speed effect.

## Next Required Experiment

Build a frozen one-draw-per-trial calibration from both datasets' ALL candidate
windows and their actual encoding populations. Record every failure. Separate
native Poisson generation, count-conditioned generation, and observation-model
misspecification. Never redraw based on decoded content. Select geometric
eligibility using truth only as a separately labeled analysis denominator.

Test speed gradients in both directions and constant speed, before and after
the exact continuity/support gates; test cell subsampling and density rescue.
Reserve independent population/path seeds for recovery assessment. Include
stationary and discontinuous negative controls. Track true arclength, sampled
position chords, and window-averaged paths to expose temporal integration bias
near turns/walls. Infer biological uniformity only if meaningful nonuniformity
is demonstrably recoverable and the uncertainty lies inside the registered
equivalence region.

## Current Claim Boundary

Supported: under these simulations, fixed-count likelihood misspecification can
substantially change an apparent recording-coverage effect on continuity.
Narrow fields still show a genuine decoder/selection penalty in the tested
larger arena, conditional on the simulation assumptions.

Not established: that coverage explains the real Tanni/PF retention difference,
that rejected real candidates are continuous replay, that physical replay speed
is spatially uniform, or that this audit by itself is a paper-ready conclusion.
