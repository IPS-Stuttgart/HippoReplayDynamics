# Unknown-path population clock recovery

Frozen before production scoring. This is a simulation recovery study, not a
reanalysis of real replay events, an equivalence test, or a fuzzy-continuity test.
Question: can a population mixture distinguish literal constant physical-arc
speed from constant conditional-population-code-arc speed without supplied paths?

## Source and geometry

Use all 33 hash-verified native 20-ms source caches from
`conditional-2d-mua-pf-tanni-all33-20260908`: 8 Pfeiffer/Foster recordings (4 rats)
and 25 Tanni recordings (5 animals). Only total spikes per full native time bin
are retained; fresh cell identities are simulated. Use profiles with at least
3 full bins. No continuity selection or model evidence filters profiles.

Generate two independent 256-path libraries per recording, alternating straight
and sinusoidally curved paths. Use the preceding literal-clock geometry prior:
uniform valid-grid endpoint proposals, 40-120 cm endpoint distance, curved
amplitude 20% of endpoint distance, 801 knots, rejection solely for map coverage.
Interpolate the source rate maps on supported local Delaunay triangles.
Physical clock: normalized Euclidean arc length. Neural clock: normalized arc
length in Hellinger distance between conditional cell-identity distributions.
This is a specific rate-code metric, not all possible neural representations.

Integrate piecewise-linear rates in clock coordinates analytically over each
full 20-ms bin; normalize integrated rates across cells. No supplied path,
generator label or latent index enters the scoring function. A uniform prior
over all candidate paths is marginalized, not maximized or selected per event.

## Generators and scoring models

Five models: physical, neural, stationary, physical_reset, neural_reset.
Stationary draws one uniform valid spatial bin for the whole event. Each reset
model draws a new path index independently per time bin, preserving that clock's
time-dependent snapshot distribution but removing shared-path continuity.
Reset is not a time-exchangeable bag or a literal time-permutation control.

For path h and time t, ell[h,t] = sum_c counts[t,c] log p[h,t,c].
Coherent log likelihood = logmeanexp_h(sum_t ell[h,t]).
Reset log likelihood = sum_t(logmeanexp_h ell[h,t]).
Stationary uses the event-summed counts and uniform spatial prior.
The multinomial coefficient is common to all models and cancels.

Generate mixtures with weights [0.6*(1-phi), 0.6*phi, 0.2, 0.1, 0.1],
phi in {0.25, 0.50, 0.75}. Generate 50 fresh Monte Carlo populations, each with
128 events per recording (PF 1024, Tanni 3200). Matched generation uses scorer
library 0; independent generation uses teacher library 1. Common exogenous
profile choices and labels are allowed across teacher conditions, but their
spike observations use independent RNG streams. The fixed namespace and seed
are `unknown_path_clocks_v1|20260909|...` with SHA256-derived substreams.

Score all events using the first 256 and first 128 candidate paths. The 128-path
condition is a nested-support diagnostic, NOT exact matching support even when
the teacher is library 0. No library is enlarged in response to results.
Save all count vectors and teacher indices for independent audit only.

## Population inference and gates

Fit all five mixture weights by event likelihood; no hard event labels.
Target phi = weight_neural / (weight_physical + weight_neural).
Profile-likelihood 95% intervals use LR=3.841458820694124 with nuisance weights
free. Verify numerical optimality and calibrate these asymptotic intervals by
Monte Carlo recovery rather than assuming their coverage.

Per dataset/teacher/support practical gate: maximum absolute bias <=0.10,
minimum empirical interval coverage >=0.90, false directional claims at phi=0.50
<=0.05, minimum correct-direction power at phi=0.25/0.75 >=0.80. Report Wilson
Monte Carlo intervals: 50 replicates do not precisely establish a 5% error rate.
Primary model-matched check is matched/256; robustness check is independent/256.
Neither event count nor path support is tuned to obtain a pass.

Monte Carlo intervals condition on fixed encoders and fixed geometry libraries;
they are not animal bootstrap intervals or biological confidence intervals.
Even a pass is limited by the restricted path prior, source encoding assumptions,
and absence of biological ground truth. A failure is not evidence of uniform
physical speed and does not refute other neural-code mechanisms.
