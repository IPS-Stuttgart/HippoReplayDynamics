# Fixed-time encoding-uncertainty content test

Frozen before any decoding outcome from this method. Date 2026-09-14.

## Question and scope

Does marginalizing uncertain RUN firing rates reduce the population dependence
of decoded content at the SAME endpoint, with better calibrated spatial recovery
than either plug-in Poisson decoding or entropy-matched likelihood tempering?
This follows failed predictor, sampling-balance and edge-trimming experiments.
Do not overwrite them or call sparse raw endpoints accepted replay trajectories.

Use the audited edge-support inputs: PF and hc11, each eight sessions/four rats,
the same 200 frozen candidates/session and three frozen equal-population splits.
All rates/unit QC use first-half RUN; quarter maps are halves of that TRAINING
half, not held-out RUN. No new unit, window, region or population selection.
Independent 20-ms windows, raw endpoint only, unchanged 8-cm grid/flat prior.
No temporal prior, smoothing of decoded paths, time shift or abstention.

## Frozen observation models

1. `poisson`: original fixed-rate likelihood, unchanged.
2. `gamma_exposure`: a moment-matched Gamma distribution for each cell/state
   rate, with exactly the original mean and variance due to finite training
   occupancy under a locally constant Poisson-rate approximation.
3. `gamma_training_drift`: primary. Adds nonnegative training-quarter rate
   variability after subtracting its estimated sampling variance.
4. `poisson_entropy_matched`: per-window, per-population temperature adjusted to
   match the primary Gamma posterior entropy. It never uses the other population
   or true position. This controls agreement obtained simply by flattening.

Let K be the exact 2D Gaussian smoothing matrix (sigma1.5 bins, constant-zero
padding, truncate4), O the raw training occupancy, S=K O, D=max(S,.05s),
Q=(K squared elementwise) O. Define T_eff=S D/Q. With original mean rate mu,
the sampling variance is mu/T_eff. This preserves the clipped-denominator
estimator's mean rather than silently changing its rate map.

Compute these quantities separately for full training, q1 and q2. Where BOTH
quarter raw occupancies are >=.05s, define temporal variance as
max(0, (mu_q1-mu_q2)^2/2 - (mu_q1/T_q1+mu_q2/T_q2)/2); otherwise it is zero.
Primary rate variance = full-training sampling variance + temporal variance.
Gamma shape=mu^2/variance, rate=shape/mu. Integrate the Poisson likelihood over
this Gamma analytically (negative-binomial predictive count distribution).
No dispersion multiplier, sweep, fitted hyperparameter or replay-gain tuning.

This is a moment-based uncertainty model, NOT an exact Bayesian posterior for
smoothed maps, NOT a measure of biological rate drift, and NOT a new discovery
of the Gamma-Poisson model. It ignores correlated rate uncertainty across cells.
Two training quarters give noisy variance estimates; external failure is plausible.

Entropy control uses bisection over log-temperature [-20,20] with 70 iterations.
Require entropy agreement <=1e-7 in every reported pair. If an entropy target is
unreachable (e.g. exact likelihood ties), mark that control unavailable; do not
drop the event or use a more favorable substitute.

## Diagnostic and outcomes

An A-only encoding-sensitivity diagnostic is the 3x3 regional TV between A
posteriors obtained using q1 and q2 maps at the original endpoint. Report its
Spearman association with independent B position error on known-position data,
and A/B content disagreement on real candidates, per session/rat. No threshold
or learned selector is fit in this experiment. A descriptive correlation is not
by itself a validated destination certificate.

Store every method's A/B posterior means, entropy, RMS width, 3x3 regional
probabilities, separation and regional TV. For known positions store A/B
Euclidean error, regional Brier score and regional negative log probability.
Truth region is the frozen 3x3 region containing the original 20-ms mean truth
position (same fixed-time target as the previous controls).

Controls: the unchanged audited q4 RUN and all four known-path sources
(stationary, moving1000cm/s, moving with gains, late jump). Rate maps/cell IDs,
whole-population simulated draws and original times remain identical across
methods. Do not claim simulations preserve real noise correlations or represent
verified native maze topology. Native real candidates have no position truth.

## Prospective decision rule

All technical/source/audit checks must pass non-vacuously. No candidate loss or
timing changes. Split0 is primary; later splits cannot rescue a primary failure.
Average events within session, sessions within rat, rats within dataset. Use
descriptive four-rat bootstrap intervals; repeated splits are not animals.

A promising remedy must, on hc11:
- Reduce real mean separation AND regional TV >=10% versus Poisson, with the
  same directions in >=3/4 rats.
- Beat entropy-matched Poisson on BOTH real separation and regional TV; do not
  certify a uniform/broader-posterior solution.
- Not increase q4 RUN mean A/B error and regional Brier score.
- Have <=2cm upper one-sided descriptive95% rat-bootstrap bound on A/B error
  increase for EACH known-position source; this is an explicit practical
  noninferiority tolerance, not proof of equality.
- Not worsen mean A/B regional Brier in any simulated source.
- Keep all 1600 real candidates; entropy-matched control must be available for
  every evaluated event/population. Missing controls block validation.

Two centimetres is predeclared here (not chosen after these outcomes); unlike
the prior trimming test, this is a noninferiority check for a different estimator.
The comparison against entropy-matched decoding and proper scores remains
mandatory. The current hc11 cohort has informed previous failed experiments,
so even passing this screen is not a final independent-confirmation claim.

If the screen passes, freeze code and test disjoint hc11 candidates (next200
SHA256-ranked IDs/session) before any success claim. Also test the original
matched-population Home-content/accepted-endpoint contrast without changing its
population identities, plus independent nonuniform known-content controls. A
smaller effect on raw random-half endpoints alone cannot close the original goal.
If this screen fails, preserve it and do not tune variance/temperature/support
rules on hc11 to reverse the result.

## Artifacts and provenance

Commit before decoding, all jobs detached on gpuserver6000. Record source hashes,
training uncertainty arrays, event-model readouts, full posterior audit arrays,
diagnostic tables, session/rat summaries, gates, plots, and a non-rescoring outcome
note. Independently reconstruct effective exposure, predictive likelihoods,
temperature matching, fixed-time counts and output metrics. No biological claim.
