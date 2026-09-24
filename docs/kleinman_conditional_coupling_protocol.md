# Conditional ripple-to-RUN coupling calibration

2026-09-24. Frozen before this assay. This is a calibration, not a treatment
contrast or a score of observed future spatial firing.

## Biological target and previous measurement problem

Test whether intervening cell-wise ripple recruitment predicts subsequent
reference-aligned spatial firing, and eventually whether that association
differs with VTA suppression. This is not a map-stability metric, confirmed
replay, or a causal plasticity test. Prior generic replay/plasticity findings
are not novelty for this project.

The prior reference-centered score can have occupancy-dependent bias when
the reference map is estimated. Replace its centering by a conditional
two-period comparison that eliminates the unknown unchanged field.

For baseline/target count arrays B_b, Y_b and exposures T0_b, T1_b, assume
independent Poisson sampling with unknown baseline field lambda_b and scalar
period gain g:
 B_b ~ Pois(T0_b lambda_b)
 Y_b ~ Pois(T1_b lambda_b g exp(beta a_b)).
The feature a_b is a standardized log reference rate estimated strictly
before baseline; it is not estimated from B or Y. Standardization uses
unweighted jointly exposed bins and is recorded in the implementation.

Condition on C_b=B_b+Y_b in each bin and M=sum_b Y_b over jointly exposed bins:
 P(Y | C,M,beta) proportional to product_b
 choose(C_b,Y_b) (T1_b/T0_b)^Y_b exp(beta a_b Y_b).

Both lambda_b and g cancel. At beta=0, compute the exact mean and variance of
sum_b a_b Y_b by dynamic programming, without a large-count approximation.
U = observed sum - conditional null mean; V = conditional null variance.
Positive U means a relative redistribution toward high reference-rate bins,
not necessarily increased field reliability or synaptic strengthening.
This uses established conditional-Poisson nuisance elimination; we do not
claim a new statistical method.
Reference: https://doi.org/10.1007/s41884-022-00082-w

Bins exposed in only one period cannot identify redistribution; report their
spikes separately, not as evidence of change. No-count, one-period-only and
zero-feature-variance cells have zero information and remain in denominators.
The exact formula assumes independent Poisson spikes. Bursts and spatial
drift are explicit stress tests, not magically fixed by conditioning.

## Fixed bank and predictor

Use the same 48 anchors and saved numerical generating profiles/occupancies
as the prior calibration; do not reselect anchors. Read actual native ripple
and non-ripple immobile spike counts between baseline and target for each
bank unit. Use the previously audited valid tracking, speed <=8 cm/s, first
10 seconds of reward visits, clipped/merged native ripple exposures.

Predictor: log[(ripple_spikes+0.5)/ripple_seconds] minus
log[(background_spikes+0.5)/background_seconds]. Standardize it across
reference-included cells within each anchor. Zero ripple or background exposure
means the predictor is not defined; retain that anchor as unavailable, not
zero recruitment. Keep zero-spike cells and pseudocount choice visible.

Generate reference maps exactly as before at reference gain 1. Reuse its 16
fixed reference realizations, cyclically across 128 independent simulated
readout repetitions. Only reference counts/peak rate determine inclusion.
For the association, center the predictor further using conditional V within
anchor. This projects out a common alignment intercept within anchor.

Anchor score A = sum_i (x_i - weighted_mean_V(x)) U_i.
Anchor information I = sum_i V_i (x_i - weighted_mean_V(x))^2.
Animal effect is sum_anchors A / sum_anchors I. Each animal contributes one
effect to a two-sided t interval with five df; this small-six-animal procedure
is itself calibrated here, not assumed valid. Do not treat cells, anchors or
simulation replicas as independent treatment animals. Report all denominators.

## Cases fixed before running

128 replicas, seed 20260924, same fixed bank:
1. unchanged independent Poisson;
2. scalar target gain exp(0.7*x), correlated with recruitment, unchanged field;
3. doubled compound-Poisson counts (same expected rates), no spatial change;
4. compound counts plus recruitment-correlated scalar gain;
5. spatial fluctuations independent of recruitment plus correlated scalar gain:
   a smooth three-harmonic log-rate perturbation shared within an animal/replica;
6. planted positive alignment coupling: target profile multiplied by
   exp(0.35*x*standardized_log_true_field);
7. planted negative coupling, coefficient -0.35.

Preserve expected target spike totals for spatial perturbations before applying
scalar gain. A shared animal/replica lognormal baseline/target gain (SD 0.4)
adds within-animal dependence in all cases. The spatial-fluctuation case
deliberately breaks the unchanged-field assumption without imposing systematic
recruitment coupling. Numerical generating profiles, exposures and unit IDs
remain fixed. This tests a limited family of nuisances, not all neural dynamics.
Actual native future spatial spikes are not used as outcomes.

## Outputs and prospective screen

Write frozen predictor table, per-anchor denominators/effects, per-animal
effects, replicate estimates/intervals, by-case calibration, technical checks
and hash/commit manifest. Preserve failures; do not replace them with zero effects.

Engineering screen, not a biological hypothesis test:
- all repetitions keep six informative animals;
- each negative case: <=10% two-sided false flags, with 95% Wilson upper bound
  <=15%; report sampling uncertainty and actual counts;
- each planted case: >=80% correctly signed interval exclusions of zero;
- exact tiny-grid enumeration and source-count checks pass.

A failing case stops promotion to real coupling/drug claims. Do not relax the
screen, choose favorable animals, or increase replicas to obtain a pass.
A pass permits only a predeclared observational association analysis; it does
not establish full serial-dependence robustness, drug randomization, replay
causality or novelty. Three experimental and three control animals and
partly track-linked drug assignment remain important limitations.
