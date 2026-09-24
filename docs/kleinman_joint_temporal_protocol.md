# Kleinman joint temporal-specificity calibration

Frozen simulation-only protocol, 24 September 2026. No native spatial-outcome
association or drug coefficient is authorized by this protocol.

## Biological purpose and limits

Before testing whether ripple recruitment predicts later spatial firing more
strongly than preceding spatial changes, establish that a shared baseline,
scalar rate changes and ongoing spatial drift do not manufacture a prospective
advantage. This is a prerequisite for the original VTA-coupling investigation,
not a replacement positive result or a novel generic replay/plasticity claim.
Track identity is still missing for a within-track drug comparison.

Select two of the previously audited available non-overlapping packets per
animal x drug x novelty, by SHA256(seed,animal,session,packet_id), seed 20260924.
This is 48 packets across six animals. Selection uses the already frozen
availability descriptor, never native spatial-change scores. No replacement.

## Numerical bank

Freeze full-RUN directional firing maps as numerical generating truth only.
Actual occupancy in the three reference traversals and R3/R4/R5 is retained.
No native R3/R4/R5 outcome scores enter this calibration. Real ripple/background
counts provide fixed cell predictors. Full-RUN generating rates must never be
used as reference estimates in an eventual native test.

Fit reference rates to independently Poisson-generated reference counts; retain
the existing reference-only cell rule (10 spikes, peak 1 Hz).
Use 16 frozen reference realizations cyclically across 128 readout repetitions.
Generating maps, occupancies, predictor counts, seed and cases are fixed.
Each case generates exactly ONE R3,R4,R5 count array per cell; both comparisons
use the identical R4 draw. Never resample baseline separately for each lag.

Only bins exposed in all three readouts contribute, ensuring the preceding
and prospective comparisons examine the same spatial support. Standardize
log reference rate on this common support, identically for both comparisons.
Conditional two-period scores remove arbitrary unchanged maps and scalar gains.
Center predictors by conditional information within packet as previously.
Within each animal, sum scores/information separately for the two lags, then
take the prospective-minus-preceding one-step effect. Weight six animals equally;
use a working two-sided 95% t interval with df=5. Score effects are attenuated,
not unbiased biological coefficients. No cell is an independent animal.

## Cases, fixed before results

1. unchanged: fixed spatial maps.
2. scalar_gain: cell recruitment-correlated scalar gain across readouts.
3. burst_gain: compound two-spike counts plus scalar gain.
4. spatial_drift: animal-shared, smooth spatial drift unrelated to predictor.
5. continuous_positive_drift: the same +0.35 predictor-linked log-shape step
   from R3 to R4 and R4 to R5, no special prospective change.
6. continuous_negative_drift: the same -0.35 step in each interval.
7. future_only: +0.35 predictor-linked shape change only after R4.
8. past_only: +0.35 predictor-linked shape change only before R4.

Shape modulation uses standardized log generating rate on the same triple
support. Normalize each period's expected total to its original occupancy-
weighted rate before scalar gain. Animal-shared log gain SD=0.4 is present in
every case; scalar gain coupling=0.7; smooth spatial-drift SD=0.6.
The continuous cases test drift per traversal, not continuous elapsed-time drift.
They are stress tests, not a complete model of spontaneous neural dynamics.

## Frozen engineering screens

All repetitions must retain six informative animals. For the six no-specific-
prospective-change cases, two-sided false-flag fraction <=0.10 and Wilson 95%
upper <=0.15. Future-only must produce positive flags >=0.80; past-only negative
flags >=0.80. These are engineering checks, not biological p-value thresholds.
Report preceding and prospective effects, their paired contrast and its Monte
Carlo uncertainty, all missing-information counts and all cases.

If any required screen fails, do not score the native temporal contrast, do not
change the bound or discard the failed case, and preserve the failed calibration.
A passing simulated assay would still not establish causality, native-model
adequacy, novelty or a VTA effect. This protocol alone authorizes no native run.
