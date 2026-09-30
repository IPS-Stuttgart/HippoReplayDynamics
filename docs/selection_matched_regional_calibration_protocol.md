# Selection-matched regional calibration: frozen development protocol

2026-09-15; seed 2026091531. No hc-11 transfer until PF validation passes.

## Estimand and frozen inputs

Regional prevalence means the fraction of retained candidate endpoints whose
represented position at the end of the last complete 20-ms window is within
the inferred 20-cm Home disc. Moving-window spikes need not all come from this
terminal position. This is not a biological ground-truth assignment.

Use eight PF first-half RUN rate maps from the preceding frontier experiment.
Primary full population uses the same training-QC mask/grid as that experiment.
Also retain exactly the previously confirmed targeted Home-rich/Home-poor pairs
(four sessions) and whole-tetrode pairs (three sessions), on their original
grids with first-half maps. Do not rematch cells. Some paired populations overlap;
record overlap and the older full-RUN eligibility limitation explicitly.

Condition on the 200 previously frozen candidate count-time profiles per session.
Preserve EVERY original spike timestamp, including the surrounding session.
Within each frozen candidate, redraw cell identities using normalized fixed-map
intensities; no spike-count thinning or generated replacement timestamps.
Rerun the original detector on real position/speed and the complete count trace.
Because detection amplitude, boundaries and duration depend only on preserved
times, cache that part once and reevaluate active-cell support on every draw.
Audit this factorization with actual full-detector reruns on saved draws.
This is conditional on the frozen candidate templates, not an unconditional
simulation of the entire behavioral experiment or previously rejected templates.

## Generators and readout

Unchanged spatial tuning; independent cell allocation conditional on each spike
time. Latent endpoints are sampled uniformly within Home or its complement on
the occupied-grid largest connected component. Three equal-weight generators:
stationary; moving at 500 cm/s on an occupied-bin graph geodesic that ends at the
chosen endpoint; late jump from a distant location to that endpoint for the
last 5 ms. Moving intensities interpolate linearly along graph edges.
Paths terminate at the complete decoding-window endpoint, not at a partial bin.
For late jumps, sparse terminal spikes are a genuine information challenge and
may cause failure; do not silently replace it with stationary calibration.

Freeze the existing independent Poisson regional odds Bayes factor and ternary
readout (<1/3, neutral, >3; silent windows neutral). No density-model/threshold
selection after observing real endpoints. Fit a two-class ternary mixture using
population-specific equal-generator-mix calibration, with the existing Jeffreys
pseudocount convention. No latent independence assumption across populations.

## Independent stages (conditional on fixed templates)

- Calibration: 20 replicas per generator, balanced class assignments.
- Null threshold bank: 40 replicas at each prevalence .05/.15/.30/.50, generator
  mix equal in expectation. Freeze the maximum per-prevalence empirical 95th
  percentile of fit TV as the session/population flag threshold. Separately
  freeze a paired high-minus-low absolute-difference threshold from these nulls.
- Validation: 20 new replicas at each prevalence for each pure generator and
  the equal mix. Extra mix-only .38 panels permit a .30 versus .38 power check.
- Perturbation: 20 new .30 mixed replicas each: Home-peak cell participation
  halved; +2 Hz nonspatial intensity in Home-peak cells; within-class spatial
  distribution tilted toward one side; global gain x2. Under fixed total counts,
  global gain cancels exactly, and is explicitly a known-insensitive control.

Target labels are assigned before spike draws. True retained prevalence is
recomputed after active-cell selection. Replicas are Monte Carlo repeats, not
independent animals. Calibration/null/validation random seeds are disjoint.
Report omitted or class-degenerate cohorts; no zero-denominator passes.

## Frozen development requirements

Declared prevalence error budget: absolute error <=.05 in >=90% of validation
replicas for EACH session/population/generator/prevalence, with mean absolute
error <=.05 as a secondary summary. This is a proposed budget to test, not a
guaranteed confidence interval. Report empirical error quantiles separately.
False-flag target <=.05; development tolerance <=.10 on independent mixed-null
validation, and expose pure-generator failures rather than widening the null.
Report sensitivity to each perturbation and to actual errors >.05; no requirement
that the invisible global-gain control be detected.

Same-content paired estimates must differ by <=.05 in >=90% of mixed validation
replicas. Power: detect a positive .08 difference in >=80% of independent-panel
.30/.38 replicates using the frozen paired-null threshold. This is explicitly a
separate-cohort contrast, NOT a physically inconsistent assignment of different
truth to cells shared by two populations in one event. Report actual retained
truth difference and the limitations of this power proxy.

## Real data and claim boundary

Only after synthetic validation, compute real per-population mixture estimates,
TV, per-category residuals, matched-null flag and naive mass. Do not publish a
calibrated enrichment/reconciliation verdict for any population failing budget
or false-flag validation. Keep those rows as descriptive, not defensible under
the current calibration. Region area fraction is a spatial-uniform endpoint
reference, not a shuffled baseline or proof about selected replay prevalence.
No expectation that Rat3 must flag. No outcome establishes tuning drift or
multiple representations. If validation fails, stop before hc-11 and report
the failure, rather than changing the generator mix or error budget.

## Numerical preflight correction

The first full attempt stopped at an exact-count assertion for Rat4/Open1,
before outcome inspection. At its ~37,000-second clock, some nominal 5-ms
steps are shorter by sub-nanosecond rounding. Endpoint completion now uses
the frozen base-bin cache's 1e-9-second tolerance. The failed run is retained;
all sessions are rerun together in v2. No window rule or threshold was tuned.
