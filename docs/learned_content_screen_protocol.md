# Accuracy-aware paired-population screen

Frozen before model fitting, 2026-09-15. Development experiment only. The previous
oracle bound shows attainability, not an independently validated predictor.

## Fixed task and observations

Keep the original four matched PF population pairs, 1,836 candidate endpoints and
513 accepted-segment endpoints. No extra cells, altered windows, rate-map changes,
likelihood changes, temporal context, or event reselection. Decode with the same
independent 20-ms Poisson likelihood and flat spatial prior.

Primary screen retains ceil(N/2) observations per session/source/map, by a learned
score with frozen seeded hash tie-breaking. Native Q4 and five simulated truth
banks remain untouched by fitting. Earlier results from those banks informed this
research program, so this is DEVELOPMENT, not a new blinded confirmation.

## Learning without target-rat calibration

Fit three models, leaving one rat out entirely each time (Rat1's two sessions
always remain together). Training uses ONLY other rats' earlier native RUN Q3
(first 20-ms observation per 250-ms parent), cal_poisson_gain1 and cal_conditional.
No test-bank labels, oracle probabilities, target-rat calibration, replay rows,
or source/session identifiers enter fitting. Prior full-RUN cell eligibility
remains inherited from the fixed experiment; this is not raw-recording blinding.

Features are high/low Home posterior mass, normalized posterior entropy,
log(1+spikes), log(1+active cells), mean-position separation divided by arena
diagonal, and regional total-variation disagreement. This explicitly uses BOTH
populations. It is not an A-only forecast of unseen B, nor an independent test of
pairwise disagreement. It tests whether accuracy-aware screening can transfer.

Construct a regression tree predicting signed Home difference, normalized
position separation, regional TV, both entropies, both normalized physical
errors and both Home Brier losses. All nine targets are standardized using
training-weighted means/SDs (SD floor 0.01). Fixed tree: depth<=8, <=128 leaves,
>=64 training rows per leaf, squared error, random_state=20260915. Equal weights
per source, rat, session within rat, and true Home/non-Home class. Export the
tree as numerical arrays, not executable pickle.

The tree creates observable feature leaves, not a truth-based selector at test
time. Learn ONE shared retention probability per leaf, jointly across the three
training sources. Exactly 50% training weight is retained in each true class
within each source. Per source, weighted Home gap must fall by 20%*t and
separation/TV by 10%*t; neither entropy nor either population's class-balanced
error/Brier may increase. Maximize t in [0,1]. Constant probability 0.5, t=0 must
always be feasible. A second equivalent LP retains the optimal t (tolerance1e-8)
while minimizing training-frequency-weighted absolute deviation from 0.5, to
avoid arbitrarily extreme policies. Save both primal/dual certificates.

At evaluation, traverse the frozen tree, rank by leaf probability, and keep the
top half with seeded ties. No target truth is used to choose the cutoff or
rebalance classes. Fractional calibration constraints do NOT guarantee the
realized hard subset satisfies them. Report that conversion honestly.

## Development gates (not external success)

Keep the prior screen's gates: complete original denominators and models; half
retention; >=20% equal-rat reduction in candidate Home gap for early/full maps,
no rat worsening; accepted-endpoint Home gap must not worsen overall/per rat;
>=10% separation and regional-TV reduction, improving in at least 75% of rats;
no pooled entropy increase; no increase in either population's class-balanced
error or Home Brier within ANY rat/source of native Q4 or the five truth banks;
retain at least20% of both true classes in every control session. Compare with
frozen simple spike-count and entropy half-screens, without promoting a comparator
post hoc. Report predictor/disagreement associations as descriptive because the
inputs already include disagreement.

Passing those numerical gates is necessary but not sufficient: independent
reconstruction and evaluation on independent recordings are still required.
If calibration t=0 or held-out development fails, retain and report that failure;
do not tune depth, leaves, sources, class balance or thresholds to rescue it.

## Audit

Hash all source files and code; freeze all three exported models BEFORE opening
target evaluation banks. Preserve exact observed counts and event identifiers.
Independently reproduce calibration features and targets, refit the fixed trees,
rebuild both LPs and verify their primal/dual certificates. Verify test scores by
independent tree traversal, deterministic selection and weighted summaries.
Never report green software tests or a training optimum as a successful remedy.
