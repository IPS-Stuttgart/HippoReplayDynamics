# First native ripple/spatial-alignment pilot

2026-09-24. Frozen after conditional-association calibration and independent
verification passed, but before native baseline/target spatial outcomes are
scored by this script.

Question: within these fixed opportunities, do cells with greater intervening
ripple enrichment show greater subsequent reference-aligned spatial firing?
This is an observational precursor to the VTA-interaction question, not a
test of ripple-caused plasticity, replay content or a drug mechanism.

## Selection and endpoint

Use exactly the same 48 identity-selected anchors as the preceding calibrations.
No replacement for zero-ripple or uninformative anchors. Keep all six animals
and all original condition/direction labels in denominator tables. Do not select
new events, sessions, cells or thresholds based on the observed association.

Fit each native cell's reference profile from the three same-direction RUN
traversals strictly preceding baseline, within the same reward epoch. Use
the established raw composite tetrode/cluster IDs, temporal alignment, 2-cm
bins, 4-cm Gaussian smoothing, >8 cm/s training, >=10 reference spikes,
>=1 Hz reference peak and 1e-5 rate floor. Use all positive-ID raw units as
the candidate universe, then reference-only inclusion. This may differ from
the full-RUN unit universe used as numerical generating truth in calibration.

Baseline and target outcomes use >20 cm/s interior RUN intervals, >20 cm
inside reward thresholds, matching the prior chronological audit. A cell's
intervening predictor is log[(ripple count+.5)/ripple exposure] minus
log[(nonripple immobile count+.5)/nonripple exposure], exactly as calibrated.
No-ripple or no-background exposure means unavailable, not zero recruitment.

Use the calibrated conditional two-period score; condition out the unknown
baseline spatial map and scalar period gain. Standardize log reference map
over jointly exposed bins, standardize recruitment across reference-selected
cells, and project out the within-anchor alignment intercept using conditional
information. Preserve zero-spike/noninformative cells and excluded spatial
support counts in the ledger. Fit no latent replay path.

Primary pilot summary: each animal contributes one score-based association,
sum(anchor score)/sum(anchor information), then an equally weighted mean and
two-sided 95% t interval across six animals, as in calibration. Also report
all animal and anchor estimates and denominators. This is a small-sample
working interval, not certified robustness to all native temporal dependence.
The one-step score estimate is not an unbiased estimate of a biological
plasticity coefficient; the planted calibration coefficients were attenuated.

## No tuning or claim escalation

There is one primary observational association, not multiple condition-specific
tests. Drug and novelty labels are retained for coverage only; do not fit their
contrasts in this pilot. No direction/event-quality/animal subgroup search.
No significance-dependent exclusion. If fewer than six animals have informative
anchors, mark the pilot incomplete rather than replacing them.

A positive association would not yet be novel or causal: replay/field plasticity
associations exist already. A null pilot, especially a wide interval, does not
prove absence of learning or rule out a drug-dependent association. Inspect
coverage and uncertainty before deciding the next step. The old replay-extent
stop remains in force. A paper claim requires a separately frozen VTA-coupling
contrast, temporal-specificity controls and an honest novelty comparison.

Native firing can violate independent-Poisson assumptions through burst,
spatially varying state and overlapping histories. The seven fixed simulations
checked limited such cases; they are not proof that all confounding is removed.
Only three experimental and three control animals and partly track-linked
drug assignment remain binding limitations.
