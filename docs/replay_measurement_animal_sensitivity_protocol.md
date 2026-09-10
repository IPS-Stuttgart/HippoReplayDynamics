# Animal-Level Sensitivity Of The Measurement Results

2026-09-11. Retrospective sensitivity analysis, not new primary gates. The
published-style continuity, predictive and simulation analyses remain frozen.
The purpose is to test animal dependence and expose small-sample inferential
limits, not to search for a new favorable endpoint.

## Fixed Endpoints

- Full two-shuffle continuity: half-minus-full acceptance, original-order
  immobile MUA and ripple cohorts, edge-only support, ten frames, alpha 0.02.
- Training-only geometry: all five original IMM predictive contrasts, in
  rejected-with-opportunity and lost-with-thinning groups. Report original
  nats and the existing per-held-out-spike normalization separately.
- Known-speed recovery: separately estimated RUN-map response minus latent
  arclength truth, minus finite-window truth, and minus known-map response.
  Use the existing full-cell, Poisson, posterior-mean, unfiltered,
  before-selection, true-coordinate slice. These are simulations using the
  observed animals' rate maps, not measured biological speed gradients.

Preserve each source's existing event -> session -> animal aggregation. Do
not pool datasets or treat overlapping detector cohorts, repeated splits,
simulation realizations or endpoints as independent animal replicates.
Expected complete animal counts: four PF and five Tanni. Fail on missing,
duplicated or nonfinite selected animal rows. All five prediction contrasts
must be represented; the Tanni composition failure may not be removed.

## Calculations

For each endpoint, report the equal-animal mean and every leave-one-animal-out
mean. These are deletion sensitivities of fixed estimates, not newly trained
decoders, confidence intervals or independent replication. Also report
per-animal values and both one-sided directional and two-sided exact sign
tests. The directional alternatives are loss for continuity/gradient
attenuation and gain for prediction. Exact zero effects contribute ties,
excluded from the binomial sign calculation and reported explicitly; an
all-tie endpoint has p=1 and no directional support. No numerical near-zero
threshold will be fitted to the results.

The sign sensitivity assumes independent animal signs with null probability
one half. It tests sign consistency, not the same magnitude estimand as the
hierarchical bootstrap, and does not capture simulation/map uncertainty.
The smallest attainable one-sided p is 1/16 with four nonzero animals and
1/32 with five; two-sided minima are twice these values. Do not interpret
a p-value above 0.05 as equivalence, absence of an effect or failure of the
underlying computational verification. The diagnostics are unadjusted,
retrospective and not new confirmatory discoveries.

Use the established exact binomial implementation, documented at
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html.
Tests must include known four/five-animal limits, ties, a single influential
animal, missing/duplicate animal rows, unequal-event weighting invariance,
source checksum rejection and preservation of adverse contrasts.

## Outputs

`replay_measurement_animal_values.csv`,
`replay_measurement_leave_one_animal_out.csv`,
`replay_measurement_animal_sensitivity_summary.csv`,
`replay_measurement_animal_sensitivity.md`, and
`replay_measurement_animal_sensitivity_manifest.json`.
The report must record sources and exact filters, mean reconstruction checks,
code/command provenance and input/output hashes. No event rescoring, new event
selection, biological scale-up or threshold optimization is permitted.
