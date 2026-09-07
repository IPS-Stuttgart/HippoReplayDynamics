# Research search log, 2026-09-08

The objective remains a genuinely novel, potentially high-importance paper.
The completed recording-coverage benchmark is not substituted for that goal.
The preceding reconnaissance was progress: authoritative reports contradicted
several optimistic historical chat claims and changed the next experiment.

## Verified existing outcomes

- PF train-only mode study: adjusted rho 0.420, interval [0.284, 0.518], 160
  events/four rats. Important scoring caveat discovered on code inspection:
  source temperature is 0.3 and the historical frozen held-out helper sums
  exponentiated supplied emissions. Powered test emissions are not normalized
  probabilities. Re-evaluate with proper held-out probabilities before repeating
  the probabilistic interpretation. This does not prove the effect disappears.
- hc-11 strict IMM holdout: 250 events/five sessions/two animals, zero strict
  clean-IMM events; stopgate, not external IMM replication.
- hc-11 PRE/POST control audit: completed eight-session/four-animal controls,
  no validated ordered or strict-clean events in the primary matched set. Some
  time-order changes are descriptive; not a validated learning-dependent effect.
- Ten-hypothesis PF behavior campaign: 0/10 after the frozen family correction.
- Commitment/composition: neither primary specialization supported.
- Denovellis surprise proxy: 2664 choices/nine animals; predicted replay-rate
  and extent increases unsupported. Reward-pump agreement 84.6%, not a perfect
  unexpected-outcome manipulation or a Bayesian-smoothing measurement.
- Newer PF spatial revision consumer, a065d092, 20260904: abstains. All 160
  causal-prefix predictor events enter recovery, but the recovery gate fails;
  smoothing-specific contrasts uncertain, with four independent rats only.
  This is stronger context than the older PF route-geometry audit alone.

## Current experiment

`pf_count_conditioned_prediction.md` freezes the identity-versus-total-rate
discriminator before results. Scorer f63f1dad; 160 events, five 70/30 splits,
three training likelihoods, two maps, two inference temperatures, five model
implementations (one static duplicate used as a correctness check).
Held-out probabilities always untempered. All 480 technical smoke rows passed.
Full 48000-row scoring completed on gpuserver6000; all technical gates and
independent output reconstruction pass. No result-driven model/threshold
selection or change in event cohort occurred. Results and limitations are in
`pf_count_conditioned_prediction_results.md`.

The primary conditional-identity score favors IMM over iid (+7.723), static
(+8.599), and diffusion (+1.879); real-map increment +0.763, all four rats
positive. But IMM-minus-diffusion reverses to -1.367 at inference T=0.3 (all
four rat estimates negative; interval includes zero). The robust lead is
cross-cell temporal coordination beyond total firing, not uniquely preferred
switching dynamics. This is progress, not completion of the high-importance
discovery goal. No new biological hypothesis is counted as a positive.

An independent source lookup during scoring verified that each session's 20
frozen event IDs exactly match its first 20 native ripple indices within RUN:
Rat1 Open1/Open2 (304/496 total native RUN events), Rat2 (119/159),
Rat3 (406/328), Rat4 (643/388). The present cohort is therefore not the 108
all-cell clean-IMM subset and not selected by a model-winning threshold. It
remains a historically inspected, early-session convenience sample, not a new
confirmatory cohort. "All-cell selected event cohort" in the generated report
means native population-event selection, not selection on all-cell IMM scores.

This asks whether there is a substantive spatial-temporal signal worth
replicating. It is not yet a new biological mechanism and does not invent
conditional multinomial decoding. A total-rate-only decoder still knows the RUN
population-total map; it is a count-only inference baseline, not a universal
model of every nonspatial neural process.

During the run, add only validation/reporting: known-map stationary versus
moving identity simulations; independent CSV/aggregation/hash reconstruction;
descriptive per-held-out-spike normalization. These do not alter the scorer.
The separate verifier also checks exact split/temperature keys, unchanged
training-posterior declarations, actual cell IDs, probability bounds, and
independently reconstructs paired contrasts and their event/animal aggregates.
It does not independently recompute raw posteriors or bootstrap intervals.

## Novelty constraints and future discriminator

- Denovellis et al., eLife 64505 (2021): switching replay dynamics established.
  https://elifesciences.org/articles/64505
- Tirole et al., eLife 79031 (2022): experience-dependent rate modulation and
  decoder-selection controls already studied.
  https://elifesciences.org/articles/79031
- Ujfalussy and Orban, eLife 74058 (2022): trajectory uncertainty representations
  during theta, including population gain as a coding signature.
  https://elifesciences.org/articles/74058
- Wu and Foster (2014): replay captures learned maze topology. Physical versus
  representational geometry is not a new broad concept.
  https://pmc.ncbi.nlm.nih.gov/articles/PMC4012305/
- Ji et al. (2026): firing-rate adaptation and field-coverage caveats.
  https://www.nature.com/articles/s41467-025-68042-3
- Bakermans et al. (2025): compositional replay and subsequent spatial-code
  change; do not present that link as untested territory.
  https://www.nature.com/articles/s41593-025-01908-3

If conditional prediction is informative, a potential next discriminator is
physical-distance versus RUN-code-similarity dynamics, evaluated with held-out
neurons and known-generator recovery instead of derivative speed heatmaps.
This is not selected as a positive result, not yet implemented, and a targeted
search failing to find identical wording is not proof of novelty. It would
need matched model flexibility, training-only metric fitting, and independent
validation before a neural-space propagation claim.
