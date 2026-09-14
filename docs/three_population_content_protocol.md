# Three-population content observability diagnostic

Frozen before generating three-population outcomes or decoding AutoPI.
This follows two failed validation attempts. Neither is overwritten or rescued.

## Question

Can agreement and spatial concentration in two independent neuron groups predict
support from a third, unused group, and identify a fixed-coverage subset with less
population-dependent content error? Agreement is not biological replay truth.

This is a diagnostic with explicit abstention, not a correction that makes every
candidate reliable. All original rows remain visible, with retained/unresolved
flags. No claims about unbiased replay prevalence or destination preference.

## Development and independent data

- Develop on all eight existing PF recordings/four rats and their fixed MUA
  candidate endpoints. Use first-half RUN maps, third-quarter calibration and
  fourth-quarter held-out RUN positions. Original full-RUN unit eligibility is a
  conditioning boundary, as in the previous studies.
- External data: all readable AutoPI CA1 recording directories in the complete
  extraction on gpuserver4090. No ranking by replay outcomes. Use the FIRST
  native circ80 foraging epoch to estimate maps and validate RUN decoding, and
  its immediately following native rest epoch for candidate detection. Include
  only author-labelled good clusters on CA1 probes. These are rest high-MUA
  candidates, not verified sleep, ripple or replay events.
- Native seconds/centimetres and sorted-cluster IDs are retained. No time shifts,
  pickle execution, or arbitrary waveform reclassification. Record all files and
  hashes, clocks, electrode regions, source-unit counts and exclusion reasons.
- Detect rest candidates using the summed good-unit spike counts in 1 ms bins,
  Gaussian smoothing SD 10 ms, peaks above mean+3 SD of that rest epoch, and
  boundaries at the mean. Duration 50-2000 ms, at least 5 spikes, at least 3
  active units and at least 10% of good units active. No ripple condition or
  continuity filter. This does not assume that every candidate is a trajectory.
- Common encoding: 8 cm grid, 1.5-bin spatial rate smoothing, RUN speed >=10 cm/s,
  occupancy >=0.05 s, >=30 running spikes, mean RUN rate <=4 Hz, peak >=2 Hz,
  split-half map stability >=0.25. Unknown cell types remain labelled unknown.
- All excluded sessions/candidates retain denominators. At least 4 external
  animals and 80% of source candidate endpoints must be technically evaluable.
  A feasibility failure is not permission to relax thresholds.

## Independent populations and fixed readout

Three seeded random disjoint groups A, B, C, each floor(n_eligible/3) neurons;
drop at most two odd remainder units. At least 5 neurons per group. A, B and C
receive the same unit count. No assignment is optimized using replay content.
Three fixed seeds, primary split 0; other splits are sensitivity only.

Use the original last complete 20 ms in each candidate, excluding any partial
last 5 ms base bin. No endpoint trimming, peak recentering, temporal smoothing or
HMM prior. Independently decode each 20 ms bin using a flat spatial prior and a
Poisson likelihood. The instability contrast is A versus C, both equal-sized.
B is used only to form diagnostic features, not to re-infer A or C.

Normalized spatial scale is D = diagonal of the valid decoding grid. Support
means posterior mass >=0.5 within radius 0.15 D of A's posterior mean. This is a
coarse region, not an asserted precise destination. Target label comes only from
C. A/B posterior means, widths, entropies, counts, active units, A-local mass,
B support at A, A/B regional TV, and A/B mean separation may enter prediction.
No C replay spikes, posteriors or outcomes may enter feature construction,
training-row selection or prediction for that event.

Fit fixed L2 logistic models (C=1) with training-only median imputation and
standardization. Weight events equally within session, sessions within rat,
then rats equally; normalize training weights to mean one. No class reweighting.
Baseline models: constant training prevalence; pooled A+B counts/active units,
posterior entropy/width/local mass and number of cells. Full model adds separate
A/B features and their disagreement. The pooled baseline has the SAME observed
neurons as the full diagnostic. Spatial lengths are normalized by D.

PF leave-one-rat-out evaluation is developmental, not external confirmation.
Fit final models on all PF primary real endpoints, serialize coefficients,
scaling, imputation and feature names, and freeze/hash before AutoPI outcomes.
No tuning on AutoPI outcomes or on Tanni/Blackstad reanalyses.

## Fixed coverage and known-truth checks

Retain the best predicted-support half of events within each recording, ceil(n/2),
with deterministic event-index ties. This is a ranking diagnostic, not a claim
that each retained event is correct. Compare with random-retention expectation,
lowest A entropy and lowest pooled entropy at exactly the same coverage.

Validation readouts: C-support log loss and Brier score; A/C regional posterior TV
on 3x3 tiles, A/C mean separation, each side's entropy and true-position error.
Use observed fourth-quarter RUN, plus two count-matched known-endpoint generators:
first-half maps and second-half maps with lognormal per-cell gain (log SD 0.4).
Generate one whole-universe count vector and partition it into the SAME A/B/C
assignments. Two simulation draws; draws/splits never count as new animals.
Use matched total counts from the original candidate endpoint. Keep zero-count
observations rather than pretending their posteriors are reliable.

Aggregate events within session, sessions within animal, then equally weight
animals. Report every animal and all fixed split sensitivities.

## Required external gates

- Exact inputs, unit identity, frozen selection and predictions pass an audit;
  C-feature perturbation cannot change predictions or retained events.
- >=4 animals and >=80% source endpoints; all evaluated events and failures shown.
- Full diagnostic improves external support log loss by >=5% versus BOTH training
  prevalence and the pooled-neuron baseline, and improves Brier score over both.
- Log-loss improvement versus pooled baseline is positive in >=75% of animals,
  with descriptive animal-bootstrap lower bound >0 (5000 draws).
- Retained half reduces mean A/C regional TV and endpoint separation by >=10%
  versus random expectation, each in >=75% of animals and with bootstrap lower
  bound >0. Also report comparisons against entropy-only retention.
- Neither A nor C mean posterior entropy increases in retained real endpoints.
- Neither A nor C mean true-position error worsens after retention on observed
  RUN or either known-endpoint generator. Report absolute errors, not only changes.
- Any failed gate means no complete validated diagnostic. No post-hoc threshold
  change, replacement seed or claim based only on a favorable individual animal.

Passing would validate a bounded, population-reproducibility screening diagnostic
on this external candidate class, not establish ground-truth replay content or
novelty over prior work. Precision of neural-population support is not precision
of true biological memory retrieval.
