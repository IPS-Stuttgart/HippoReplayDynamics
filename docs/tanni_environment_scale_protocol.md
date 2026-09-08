# Environment scale and independently predictive temporal structure

Frozen exploratory protocol, 2026-09-08, before inspecting the area-stratified
predictive outcomes. This is a new question in reused data, not independent
confirmation after the many preceding project hypotheses.

## Question and prior work

Does the apparent loss of geometric trajectories with larger environments
also appear in held-out neural predictive temporal organization? A dissociation
would motivate a recording/representation study; a robust change in predictive
organization would motivate a separately controlled biological scaling test.
Neither outcome by itself demonstrates physical replay speed or a mechanism.

[Tanni et al. 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC9616721/) recorded
five animals in four familiar environments with area doubling A through D.
A was first and last; B/C/D visit order was randomized per animal. Recording
duration scaled with area. Physical arena size is therefore not separable from
context identity or duration/exposure in these data. The source paper already
establishes firing-field/density scaling, so that is not a new endpoint.

## Frozen inputs

- Geometry/prediction manifest SHA256:
  `8861c60bf36b98ce179237276e8d16cb6294c1d71077c9100b7f4776c36237c0`.
- Independently verified geometry labels and proper held-out prediction,
  all 5,224 Tanni high-MUA candidates, 25 recordings, five animals.
- Coverage input manifest SHA256:
  `2ed1c4d910a3ddc4027eefb398d48a55bae6829c3bf16e8f57955beb9ee55613`.
- Native NWB arena_size and animal identity must agree with source metadata.
  Never infer walls from tracking extent or choose sessions using outcomes.
- One frozen 70/30 training/held-out split (split0) is primary. Splits1-4
  are separate sensitivity analyses, never pooled as extra biological events.
- Geometry is the training-only edge10 rule: flat-prior independent Poisson,
  overlapping20ms/5ms, <20cm jumps, >=10frames, >=40cm displacement. It is
  geometry only, not the two-shuffle-validated original-author replay label.
- Prediction uses frozen training posteriors, proper held-out cell identity
  log scores, and whole-bin order x spatial-map controls. No rescoring/tuning.

## Primary endpoints and analysis

1. Session fraction passing geometry.
2. Session mean original-minus-shuffled IMM score per held-out spike.
3. Session mean IMM order-by-map interaction per held-out spike.

The ratios are paired within event/split, then averaged across events in a
session. Zero-held-out-spike ratios remain missing and their denominator is
reported. This is mean event information per spike, not a pooled-spike ratio.
Report raw score endpoints and IMM versus other-event composition separately.
Do not use an order sensitivity result to rescue the failed Tanni composition
baseline from the parent experiment.

Primary size effect: B/C/D only, one session per area per animal, ordinary
within-animal slope against log2(area/A). Each animal has equal weight, not
weight proportional to events or recording duration. Report all five animal
slopes, mean slope, t-based95% interval with4df, and leave-one-animal-out means.
Enumerate all6^5 within-animal B/C/D label permutations as a reference for
the equal-animal mean slope; two-sided, inclusive ties, no Monte Carlo +1.
Holm-adjust the three primary permutation-reference probabilities. These are
conditional exploratory references, not a guarantee of causal area inference.
Small animal count, context/duration coupling and heterogeneous session support
remain limitations. A nonsignificant slope is not evidence of invariance.

## Predeclared diagnostics, not replacement primaries

- Repeat primary point estimates separately for the other four cell splits.
- Include A by first averaging its two visits, to avoid double weighting A.
- Report A-return minus A-first for all endpoints as a temporal/exposure check.
- Event support: units, spikes, event duration, training posterior entropy,
  valid spatial bins, and native session duration, by area and animal.
- Session-level B/C/D regression with animal intercepts, log training cells,
  log(1+median training spikes), and log median event duration. Report area
  coefficient, design rank/condition number and area residual variance. A second
  diagnostic adds normalized training entropy. This is descriptive adjustment,
  not causal mediation: these covariates can themselves respond to environment.
  With15sessions, do not fit arbitrary additional predictors or tune subsets.
- Leave one animal out of each adjusted regression; retain rank failures.
- Diffusion order controls, unnormalized IMM scores and the other-event
  composition comparison remain descriptive, not alternative winning endpoints.

## Decision and stop rules

Do not label a null predictive slope as constant physical speed. A coherent
size association needs the frozen primary analysis, animal consistency and
support/visit-order diagnostics, followed by matched known-path recovery
and independent data before a biological scaling claim. A geometry-only
association would strengthen the measurement paper, not establish novel
biology. Mixed or weak effects close this candidate route without changing
priors, selecting favorable rats, redefining MUA, or choosing a cell split.

The broad high-importance discovery goal is not achieved by a green technical
audit, an exploratory p value, or a methods-only result.
