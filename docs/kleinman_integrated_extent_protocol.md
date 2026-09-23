# Direct integrated-likelihood extent calibration

Frozen before scoring this estimator. This replaces neither the preceding failed
screen nor its archived protocol. No real replay or reward contrast in this run.

## Question

Does a direct spike-composition likelihood recover path extent more faithfully
than differences of independently decoded positions, while remaining insensitive
to global firing gain and event duration? This is a prerequisite for a biological
reward-content contrast, not a new biological result or a proposed methods paper.

## Data reuse

Reuse all 17,280 synthetic events from `kleinman-content-calibration-20260923`:
the same six first-chronological RUN-pass maps, conditions, event IDs and seeds.
Reconstruct their exact 1-ms Poisson counts and verify total/active-cell counts
against the original output. No new favorable seed bank or selected sessions.
Known map is not known trajectory. No latent truth is passed to the fitter.

## Forward model

- Unknown start and end positions independently take nine fractions of the
  inter-visit-threshold interval: 0, 1/8, ..., 1. Unknown directional map: both.
- Moving templates have linear or cosine timing, not pause-step timing. Static
  templates are included once per location/direction. This yields 306 templates
  before map-support exclusions. Do not constrain origin to the animal's end or
  direction to its incoming RUN direction.
- Integrate each template's known cell rates at 1-ms resolution into disjoint
  10-ms observation bins. Unlike the failed 40-ms estimator, a bin need not
  contain one spatial state. Templates require >=95% supported latent states.
- Given the total spike count in each 10-ms bin, use the multinomial likelihood
  of cell identities: q_i = integrated_rate_i / sum_j integrated_rate_j.
  Constants shared across templates are dropped. This removes a uniform global
  gain without knowing it. It does NOT remove cell-specific rate changes or
  time-varying gain within a 10-ms bin.
- Main estimate: extent of the maximum conditional-likelihood template, expressed
  as fraction of inter-threshold span. Average parameter values over numerical
  ties within 1e-10. No geometric continuity trimming, trajectory-selection gate,
  HMM or Bayesian motion prior. This IS a restricted monotone-trajectory model;
  it cannot establish arbitrary replay content or constant-speed biology.
- Report flat-template likelihood-weighted extent as a sensitivity diagnostic,
  not a second chance to pass the main screen.

## Predictive check

Separately fit best moving and static templates to alternating disjoint 10-ms
bins. Score the other bins, swap and sum. Report moving-minus-static conditional
held-out log score. Never fit with test-bin spikes. This is within-event
cross-fitting, not held-out neurons or an independent event replication.
This score does not select events for the primary extent calibration.

## Fixed screens

Primary matched-timing screen: 200 ms, expected 48 and 96 spikes, static plus
linear/cosine paths with true extents 25/50/75% of span. Each of 12 animal/end
strata needs all expected conditions/repeats, finite outputs, median absolute
extent error <=10% of span, absolute median signed error <=10% in every condition,
median long-path extent above short-path extent for each timing/count, median
extent shift between 48 and 96 expected spikes <=10% for each true shape, and
static false moving-extent estimates (>10% span) <=10% at each count.

Mandatory stress screens, reported separately:

1. Same error/count/static checks for pause-step moving paths (timing absent from
   the fitted library), still 200 ms, 48/96 expected spikes.
2. Duration-only effect: maximum range of median estimated extent across
   100/200/400 ms <=10% of span for every shape, count 48/96, animal/end.
3. All count-24 results remain visible; no unannounced low-count exclusions.

Global readiness requires every animal/end pass matched timing, unseen timing
and duration-only screens. Per-stratum results remain visible. Engineering
tolerances are not biological equivalence limits. An oracle-like matched-bank
success alone does not validate all real candidate classes or map drift.

No threshold change, increased repeats, strongest-session replacement or reward
unblinding after results. If the screen fails, do not infer a biological null;
record exactly which nuisance/stratum fails. Any different endpoint needs a new
justification and protocol, not a silent replacement of this one.

## Deliverables

Per-event estimates and predictive scores; per-condition and animal/end screen
tables; calibration figure; immutable input/output hashes and code commit;
independent likelihood checks; compact paper archive. Run on gpuserver4090 with
tmux and durable terminal records. No real replay or reward effect is scored.
