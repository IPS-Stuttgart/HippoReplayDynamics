# Nested Candidate-Panel Size and Speed Recoverability

Status: frozen design before generating this experiment's observations. This
follow-up was motivated by the completed small-panel calibration results, not
prespecified before those results. It changes candidate count, not thresholds.

## Question and Scope

Do missing selected-event statistics primarily reflect small candidate panels,
and does collecting more candidates restore informative speed-gradient inference
under the existing surrogate models? Recovery of a finite statistic alone is
not sufficient: report calibration, interval width, power, false equivalence
and abstention at every candidate budget.

Use all 33 audited source sessions (PF: eight/four animals; Tanni: 25/five).
Reuse the frozen RUN A/B encoding/generator arrays and the 11-30 source duration
profiles per session from the speed-identifiability experiment. These duration
profiles come from candidates, not selected decoded trajectories. Do not select
sessions or durations based on previous decoding success.

## Paired Simulation

- Candidate budgets: **30, 100, 300**. Generate 300 candidates once for each
  session/draw/condition; smaller budgets are strict prefixes. This is a finite
  sensitivity range, not a declaration that 300 suffices for every population.
- Within each draw, repeatedly permute all source duration profiles in balanced
  cycles until the maximum budget is reached. Record the profile ID and duration
  for every synthetic ordinal. Reusing a duration MUST NOT reuse a path or spike
  observation: independent substream seeds include the synthetic ordinal.
- Paired A/B and Poisson/shared-gain observations use identical latent paths.
  Budgets share identical observations for shared ordinals. New master seed
  **20260917**; old simulated panels are not reused as new observations.
- Preserve the existing continuous-path generator, 1 ms latent/spike bins, rate multiplier
  3, mean base speed 1000 cm/s, and normalized horizontal coordinate q. Truth is
  v=1000*(1+g*q) cm/s. This is NOT a physical wall-distance gradient and does
  not establish biological uniformity or real replay ground truth.
- Same schedule: 40 fit and 99 residual-calibration panels with g uniform in
  [-0.75,0.75]; 100 new uniform-g test panels plus 20 test panels at each of
  g=-0.50,-0.25,0,+0.25,+0.50. Fit/calibration use only A/Poisson. Tests cross
  A/B maps and Poisson/shared gain. No B/gain/test observations enter fitting.

## Frozen Decoding and Inference

Independent flat-prior Poisson decoding; no motion prior. Retain MAP and
posterior mean, unfiltered and >=2 cells/3 spikes, all supported steps and selected
continuous cores. Preserve the current 20 ms windows/5 ms stride, continuity
rule, every-four-frame speed steps, no unsupported-bin bridging, equal-event
moment statistic, minimum five contributing events and minimum q variance 0.01.
The primary readout remains bin-supported posterior mean in selected cores.

Fit the same inverse Gaussian and inverse-conformal baselines separately at
each candidate budget, with disjoint fit/calibration/test draws. Include the
unchanged event-bootstrap baseline. Missing calibration statistics remain
infinite residuals; failed fits/statistics yield unbounded abstention. Never
drop failures from all-panel denominators or clip intervals to the truth range.
Keep strict +/-0.25 equivalence and 0.10/0.50 sensitivity bands. Report both
all-panel and finite-only coverage, with finite availability and widths.

Also repeat the frozen excluded-animal pooled inverse calibration separately
at every budget. Exclude all sessions/panels from that animal from fitting and
residual calibration. Its RUN maps remain available for decoding. This remains
retrospective, conditional simulated-population transfer, not a new-animal
conformal guarantee. No choice of budget/model is promoted after seeing results.

## Reporting and Audit

Persist per-event moments and support/continuity counts, profile membership,
per-budget path/count hashes, complete panel statistics, fits, decisions and
session summaries. Reconstruct ALL budget statistics from persisted event
moments; independently verify at least one fit/calibration/test-uniform and
every fixed-g draw per session from freshly regenerated paths/counts. Check
prefix identities, fresh repeated-profile observations, all interval endpoints,
boolean decisions and denominators. A technical audit does not validate the
generative model as biology.

Aggregate sessions within animals, then give equal-animal summaries and paired
budget differences (100-30 and 300-30). Use 5000 animal bootstrap draws, seed
20260918. There are four/five animals, not thousands of independent biological
replicates. Budget comparisons are paired; leave-one-out training sets overlap;
bootstrap intervals are descriptive conditional on the fitted baselines.

Primary figures: finite selected-statistic availability vs candidate budget;
coverage, finite output and interval width vs budget; known-gradient recovery
and equivalence/power vs budget. Show all animals and no-success sessions.
Interpretation must distinguish three outcomes: more candidates restore both
availability and informative calibrated recovery; restore availability only;
or remain insufficient within the tested budget. None licenses a real-data
constant-speed claim without a separate appropriately validated analysis.

This experiment closes the candidate-budget gap, not all remaining paper work.
The final integrated methods, limitations, artifact registry and claim audit
remain required. No biological replay rescoring or threshold tuning is allowed.
