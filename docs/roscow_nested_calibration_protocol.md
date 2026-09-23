# Roscow task-arm probability calibration: bounded development check

Frozen 2026-09-23 before inspecting calibrated results. The preceding arm-readout
experiment found above-null ranks in two animals but poor probability scores in
all three. This follow-up is exploratory development on the SAME task cohort,
not a previously unseen replication. Outer test trials are excluded from fitting;
the cohort as a whole has already informed the choice to investigate calibration.

## Keep the scientific target and old failure intact

The intended biological endpoint concerns represented sequence content after
outcomes. This check does not test it, does not change the original three-animal
accuracy gate, does not add trials, and cannot validate missing trial-cycle or
spatial anchors. Do not use calibrated task probabilities as rest replay labels
without further validation. No reward/surprise or PRE/POST contrast is opened.

## Fixed procedure

Use the previously frozen 338 trial count vectors / 17 sessions, exactly the same
CA1 units, pre-arrival interval and flat arm prior. No cell/window selection.
Keep one-second training-rate shrinkage and the original three readouts.

For each outer held-out trial:

1. Remove that trial completely.
2. Within the remaining trials, leave each trial out and fit rates using only the
   inner training trials, including the pooled-rate shrinkage prior.
3. Select a scalar temperature using inner true-arm mean log score from the grid
   1, 2, 4, 8, 16, 32, 64, 128, infinity (uniform). Within 1e-12 ties prefer the
   larger temperature. This is a fixed uncertainty adjustment, not a model search.
4. Refit rates to all outer-training trials, apply the selected temperature to
   the outer test trial's likelihood logits, and record its probability/log score.
   Never use that trial's label or spike vector to select temperature.

Temperature scaling follows the standard scalar-logit construction discussed in
Guo et al. 2017 (https://proceedings.mlr.press/v70/guo17a.html); this is not a new
method and does not by itself establish calibrated probabilities.

Include uniform as a legitimate selected result: training data may not support
useful probabilities. Positive finite temperature does not alter arm rankings;
the uniform option discards a prediction rather than creating information.
Compute likelihoods/log scores directly in log space to avoid underflow.

Refit this entire nested procedure for every circular label shift. Session-equal
animal summaries use 10,000 null draws (independent session shifts, including
identity), seed 20260923. Use true-arm log score above uniform as the primary
calibration diagnostic; accuracy and Brier score are secondary diagnostics.
These are development screens, not biological significance tests.

## Decision

Report every animal/readout. A useful-probability screen requires mean session
log-score improvement over uniform > 0 AND above the animal's 95th shift-null
percentile, for Poisson and composition separately. Do not pool animals to hide
a failure. Report uniform-selection frequency and temperature distribution.
Even a pass cannot replace the failed original cohort screen or authorize rest
scoring. If the bounded check fails, preserve it; do not expand the grid or tune
windows to obtain a desired result. Reassess the biological assay/data instead.
