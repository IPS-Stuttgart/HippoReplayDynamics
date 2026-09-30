# Regional content calibration and discrimination diagnostic

Frozen before new results, 2026-09-15; seed 2026091501. This implements the first
two diagnostic proposals, not a new replay estimator or a claim of impossibility.
The subsequent cell-participation fitting and distance-to-Home arguments remain
untested. Shared activity is not necessarily a scalar gain; held-out prediction
alone cannot validate spatial truth. Low AUC of a chosen statistic is not a
universal identifiability bound. AUC and Bayes classification error are distinct.

## Unchanged inputs

Eight PF sessions, four animals; first-half-only maps and cell QC from the frozen
20260914 experiment, common valid grid, inferred 20-cm Home disc. Keep all 200
candidate endpoints per session at their last complete 20 ms. No trimming,
trajectory selection, rate-floor tuning, or fit to replay content. Exact inputs,
seeds, cell IDs, counts, and readouts are retained with hashes.

## Expanded known-position calibration

Use every nonoverlapping 20-ms window in the second RUN half with complete
tracking support, bracketing position samples at least 10 and at most 200 cm/s,
no tracking gap over 100 ms, and full containment in a supported RUN interval.
Position is the average of 20 interpolated 1-ms midpoints. A supported running
bout ends at an eligibility gap over one second, bridging brief speed/tracking
and region-boundary exclusions; alternate whole bouts between calibration and
validation, not adjacent windows from the same bout. Report window AND bout
counts by Home class. The Home disc boundary must not be crossed within a
window; those boundary-straddling windows are reported and excluded from this
binary known-position control, not from real replay endpoints.

Binomial thinning: draw a desired total from the frozen real endpoint pool,
set p=min(1, desired/observed RUN total), independently retain each spike with p.
Zeros remain zero; never add spikes. Save p, desired and achieved totals and the
fraction of unattainable requests. This is approximate count matching, not exact
matching. Compare native, thinned/frozen-decoder, and thinned/exposure-adjusted
readouts. The latter uses tau*p in the silence term. Because p depends on observed
counts, even exposure correction is a diagnostic, not exact generative inference.

For full-population session prevalence, use three frozen regional BF categories
(<1/3, neutral, >3; silent windows neutral). Fit a one-dimensional mixture of
empirical class-conditional category probabilities, not a latent-class agreement
model. Bootstrap whole RUN bouts and 30-s real-event blocks, 200 repeats. Build
marginal percentile compatibility constraints from those bootstraps and report
LP prevalence intervals at transfer slack 0, .05, .10, 1. These are assumption-
indexed compatibility/sensitivity intervals, NOT guaranteed 95% biological CIs.
Report mixture fit residual and held-out RUN truth; incompatible sets stay failed.
Missing calibration classes return [0,1], not invented emissions. Utility width
<=.20 is a prospective descriptive convention. No fit uses target labels.

## Discrimination frontier

At fixed region truth (positions uniform over Home or complement), compare:
1. Exact independent Poisson generator with shared known session gain, calibrated
   to the full population's native endpoint mean count under balanced classes.
   Matched Poisson readout uses that gain; the frozen gain-one readout is separate.
2. Observed-total-conditioned multinomial generator, totals sampled independently
   of truth from the SAME population's endpoint counts. Matched likelihood is
   n log(lambda) - N log(sum lambda). Also report the frozen Poisson decoder on
   those counts, explicitly mismatched. This corrects the earlier count-conditioned
   generator versus independent-Poisson distinction.

Generate 512 independent test windows per class, spike multipliers 1,2,4,8,
full population, coverage-rich/poor halves, remove half/all Home-peak cells,
and seeded size-matched random removals. Home-peak means the RUN map's peak lies
inside the disc, not an inferred replay field. Original targeted matched cell
lists are included separately after intersection with current training-only QC;
record lost IDs and do not call them the unchanged original matched populations.
Every variant uses the common grid. No sorting or selection by target scores.

Report AUC, error of the zero log-BF decision (balanced classes), Brier/log loss
at the known balanced class prior, silence, spikes and active cells. For the exact
matched likelihood this threshold is the generator's balanced Bayes classifier;
for a mismatched readout it is only a fixed-rule error. Even matched performance
is a specified synthetic best case, not real replay truth.

Separate full-population terminal segments 40/100 ms use their own eligible
observed terminal-count pools, multiplier 1, constant synthetic region throughout.
Report short-event exclusions and do not relabel segment content as the original
20-ms endpoint. Replicate seeds are computational repeats, never biological units.

Stop before replay-specific encoding correction unless these diagnostics warrant
it. No latent-class rescue, hc11 confirmation, speed claim, or universal no-go.

