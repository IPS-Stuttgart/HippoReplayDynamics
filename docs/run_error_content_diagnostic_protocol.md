# Known-location-trained content reliability diagnostic

Frozen 2026-09-14 before fitting or applying this diagnostic. This follows FAILED
agreement-targeted prediction and regional prevalence calibration. No old test is
reinterpreted as successful. Tanni recordings are external to training but have
already been examined under other failed methods; this is external development
validation, not a pristine confirmatory dataset.

## Question

Can a predictor trained on actual RUN localization error select readouts that
are both more accurate and less sensitive to observed cell population in an
independent dataset? Previous prediction used A/B agreement as its label, which
favored broad inaccurate posteriors. Here the label is the normalized true
position error of population B, using third-quarter RUN only in PF.

## Fixed observations and predictors

Reuse population-content-stability-20260914 fixed disjoint cell halves, original
candidate endpoints, independent Poisson flat-prior 20-ms decoding, and its
matched/mismatched conditional known-path banks. No trajectory selection, map
refitting, time-window changes, new events, or temporal smoothing.

First-half RUN maps. Original full-RUN unit eligibility and grid definition
remain conditioning limitations. Simulations are conditional on observed
per-population spike counts, and do not reproduce all biological correlations.

Only A replay observations may enter predictors. B RUN-derived encoding is
allowed; B replay spikes, posterior, coordinates and agreement are excluded.
The full feature set contains A spike/active counts, active fraction, entropy,
posterior width relative to arena diagonal, peak mass, cell count, A local
relative summed firing rate, normalized arena size, nearest supported-bin
distance, B local relative summed rate at A's decoded position, and A/B local
rate-code gradients. The latter is the mean across adjacent 8-cm supported-grid
neighbors of sum_cells((sqrt(rate_x)-sqrt(rate_y))^2)/distance^2, sampled at the
supported bin nearest A's mean. It is a map sensitivity feature, NOT an estimate
of biological neural-sheet speed or a calibrated information bound.

Models fixed in advance: constant, spikes+entropy Ridge baseline, and full Ridge.
Alpha=10, sample weights equalize animals and sessions, PF-only medians/scales,
missing-value indicators. Target log(1+B_error_cm/arena_diagonal_cm). Predictions
lower-bounded at zero. No external feature/threshold/alpha selection.

PF training uses only primary split0 third-quarter RUN. Leave-one-rat-out PF
predictions are development checks, with all preprocessing fit inside folds.
Freeze all coefficients before opening Tanni through this diagnostic. PF
fourth-quarter RUN/replay sensitivity uses the respective held-out-rat model.
Apply the frozen all-PF model to all 25 Tanni sessions / five animals. External
RUN error, simulated truth and B replay outcomes never train the predictor.

## Selection and readouts

Primary policy: retain lowest predicted B error 50% per session/source/split/draw,
ceil(n/2), deterministic event-ID tie breaking. Split0 primary; splits1/2 remain
sensitivity, not replacement or extra independent animals. Comparison policies:
spikes+entropy predicted error, lowest A entropy, highest A spike count. The
unfiltered mean is the exact expectation under random fixed-size retention.

Report external forecast log-MSE/correlation against known B RUN error, real
endpoint separation and nine-tile posterior TV, A/B entropy and width, and A/B
known-position errors in RUN and both simulated sources. Also report retention
by known true spatial tile and errors relative to a reference reweighted to the
retained true-tile distribution, so merely selecting easy locations is visible.
Tile truth is a validation-only variable, never a predictor/real selection input.

## Predeclared external gates

- Complete fixed external cohort, input hashes and numerical reconstruction.
- On native held-out RUN, full predictor log-MSE improves >=5% versus constant
  and >=5% versus spikes+entropy; risk/error correlation positive in >=4/5 rats.
- At 50% retention, real regional TV and endpoint separation each decrease >=10%
  relative to random expectation; positive reductions in >=4/5 rats and lower
  bounds of descriptive equal-animal bootstrap CIs above zero (2,000 resamples).
- Mean A/B entropy does not increase; neither side increases by >0.01.
- Mean A/B true error does not worsen on RUN or either simulator; neither side
  worsens by >2 cm. Relative to retained-location-matched references, mean A/B
  error also must not worsen in all three known-truth sources.
- Neither missing truth/error values nor failed animals can be silently removed.

All gates are required for this external screen. A failed screen is not fixed by
choosing another retention, split, source, predictor or animal. Passing would
support a conditional diagnostic for retained content, not representative
replay-Home prevalence, proof of replay truth, or correction of every event.
Before declaring the full user objective achieved, also audit transfer back to
the original matched-population contrast and independent-recording scope; do not
replace that question with only an easier agreement metric.
