# Cross-fitted distribution-level composition recovery

Frozen before this experiment's outcomes. This is one exploratory measurement
successor, not a replacement or correction of the original real-data statistic.
No real neural data are rescored. Use existing known-track simulations only.

## Fixed estimator

For each session and cell split, learn class-conditional histograms of evaluation
track-2 posterior mass, with five fixed equal-width bins on [0,1] and Jeffreys
0.5 pseudocount per bin. Treat Poisson as the existing primary readout and
conditional-count as a separately reported sensitivity. No bin-count search.

Split original count anchors into five folds by fixed hash, balanced within
PRE/POST and ripple/MUA strata. Both true-track copies and every cell split of an
anchor stay in the same fold. Only other folds train the two templates. Require
at least five distinct nonempty training anchors per template/split. All epochs
may train templates: they are simulated known-track signals, not claims about
real PRE experience. Pooling the count regimes is an explicit assumption tested
below, not an established biological fact.

Estimate one bounded mixture proportion by maximizing the held-out weighted
categorical likelihood sum w log((1-pi)*f1(q) + pi*f2(q)). This is a mixture of
calibration distributions, NOT use of posterior odds as event likelihood ratios.
No true labels enter the mixture optimizer. Labels are used only for training
templates, defining controlled test mixtures and evaluating accuracy. Identical
templates or empty test sets return missing, never a default 0.5 success.

## Targets and weights

Primary: POST ripple count anchors in the two frozen strict RUN-pass sessions,
with true mixtures 0.25, 0.5 and 0.75. Also report 0.4/0.6 sensitivity, PRE,
MUA-only and all four diagnostic sessions. Only ordered copies are used, since
sequenceless shuffled copies would duplicate the observations.

Every original count anchor has total weight one. Divide that weight equally
among its nonempty test cell splits, then between the two true-track copies
according to the chosen known mixture. No spike-count weighting. Templates
require paired nonempty track readouts so missing spikes cannot change the true
test mixture. Report incomplete likelihood coverage separately.

Repeat the point estimates in fixed count strata (1-4 vs >=5 evaluation spikes)
and path-span strata (<100 vs >=100 cm), retaining the same trained templates.
These are calibration-domain shifts, not biologically motivated classification
thresholds. Their results are diagnostic, not a source of favorable re-selection.

Also report fits under the existing full, retained and half sequence-selection
masks. Start from the same anchor/split weights before applying the masks; do
not renormalize a retained anchor to erase its chance of selection. Report the
resulting true weighted track fraction and empty sets. The existing simulation
varies half-cell subsets by anchor rank; it does not exactly reproduce the real
fixed-half-population realization. This stress test cannot certify transport to
real replay.

## Uncertainty and advancement

For the primary POST ripple full-count/full-path targets, use 2,000 count-anchor
bootstrap draws. Keep every representation of an anchor together, preserve its
fold, and REFIT training histograms using each draw's anchor weights. Exclude
all copies of a held-out fold even when sampled multiple times. Intervals remain
conditional on the existing maps/spike simulations, not animal-population CIs.
Require >=95% finite bootstrap estimates before reporting a percentile interval.

A same-generator recovery gate requires complete point likelihood coverage,
finite intervals containing each true 0.25/0.5/0.75 mixture, and intervals for
0.25 and 0.75 lying respectively below and above 0.5. Report boundary fits and
failures. Both primary sessions must pass the existing Poisson-primary gate;
conditional-count cannot silently replace it. This is a demanding pilot
measurement criterion, not biological equivalence or proof of replay identity.

No gate here authorizes automatic real-data correction. Actual selected-event
calibration is sparse, and transport across latent path/count distributions must
be justified separately. If same-generator recovery fails, preserve that failure
and do not tune histograms or exclude the weaker animal using these outcomes.
