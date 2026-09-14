# Split-cell spatial-consistency diagnostic: frozen development screen

Frozen 2026-09-15 before new measurements. This is a diagnostic with explicit
retention, not a correction of all-event regional prevalence. The objective
still requires the original matched-population discrepancy and independent
recording validation; no success is declared on development data alone.

## Distinction from prior failed tests

Previous three-population features measured posterior concentration/agreement.
Previous cell-predictive context compared temporal filters and independent
decoding. This test uses no temporal context and asks whether held-out cell
identities in the SAME 20-ms bin support locations inferred from other cells.

For each original population separately, split its cells into three seeded folds.
For a fold, condition on total spike counts separately in training and held-out
cells. Define their spatial log likelihoods as:

    L_train(x) = sum_i n_i log(lambda_i(x)/sum_train lambda_i(x))
    L_test(x)  = sum_i n_i log(lambda_i(x)/sum_test lambda_i(x))
    gain = log mean_x exp(L_train(x)+L_test(x))
         - log mean_x exp(L_train(x)) - log mean_x exp(L_test(x)).

This is a shared-location versus independent-location predictive compatibility
contrast, NOT a temporal replay test, not a p-value and not a location-independent
firing-rate null. Conditioning removes firing-total evidence; it does not remove
all sampling or shared-assembly confounds. A fold with zero training or zero
held-out spikes has gain zero. An entirely silent bin never gains spatial
consistency from shared absence-of-spiking.

Population score = median of its three fold gains. Pair score = minimum of the
two population scores. Fold seed 20260915 with session and side identities,
frozen before scoring. No posterior mean, disagreement, Home label or known
position enters the score or fold assignment. Cells shared by the original
matched populations retain the same observations. Both populations contribute
to this OFFLINE diagnostic; do not call it an A-only prediction of held-out B.

## Fixed data, scoring, and retention

Use all four originally confirmed matched PF pairs and the independently
audited regional-prevalence-calibration-v2-20260914 banks. Keep the original
1,836 candidate times and 513 accepted-segment endpoints. Decode with unchanged
independent Poisson likelihoods, uniform spatial priors and early-RUN maps.
The original full-RUN maps are a real-only sensitivity. No posterior is modified.

Retain ceil(n/2) highest pair scores per session/source using a seeded hash of
observation index for ties. Do not add a count cutoff after viewing the outcome.
Comparison rankings at identical coverage: minimum population spike total,
and lowest maximum population entropy. All-event metrics are the exact
random-retention expectation, not a favorable random draw. Report retention
and score-tie fractions, including zeros. Half selection changes the analyzed
subset and can change its biological composition; it does not debias full-set
prevalence.

Known truth: native RUN-Q4 and the existing matched Poisson, gain4, conditional-
total, late-map-drift and 80%-shared-assembly banks. Report raw and CLASS-BALANCED
Home Brier error and physical mean-position error. Class-balanced metrics give
equal weight to Home/non-Home; removing a hard true region must not improve the
summary merely by changing its prevalence. Both true classes must retain at
least 20% of observations in every session/source. Truth is used only in testing,
never in scoring or retention.

## Required development gates

1. All sessions/sources/maps/observations present; baseline regional masses
   reproduce their source cache; raw observations, hashes and masks unchanged.
2. The early and full-map candidate high/low Home-mass gaps each decrease >=20%
   under score retention, and do not worsen in any rat. The already accepted
   segment endpoint gaps cannot worsen for either map. Equal session means
   within rat, then equal rat weights. Absolute SESSION gaps are averaged;
   signed cancellation across sessions cannot produce a success.
3. Real endpoint mean separation AND nine-tile posterior TV each decrease >=10%
   for early maps, and neither population's mean entropy increases. At least
   75% of retained rats must improve each disagreement metric.
4. On every known-truth source, both populations' class-balanced Home Brier
   errors and physical mean-position errors do not worsen overall or in any rat.
   Every source/session retains >=20% of both true classes.
5. Predictive diagnostic check: Spearman correlation between pair score and
   endpoint separation is negative overall (session-balanced, within-session
   ranks) and in at least 75% of rats on real candidates. Also report regional TV
   and known-position errors. No event is a substitute for an independent rat.
6. Source reconstruction, a separate direct probability reconstruction of fold
   gains/posteriors, deterministic selection, summaries and gates pass tests.

Failure is retained, with no retuned seed, coverage, likelihood or exclusions.
Even a pass is only a development screen. Freeze the unchanged diagnostic on
an independent recording cohort with adequate units and both truth classes
before claiming the requested validated remedy. Do not treat agreement with
unknown replay truth, simulation success alone, or selection of diffuse
posteriors as completion.
