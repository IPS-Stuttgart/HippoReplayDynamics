# Regional likelihood-mixture prevalence: frozen development screen

Frozen 2026-09-15 before new target estimates. This retains the original
matched-population Home-content problem. It is not a new decoder or a claim
that average posterior probability equals biological replay prevalence.

## Rationale and estimand

Mean uniform-prior posterior mass shrinks weak observations toward the spatial
prior. Instead estimate a regional mixture proportion from ALL individual
regional likelihood ratios, independently for each original population:

    log BF_j = log mean_{x in Home} L(counts_j | x)
             - log mean_{x outside Home} L(counts_j | x)
    p_hat = argmax_{0 <= p <= 1} sum_j w_j log(1-p + p BF_j).

This is an application of established maximum-likelihood prior adjustment /
mixture estimation, not a new statistical principle: Saerens, Latinne and
Decaestecker (2002), doi:10.1162/089976602753284446.
https://dipot.ulb.ac.be/dspace/bitstream/2013/68391/1/Decaestecker_NeuralComp02.pdf

Unlike the failed linear PACC correction, this retains the distribution of
likelihood ratios rather than only their mean. Unlike the separate regional
bounds experiment, it uses known spatial encoding likelihoods rather than
ternary population responses or latent-class conditional independence.

The target becomes a MODEL-IMPLIED aggregate Home proportion, not a verified
count of Home replays. Correct class-conditional likelihoods, including the
within-region spatial mixture, are essential assumptions. Agreement alone is
not validation. We do not force high/low populations to have a common estimate.

## Fixed sources and methods

Reuse the audited regional-prevalence-calibration-v2-20260914 count banks and
their unchanged four confirmed matched pairs: Rat1/Open1, Rat1/Open2,
Rat2/Open1, Rat4/Open2. Source output hashes and independent reconstruction
record must be checked. Shared cells retain identical source observations.

Primary observation likelihood: independent Poisson with the existing early-RUN
maps, fixed 20-ms observations, no temporal prior, no event removal or retiming.
The original full-RUN map is a REAL-only sensitivity. The existing independent
decoder posterior mass must be reproduced to 1e-9 before any interpretation.
Secondary conditional-on-total-spikes likelihood is explicitly exploratory;
its results cannot replace the primary after inspecting outcomes. Zero-spike
conditional observations have BF=1 and contribute no regional information.
Poisson silence retains the location-dependent absence-of-spiking likelihood.

Constrained boundary optima are reported as boundary, not concealed as reliable
zero/one estimates. If all BFs are one the result is nonidentified (NaN), not
the spatial prior or an artificially agreed estimate. Record information and
conditional profile-likelihood support width, but do NOT call this a calibrated
confidence interval: native observations are temporally clustered and weighted
synthetic panels are reused. The profile width is not a success gate.

## Truth evaluation and development gates

Native fourth-quarter RUN provides known positions. Simulations are the existing
independently generated matched Poisson gain1, gain4, conditional-total,
late-map-drift and 80%-shared-assembly banks. None of their labels enter fitting.
For each bank use the frozen prevalence panels 0.05, 0.15, 0.30, 0.50, 0.75,
weighting observations within each true class to construct known mixtures.
Report natural-prevalence native RUN separately. Panels are not independent
replicates. Average panels and populations within session, then sessions within
rat, then equally weight rats. Keep all missing/boundary/failure rows.

Advance only if ALL of the following primary Poisson tests pass:

1. All four sessions, both populations and all sources complete; source hashes
   intact and independent likelihood, optimizer and summary reconstruction pass.
2. Native RUN panel mean absolute prevalence error improves >=20%, is <=0.05,
   and does not worsen in any rat; natural-prevalence RUN cannot worsen.
3. Matched Poisson panel error improves >=20%, is <=0.05, with no rat worsened.
4. Gain, conditional-total, map-drift and shared-assembly truth errors do not
   worsen relative to unadjusted posterior mass, overall or in any rat.
5. Original absolute high/low Home discrepancy decreases >=20% for BOTH early
   and original full maps; no rat worsens. Previously accepted, fixed segment
   endpoints cannot worsen for either map.
6. Every primary REAL estimate is identified and strictly interior (no 0/1
   agreement accepted as a fix).

This is a strict DEVELOPMENT falsification screen, conditional on previously
selected populations and full-RUN eligibility. It is not an untouched external
test. Passing would justify freezing an independent-recording validation, not
completing the user goal. Failure is retained and does not authorize changing
likelihoods, thresholds, regions, cohorts or event times until the result passes.

## Artifacts and audit

Write event likelihood-ratio caches, session/panel estimates, equal-rat truth
summary, original-content contrast summary, gate summary, source/code manifest
and a concise markdown report. Independent audit reconstructs regional
likelihoods from cached counts/rates using separate direct probability code,
checks scalar optimization by a separate bounded optimizer and stationarity,
and independently recomputes the reported gate-relevant summaries. This new
audit complements, not replaces, the already completed native-clock/count audit.
