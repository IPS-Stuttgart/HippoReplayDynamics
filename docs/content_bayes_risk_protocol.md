# Accuracy cost of posterior-only agreement adjustments

Frozen 2026-09-15 before scoring. This is a scope/feasibility diagnostic, NOT a
replacement for the active independent-data remedy goal or its accuracy gates.

## Why check this now

Multiple posterior adjustments have reduced disagreement while worsening known
position risks. The finite-cell exchange also failed its accuracy gates, but
exchanges change the observations and are NOT covered by the fixed-information
argument below. Their failure is not an impossibility proof.

Audit whether the gain-1 Poisson benchmark is actually model-correct for the
baseline. If it is, a standard squared-loss identity supplies a necessary
constraint on any correction that only changes the posterior from those spikes.
Do not treat agreement obtained by making posteriors uninformative as success.

## Standard identity and exact scope

Let Y indicate the Home region, N the spikes available to one population, and
q(N) = E[Y | N] under the true joint distribution. For ANY measurable adjustment
a(N) using those same observations (or independent randomization),

    E[(Y - a)^2] - E[(Y - q)^2] = E[(a - q)^2].

Proof: expand Y-a = (Y-q) + (q-a). The cross term vanishes because
E[Y-q | N] = 0. For independent randomization, condition on N and that
randomization. Thus a nontrivial change has strictly positive expected Brier
cost. Requiring both true-class conditional Brier risks to be nonworse implies
nonworse total Brier for any positive class prior, so equality requires a=q
almost surely. This does not hold for arbitrary encodings on real replay,
unmatched priors, data-dependent reused calibration labels, or changed neural
observations. The present adjustments are fixed before this audit, not fitted
to its labels. Independent external data would still be needed for a remedy.

This is established decision theory, not a new theorem. References:
- Bayes estimation and conditional squared-risk decomposition:
  https://stat210a.berkeley.edu/fall-2025/reader/bayes-estimation.html
- On loss functions which minimize to conditional expected values and
  posterior probabilities: https://authors.library.caltech.edu/records/59wem-6hd53

Here every audited Poisson rate is strictly positive. The uniform-bin Poisson
mixture therefore gives positive probability to EVERY finite nonnegative
integer count vector. On this countable support, a=q almost surely means a=q
at every count vector. Thus a fixed same-observation rule cannot promise exact
expected no-harm on this model-correct benchmark and still change the Home
probability on a special collection of replay-like count vectors. This stronger
corollary does not apply when added observations change the information, or
when uncertain/misspecified encoding makes the baseline non-Bayes. It is an
expectation-level statement, not a proof that
every finite-sample safeguard must reject a change. Rare changed vectors can
be absent in a finite bank, hiding a strictly positive expected cost.

For two populations, write C_s = E[(a_s-q_s)^2] and
D = ||q_high-q_low||_2, D_new = ||a_high-a_low||_2. The L2 triangle inequality
gives D_new >= max(0, D-sqrt(C_high)-sqrt(C_low)). Consequently reducing RMS
Home-probability disagreement by r requires C_high+C_low >= r^2 D^2 / 2.
Both populations must retain their original information for C_s to be the
corresponding Bayes Brier cost. The inequality itself is numerical algebra
under any common nonnegative weighting. It does not refer to the original
replay mean Home gap, class-conditional J, regional TV or physical position
error and must not replace those study endpoints.

## Audited inputs and prior correction

Use only original encoding.npz and cal_poisson_gain1.npz from
regional-prevalence-calibration-v2-20260914 for Rat1/Open1, Rat1/Open2,
Rat2/Open1 and Rat4/Open2. Verify the audited exchange reference input hashes.
Reconstruct every calibration position and count from its frozen generator:
20260914|regional_prevalence|SESSION|cal_poisson_gain1. Each class contains
2,000 uniform-within-class position draws; every cell uses Poisson(0.02*rate)
with early_run rates, NOT full_run or gain-4 rates. Verify truth positions,
region labels, totals and the cached baseline Home probabilities.

The simulation sampling prior is 50/50 true classes; the decoder prior is
uniform across bins. Let pi be the fraction of supported bins inside Home.
Use class weights 2*pi and 2*(1-pi) for observed Brier and paired L2 estimates.
The marginal count-density ratio between uniform-bin and balanced-class draws
is r(N)=2/[q(N)/pi + (1-q(N))/(1-pi)]. For each observed N, conditional Bayes
risk is q(1-q); adjusted risk is q(1-a)^2 + (1-q)a^2. Their difference is
(a-q)^2 exactly. Average r(N)*(a-q)^2 to obtain a Rao-Blackwellized Monte Carlo
estimate of the expected cost under the uniform spatial prior. This integration
is NOT exact population integration on the real rate maps.

Also report weighted observed Brier differences, squared adjustments, the
finite-sample cross term, and its stratified standard error. The sample cross
term need not be zero; small empirical nonworsening is not a contradiction of
the expected-risk identity. Do not decide significance or select corrections
from these already inspected development banks.

## Fixed adjustments and checks

- Baseline unchanged posterior.
- Temperature 2 on spatial log likelihood, then compute Home mass.
- A half-mixture of the spatial posterior with the uniform spatial prior.

All populations retain exactly the same original spikes for these adjustments.
No threshold fitting, row/event dropping, cell exchange or phase pooling.
Reconstruct baseline likelihood independently by summing full single-cell
Poisson log probabilities including factorials; compare every posterior.

Numerical positive/negative controls enumerate 441 joint count vectors for
two Poisson neurons and four equally likely locations. Bound the omitted tail
below 1e-12. A half-mixture must worsen expected Brier; an extra shared neuron
must improve expected Brier and remove inter-population disagreement in this
toy. The latter deliberately violates the same-information premise and is NOT
a proposed remedy. Test incorrect prior and conditional class weighting too.

## Outputs and decision

Write accuracy_costs.csv, agreement_cost_bounds.csv, exact_toy.csv,
four compact probability CSVs, gates.csv, pre_analysis.json, report.md and
manifest.json. Record source/code hashes and limitations. Run on gpuserver6000
under a detached service; do not read/score native Q3/Q4, test banks or replay.
Inherited provenance hashes may verify those files without decoding them.

Numerical success establishes the scoped accuracy constraint and verifies its
benchmark assumptions. It does NOT complete the objective. It redirects future
work from universal fixed-information posterior repair toward genuinely better
encoding/measurement or a diagnostic validated with independent observations.
Neither the prior exchange search nor all possible remedies are ruled out.
