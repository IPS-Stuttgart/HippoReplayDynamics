# Regional content: assumption-indexed identification bounds

Frozen 2026-09-14 before the new outputs; seed 2026091451. This is a feasibility
experiment, not a calibrated biological truth or speed-uniformity claim.

## Question and cohort

Can disjoint population readouts constrain regional content without assuming
their errors are independent? Primary real cohort: the eight PF recordings in
the frozen edge-support benchmark, unchanged last-complete 20-ms windows (200
per recording). Home is the previously inferred 20-cm disc on the actual common
valid grid. No endpoint trimming, reselection or new replay labels. Report all
recordings, including missing calibration classes and empty feasible sets.

Stage 0 separately reuses the four confirmed targeted matched populations,
both frozen endpoint cohorts and encodings. These populations can overlap and
are NOT the disjoint views used for latent-class fitting. Audit event-level
posterior odds/prior odds, not mean posterior divided by an approximate area.

## Readouts and bounds

Use existing training-only RUN mean rate maps, floor 1e-4 Hz, independent Poisson
decoding and flat common spatial prior. Form four disjoint groups by sorted
RUN-only Home/whole-grid mean-rate ratio (cell ID breaks ties). Group sizes
differ by at most one; retain every eligible cell. Ternary readout: regional
log Bayes factor above log(3), below -log(3), otherwise neutral. Zero spikes
are explicitly neutral for this detector, even though Poisson silence can have
spatial evidence; preserve the raw posterior and silence flag separately.

Let w[z,j] be the joint probability of binary Home truth z and the complete
four-view ternary response pattern j (81 possibilities). No factorization of
w is assumed. Fit constraints, not a preferred point estimate:

- w >= 0 and sum(w)=1;
- observed pattern probabilities have simultaneous Clopper-Pearson bounds;
- each class/view/readout probability has calibration lower/upper bounds.

The latter constraints are linear after multiplying by class prevalence.
Minimize/maximize sum_j w[1,j] to obtain prevalence bounds. Conditional
P(z=1|pattern) bounds use a Charnes-Cooper linear-fractional transformation.
Bounds are sharp for this specified polytope, not for all imaginable models.
Unrestricted calibration gives [0,1], an essential negative control.

Nominal per-analysis error budget is .05: .025 for 81 simultaneous observed
pattern intervals, .025 for the 24 calibration entries. Bonferroni is used;
no conditional independence BETWEEN VIEWS is required. Independent sampled
windows and calibration-transfer assumptions ARE required for a population
coverage interpretation. Real MUA/RUN windows are temporally dependent: their
outputs are model-based compatibility bounds, NOT certified 95% biological CIs.
There is no across-session/method multiplicity or animal-superpopulation claim.

Transfer slack epsilon=0,.05,.10,.20,1 widens every conditional calibration
probability. Epsilon=1 removes calibration completely. Empty sets are reported
as incompatible, never repaired by changing thresholds. A useful prevalence
interval is prospectively width <= .20; useful event bounds exclude .5 (all
event widths also reported). These are diagnostic utility conventions, not
biologically established thresholds. Report continuous widths as primary.

## Validation and controls

For each source session, independent synthetic calibration has 2000 examples
per truth class, truth positions uniform over states inside/outside Home.
Count distributions come from the fixed real full-population endpoint counts.
Conditional multinomial spike allocation uses the RUN maps at the true state;
decoding stays the frozen Poisson decoder. Thus event gain and total counts are
conditioned upon; no claim of a full biological spike generator is made.

Twelve fresh target panels of 512 windows per session/condition, prevalence .30:
native counts; 8x counts with matching calibration; 8x counts with quarter-2
training-map drift; 8x counts with 90% of expected cell allocation supplied by
a shared, independently drawn nonspatial state (Home frequency .70). The last
condition preserves a single spatial truth but breaks its correspondence with
shared population activation. Reuse 8x matched calibration for both stresses.
Calibration is reused across target replicates: coverage frequencies are a
conditional stress diagnostic, not 12 independent calibration experiments.
Target class labels never fit the bounds or latent-class estimates.

Compare a two-class conditional-independent categorical EM (eight fixed
initializations, up to 1000 iterations) and a held-out prevalence-only score.
Orient EM labels by fitted Home-positive emission, not target truth. Record
convergence and multi-start prevalence spread, not a false-confidence CI.
Assess Brier/log scores against actual simulation truth. Good fit/agreement is
not evidence that the latent class is spatial truth. No monotone coverage slope
is imposed as an alleged mechanistic finding.

Known-position observed RUN: first half of the frozen held-out RUN-Q4 windows
calibrates, second half evaluates, no new map fitting. This limited sample is
not spike-count matched to replay and is labeled accordingly. Also use those
RUN calibration intervals on real endpoints only as transfer sensitivity.
Insufficient Home examples must yield broad bounds rather than fabricated
sensitivity. Synthetic-calibrated real bounds are a separate model-dependent
readout, not validation that synthetic assumptions hold during replay.

## Verification and stopping

Tests: normalized ternary patterns, odds formula, silence policy, disjoint
partitions, LP sharp extrema on analytic toys, no-assumption [0,1], dependence
allowed, empty-set handling, monotone widening with epsilon, conditional
fractional bounds, label switching, and no target-label inference inputs.
Independent auditor reconstructs real counts/readouts, calibration intervals,
and samples saved LP constraints with an independent optimizer formulation.
All outputs carry code/input hashes and unchanged-window identity. Figures
must be visually checked. If informative bounds require untenable transfer
assumptions, report that limitation, not a remedy, and do not scale to hc11 as
confirmation. Its previously failed fixed-endpoint activity screen remains.
