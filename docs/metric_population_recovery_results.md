# Population Metric Recovery: Results And Decision

2026-09-09. This is a simulation-based measurement assessment, not evidence that
real replay follows either generating mechanism. The broad high-importance
discovery objective remains unmet.

## Provenance

- Worktree: `/home/florianpfaff/HippoReplayDynamics-metric-population-recovery`.
- Frozen producer: `201a45c7e7f78dd6d708bba887e583e4dd077cea`.
- Passing verifier: `49f60e1f17728150745b9a2ce3f5354957317823`.
- Run: `/mnt/seagate10tb/florianpfaff/metric-population-recovery-all33-20260909`.
- Passing audit: same prefix plus `-audit-v2/metric_population_audit.json`.
- Final report: same prefix plus `-report-v2/metric_population_report.md`.
- Protocol: [metric_population_recovery_protocol.md](metric_population_recovery_protocol.md).
- Antecedent: [event-level oracle recovery](metric_oracle_recovery_results.md).

There are 633,600 freshly sampled events using 33 frozen RUN encoders and native
count-profile libraries, across nine source animals. Each of 50 independent
simulation replicates has a known neural fraction of .25, .50 or .75 among
moving events. Moving mass is .60; stationary and independent-position mass
are .20 each. Fitting estimates all four proportions without seeing these
generator labels. The first 32 events per recording form a smaller nested
cohort; 128 per recording is the primary cohort.

The audit independently regenerated every path, generator label and all 6,600
count arrays, checked 7,603,200 likelihoods with a backward recursion, performed
594 library crosschecks and certified all 1,800 mixture optima and their profile
interval boundaries. Maximum score error was `9.094947017729282e-13` nats.
The first audit failed because its metadata join dropped `simulation_index`;
the verifier was fixed and an end-to-end regression added. The failed log is
preserved. No simulated data or producer scores were changed.

## Primary Results

Each PF population contains 1,024 events; each Tanni population contains 3,200.
Power here means correctly rejecting an equal physical/neural mixture when the
true neural fraction is .25 or .75. Coverage is for nominal 95% profile intervals.

| Dataset | Condition | Largest absolute bias | Lowest coverage | Equal-mixture false direction | Power, physical / neural enriched |
|---|---|---:|---:|---:|---:|
| Pfeiffer/Foster | Exact generating model | .023 | 92% | 2% | 28% / 22% |
| Pfeiffer/Foster | Training-cell geometry | .118 | 86% | 8% | 38% / 10% |
| Pfeiffer/Foster | Unmodeled gain drift | .055 | 90% | 10% | 24% / 18% |
| Tanni | Exact generating model | .052 | 96% | 0% | 52% / 40% |
| Tanni | Training-cell geometry | .118 | 84% | 12% | 70% / 24% |
| Tanni | Unmodeled gain drift | .036 | 92% | 8% | 52% / 34% |

All practical screens fail, including the smaller cohorts. No threshold was
changed. Exact-model interval coverage is broadly compatible with its nominal
target; the primary failure is insufficient directional power. Median interval
widths span approximately .62-.69 in PF and .49-.58 in Tanni under the exact
model. These are broad relative to the .25 departure from the equal mixture.

The estimated-geometry condition systematically underestimates neural mixture
mass in several scenarios and reduces coverage. The drift condition has higher
observed false-direction fractions than the exact condition, but 50 replicates
give wide Monte Carlo intervals: these differences alone are not a precise
population-level rate-inflation estimate. For example, 10% is 5/50, with a Wilson
interval of approximately 4.3%-21.4%; 0/50 still allows up to approximately 7.1%.

The maximum recorded mixture KKT error is below `1.75e-6`. The large estimation
uncertainty is not explained by an optimizer failing to find its optimum.

## Interpretation

Soft likelihood aggregation retains more information than hard class counts,
but it does not make this particular short-event contrast reliably recoverable
at the tested cohort sizes. This does not prove equality, a universal lack of
information, or that a larger independently sampled cohort could never work.
It also does not test fuzzy continuity criteria.

The tested kernels are matched in uniform equilibrium, dwell and mean entropy.
They distinguish stochastic physical-distance and conditional-identity-Hellinger
transitions. They are NOT literal constant-physical-speed and constant-neural-
code-speed trajectories. Their weak separation cannot be used to close the
collaborator's literal kinematic hypothesis. Similarly, code-space geometry is
not anatomical distance between neurons.

Common mixture proportions across recordings and known RUN maps make the exact
screen optimistic. Heterogeneity, estimation error and uncertainty in real event
definitions would require further validation. Do not run a biological mixture
classification with this estimator or pool simulation replicates into a larger
apparent sample to bypass the failure.

## Next Search Direction

Close this estimator as a proposed route to a current biological claim. Return
to two precisely specified kinematic alternatives on identical continuous path
geometries: constant physical arc-length speed versus constant population-code
arc-length speed. Match event duration and geometric extent; make their clock
warps explicit rather than replacing them with entropy-matched random walks.

Before examining real replay, verify that those alternatives produce different
latent predictions and that native-count simulations can recover the difference
using the same independently decoded, uniform-prior observation pipeline. Include
stationary/discontinuous controls, resolution changes and rate perturbations.
Any continuity selection must be calibrated against these controls, not chosen
to equalize acceptance fractions between datasets. This is a proposed next test,
not an implemented or positive result.

## Novelty Boundary

Mixture-based replay dynamics are already established in
[Denovellis et al. (2021)](https://elifesciences.org/articles/64505).
Replay firing-rate modulation also has direct precedent in
[Tirole et al. (2022)](https://elifesciences.org/articles/79031), making encoding
stability a substantive assumption rather than merely a numerical nuisance.
And constant-speed replay was discussed in
[Davidson et al. (2009)](https://pmc.ncbi.nlm.nih.gov/articles/PMC4364032/).

The candidate contribution would have to be a new, robust mechanistic
discrimination or an independently validated measurement result, not simply
another mixture decoder or a nonsignificant speed correlation. Nothing in this
run alone meets the requested high-importance-paper standard.
