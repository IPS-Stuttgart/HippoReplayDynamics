# Literal Replay Clock Recovery: Results And Next Decision

2026-09-09. Simulation-based measurement study. No real replay event was scored
or assigned a mechanism. No high-importance biological discovery is established.

## Artifacts

- Worktree: `/home/florianpfaff/HippoReplayDynamics-literal-clock-recovery`.
- Frozen producer and passing independent verifier:
  `54ded4f4e4c33896f4b5bbe5cc0024d846cc406d`.
- Reporter and additional regression tests: `1f857707`.
- Run: `/mnt/seagate10tb/florianpfaff/literal-clock-recovery-all33-20260909`.
- Audit: same prefix plus `-audit/literal_clock_audit.json`.
- Report: same prefix plus `-report/literal_clock_report.md`.
- Validation: same prefix plus `-validation-v2/validation.json`.
- [Frozen protocol](literal_replay_clock_protocol.md).
- [Earlier stochastic-kernel population recovery](metric_population_recovery_results.md).

The run contains 1,056 geometric paths and 21,120 independently sampled spike
observations across 33 source recordings and nine animals. Each path has matched
physical/code generators, exact/gain-drift emissions and five observation
realizations; each is decoded at 8 and 16 cm. There are 42,240 score/decoder rows.
Native MUA count profiles supply durations and per-bin spike totals, not real
spike identities or known biological trajectories. No continuity selection was
applied. Source animals provide encoders, not independent biological results.

The verifier regenerated all sampled paths and observations, reconstructed
interpolated rate maps and clocks, checked all score/decoder rows and hierarchy
reductions, and doubled integration order from 128 to 256. Maximum probability
change was `1.3556290706789875e-6`; zero class labels changed. Numerical readiness
passed. The relevant combined test suite passed 180 tests; Ruff and format checks
passed. A first validation log records reporter-style issues, since corrected.

Metadata caveat: in the frozen run, `completed[].events`, `rows` and `runtime_s`
were inherited from the SOURCE MUA cache. They are not this simulation's counts
or per-record runtimes. Simulation counts are `n_paths`, `n_rows` and
`n_observations`; total simulation runtime is the top-level `runtime_s` (6.07 s).
Future producer outputs omit these ambiguous inherited fields. The original
hashed artifact remains unchanged; no simulation was rerun to alter the result.

## Primary Result

The scorer knows the geometric path, endpoints, duration and original rate map.
It tests only which clock generated the event. This is an optimistic oracle,
not an operational unknown-path decoder or held-out neural prediction.

| Source dataset | Exact oracle accuracy | Gain-drift accuracy | Least accurate animal, exact | 80% practical target |
|---|---:|---:|---:|---|
| Pfeiffer/Foster | 61.56% | 62.03% | 59.22% | Fail |
| Tanni | 56.91% | 56.83% | 54.88% | Fail |

Observation repeats are averaged within path, paths within recording, recordings
within animal, and animals within dataset. Both conditions remain above chance
descriptively for every animal. No Monte Carlo confidence interval or population
mixture recovery test has been run for these literal clocks. The 80% threshold
is a predeclared practical single-event target, not an information-theoretic
boundary. Failing it does NOT rule out aggregate inference from many events.

The clocks' bin-midpoint positions differ by an RMS average of 2.48 cm in PF and
2.29 cm in Tanni. Independent flat-prior decoding has mean errors of 24.57 and
36.09 cm respectively, against within-bin time-mean truth. These are different
measurements, not a formal signal-to-noise bound. They explain why a visually
similar decoded trajectory need not distinguish the mechanisms.

The conditional-code distance per cm varies modestly along the tested paths
(mean coefficient of variation .164 PF, .156 Tanni). Constant-code-speed paths
have approximately .135/.136 variation in their bin-level physical arc speeds;
constant-physical-speed paths have numerical-zero variation. These relations
are properties of the constructed clocks, not biological observations.

## Decoder Speed Diagnostic

For the exact constant-physical-speed generator, the mean of event-level
decoded/true-bin-mean step-speed ratios is 2.84 in PF and 4.91 in Tanni at 8 cm.
At 16 cm it is 2.87 and 4.93. These are hierarchical means of ratios, not ratios
of the pooled average speeds. They illustrate substantial noise inflation in
this unselected synthetic candidate regime; coarsening alone barely changes it.
They are NOT correction factors to apply to real replay speed.

The independently decoded conditional likelihood deliberately does not use a
temporal prior. Its failure is not evidence that an HMM imposed constant speed.
Conditioning on each bin's total removes rate-envelope information; this is not
the full Poisson decoder. Inhomogeneous rates are integrated across each moving
bin, avoiding the false assumption that the latent position stands still for
20 ms.

The <20 cm decoded-step fraction is only a diagnostic. In this non-overlapping
20 ms analysis it is not Foster's full continuity/event criterion, nor the same
speed restriction as 20 ms windows advanced every 5 ms. It cannot be interpreted
as an established replay acceptance rate. This run did not test fuzzy processing.

## Scope And Next Decision

Do not run a real-event clock classifier based on this result. Do not claim
uniform replay speed, a compensating neural-sheet propagation mechanism, or a
general inability to measure kinematics. Code distance here is Hellinger distance
of normalized RUN cell-identity rates, not anatomical distance or full-Poisson
population geometry. Endpoints are 40-120 cm apart; curves can be longer. This
does not exhaust paths across the largest Tanni environments. Maps are treated
as known and stationary/fragmented nulls are not tested in this oracle screen.

The earlier population-mixture failure used different stochastic generators and
does not automatically transfer to these explicit clocks. A justified next
measurement test is whether likelihood aggregation can recover a population
clock proportion WITHOUT supplying each event's true path. Unknown-path nuisance
integration and independent synthetic calibration are necessary; repeating the
oracle on real events is impossible and hard winner counts discard information.

Another independent question is whether retaining within-bin spike timing adds
useful discrimination that 20 ms binning removes. Any such study must freeze new
simulations and compare matched observations, not tighten selection after seeing
which real events favor the preferred mechanism. Neither follow-up is a result
of the current study.

For fuzzy continuity, the objective remains calibrated decisions above a known
resolution floor, with an unresolved category below it. Stable acceptance rates
alone are not evidence that either a trajectory or a biological mechanism was
recovered. The broader publication goal remains open.
