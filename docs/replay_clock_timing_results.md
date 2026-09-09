# Paired Timing Recovery: Results And Next Decision

2026-09-09. The requested high-importance-paper goal remains unmet. This is
concrete progress in measurement validation, not a biological discovery.

## Provenance And Verification

- Worktree: `/home/florianpfaff/HippoReplayDynamics-clock-time-resolution`.
- Frozen producer and independent audit: `488fb6fede04a3766b2b39ad81f245fb99002b0c`.
- Reporter and report tests: `92f75b62`.
- Run: `/mnt/seagate10tb/florianpfaff/clock-timing-recovery-all33-20260909`.
- Audit: same prefix plus `-audit/clock_timing_audit.json`.
- Report: same prefix plus `-report/clock_timing_report.md`.
- Validation: same prefix plus `-validation/validation.json`.
- [Protocol](replay_clock_timing_protocol.md).
- [Parent literal-clock results](literal_replay_clock_results.md).

There are 42,240 fresh fine-timed simulated observations, on the same 1,056
frozen paths from 33 RUN encoders/nine animals. Each observation is evaluated
four ways, yielding 168,960 score rows. Fine/coarse comparisons use exactly the
same spike counts; only their aggregation and likelihood conditioning differ.
No real event was rescored or selected. This is not a new animal replication.

The independent verifier regenerated every observation, checked every raw score,
score decomposition and hierarchical summary, and verified exact expected
data-processing inequalities. Maximum 16-vs-32 quadrature probability discrepancy
was `3.398476656930882e-7`; maximum difference from the parent's 20-ms marginal
was `1.2993403753353006e-6`. All exact fine-joint expected-information excesses
over coarse were positive; the minimum was `5.4424964417876254e-5` nats.
Numerical readiness passed. The combined relevant suite passed 189 tests, with
Ruff/format checks clean. The report figure was inspected visually.

## Matched Accuracy Results

| Dataset | Emission | 20 ms identities | 1 ms identities | Timing only | 1 ms joint | Identity improvement |
|---|---|---:|---:|---:|---:|---:|
| Pfeiffer/Foster | Exact | 61.09% | 62.75% | 50.33% | 62.30% | +1.66 pp |
| Pfeiffer/Foster | Gain drift | 61.04% | 61.76% | 50.39% | 61.64% | +0.72 pp |
| Tanni | Exact | 56.29% | 56.51% | 50.58% | 56.54% | +0.21 pp |
| Tanni | Gain drift | 56.82% | 57.51% | 50.31% | 57.46% | +0.69 pp |

The frozen 80% event-level practical target fails in every arm. No paired gain
reaches the separate descriptive five-percentage-point screen. There is no
claim of statistical equivalence or significance. PF identity gains are positive
in all four source rats; Tanni gains are mixed (three of five positive). These
are simulation-performance summaries conditional on the source maps.

The correct reference is this run's matched coarse observation, not the earlier
run's independent 20-ms Monte Carlo realization. Comparing the old 56.91% Tanni
value directly to this run's fine value would conflate sampling variation with
the effect of time resolution.

## Expected Information

Under exact emissions, the hierarchy-averaged expected true-minus-wrong margins
are conditional KL divergences for these two known-path clock hypotheses.

| Dataset | 20 ms identity | 1 ms identity | 1 ms timing | 1 ms joint |
|---|---:|---:|---:|---:|
| Pfeiffer/Foster | .317442 | .355259 | .000971 | .356230 |
| Tanni | .102515 | .110659 | .000230 | .110889 |

Thus coarse identity retains approximately 89.1% (PF) and 92.4% (Tanni) of the
fine-joint expected discrimination information in this simulation. These are
ratios of hierarchical mean KL values, NOT fractions of all information in the
brain or all information relevant to real replay. Timing-envelope information
is particularly small after conditioning on each native 20-ms total. Expected
information and finite-sample classification accuracy are different metrics;
the latter need not increase in every finite Monte Carlo sample.

## Scientific Boundary

For the tested spatial rate-code clocks, 20-ms binning is not the dominant
explanation for weak event-by-event recovery. Moving to 1 ms retains some extra
information, but it does not make the single-event mechanism label reliable at
the predeclared target. The true geometric path, start/end and original maps
are still supplied to the scorer: unknown-path estimation will need its own
validation. No fine-bin point-position decoder or HMM was used.

This is a discretized marked inhomogeneous-Poisson simulation conditioned on
native parent-bin totals. It does not model ripple-phase locking, refractory
effects, neuronal noise correlations, or a code carried specifically by precise
relative spike timing beyond location-dependent rates. It therefore cannot
exclude useful timing information in actual spikes outside this observation
model. The code clock is conditional-identity Hellinger geometry, not anatomical
neural-sheet propagation or every possible neural metric.

The results neither prove uniform physical speed nor establish that fuzzy
continuity succeeds or fails. Nor does sub-80% single-event accuracy rule out
population inference: pooling likelihoods is different from assigning labels.

## Next Search Step

Do not rescore real replay with this oracle and do not lower the practical
target after seeing results. The surviving measurement question is aggregate
clock inference with unknown geometric paths and nuisance rate variability,
validated on fresh simulations before real data. The earlier stochastic-kernel
mixture failure does not automatically transfer to these literal clocks.

An alternative empirical direction would require an explicitly different,
falsifiable timing-code prediction, not simply making time bins smaller again.
Either route needs an independent novelty check and recovery of a signal beyond
decoder/recording artifacts. The current evidence supports a bounded measurement
finding; it does not meet the requested high-importance-paper objective.
