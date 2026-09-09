# Exact-Generator Recovery: Results And Decision

2026-09-09. Simulation diagnostic, not a real replay result. This does not
establish constant physical speed, constant neural-space speed, a biological
propagation mechanism, or a high-importance paper discovery.

## Frozen Run

- Worktree: `/home/florianpfaff/HippoReplayDynamics-metric-oracle-recovery`.
- Producer commit: `d59a901f9d3949409aab5c1c4d0ee6de9ab528ef`.
- Verifier/reporter commit: `e87c0304ff4d46ed7c0d4e4225ed4be7c09c47b1`.
- Run: `/mnt/seagate10tb/florianpfaff/metric-oracle-recovery-all33-20260909`.
- Audit: same prefix plus `-audit/metric_oracle_audit.json`.
- Report: same prefix plus `-report/metric_oracle_report.md`.
- Protocol: [metric_oracle_recovery_protocol.md](metric_oracle_recovery_protocol.md).

All 33 source recordings and nine animals were retained: eight Pfeiffer/Foster
recordings from four rats and 25 Tanni recordings from five rats. No real replay
events were rescored. We reused the frozen finite-recovery simulations, with
native spike counts primary and fourfold support diagnostic. Each of 256
path/profile families per recording has five observation realizations; each
realization is classified separately before averaging its accuracy indicator.
Calibration and evaluation use separate path/profile trial indices.

The independent audit passed 1,047,552 reconstructed scores, 337,920 copied
scores checked against the audited parent, 1,584 log-domain hmmlearn checks,
and all 76,032 split-reduced trial rows and downstream summaries. Maximum
absolute likelihood discrepancy was `2.2737367544323206e-12` nats. The verifier
uses a backward recursion rather than the producer's forward recursion.

## Primary Classification

Balanced accuracy, with 50% chance and conditional simulation 95% intervals:

| Information / estimator | Pfeiffer/Foster | Tanni |
|---|---:|---:|
| True latent path | 58.50% [55.08, 61.82] | 60.88% [58.69, 63.03] |
| Whole event, exact generating geometry | 53.67% [51.25, 56.09] | 52.25% [50.91, 53.55] |
| Whole event, training-cell geometry | 52.19% [49.61, 54.57] | 51.61% [50.36, 52.86] |
| 40-ms forecast, full RUN geometry | 52.11% [49.88, 54.34] | 50.96% [49.86, 52.15] |
| 40-ms forecast, training-cell geometry | 51.99% [49.73, 54.06] | 50.67% [49.54, 51.80] |

The CSVs, not rounded values here, are authoritative. Intervals condition on
source rate maps and the fixed simulation/calibration design. They are not
biological population intervals. These forecast numbers differ from the prior
finite-recovery report because that report classified the median split contrast;
this report averages separately classified observation realizations.

The true-path versus native exact-observation accuracy gap is +4.82 percentage
points in Pfeiffer/Foster [1.19, 8.44] and +8.63 in Tanni [6.38, 10.83]. The
native whole-event versus full-geometry forecast differences are small, with
intervals crossing zero in both datasets. Correcting the estimated neural
geometry improves native whole-event accuracy by +1.48 points in Pfeiffer/Foster
[0.23, 2.77] and +0.64 in Tanni [-0.09, 1.38]; this is not a large recovery gain.

## Detection Is A Separate Question

The structure detector asks whether either moving generator beats both
stationary and independent-position alternatives at calibrated null thresholds.
Its power is not the same quantity as physical-versus-neural classification.

| Dataset | Support | Minimum power across moving generators | Maximum empirical null FPR | Practical operating pass |
|---|---:|---:|---:|---|
| Pfeiffer/Foster | Native | 21.48% | 3.44% | No |
| Pfeiffer/Foster | 4x | 40.31% | 2.27% | No |
| Tanni | Native | 9.40% | 2.88% | No |
| Tanni | 4x | 26.30% | 2.75% | No |

Both true-path diagnostics pass the frozen operating rule. Neither native nor
fourfold whole-observation analysis passes. The chosen 50% minimum power is a
practical target, not an information-theoretic theorem. The empirical FPRs are
not high-confidence population bounds. Native and fourfold observations are
different simulated draws, so small accuracy decreases at fourfold support
must not be interpreted as evidence that additional spikes reduce information.

## What This Changes

1. There is weak but detectable separation of these specified physical and
   neural transition distributions. They already overlap substantially for
   short events even when position is known exactly.
2. Observation loss further reduces separation. Whole-event inference with the
   exact known generators remains only slightly above chance. More flexible
   thresholding cannot eliminate the Bayes error of the same observation model.
3. This is not a universal limit on distinguishing physical and neural
   mechanisms. The tested generators are stochastic distance kernels with
   matched uniform equilibrium, dwell and mean transition entropy. They are not
   two constant-speed theories and do not encode anatomical neural-sheet
   distances. Conditional spike-identity emissions also discard total-rate
   information by design.
4. Do not tune an event classifier until it yields a desirable biological
   allocation. No new real-data mechanism classification is authorized by this
   run, and fuzzy continuity was not tested here.
5. Weak individual classification does not rule out population-level inference.
   A calibrated likelihood-mixture analysis may pool weak information across
   independent events; hard winner counts would throw much of that information
   away. This remains a hypothesis requiring recovery, not a result.

## Next Discriminating Test

Test population-level physical/neural mixture recovery on newly generated,
independent event ensembles before examining any real-data mixture estimate.
Keep stationary and independent-position events as nuisance components and
include equal-mixture, physical-enriched and neural-enriched ensembles. Assess
interval coverage, bias, false directional claims and detection power, with
recording/animal hierarchy explicit. Test observation-model mismatch as well
as exact-generator recovery. Reusing individual simulated score rows many times
must not be presented as additional independently observed events.

If this also fails, reformulate the biological alternatives to make genuinely
different, independently motivated predictions or identify a recording regime
with adequate information. Do not rename the same weak event classifier or
relax its thresholds. The broad discovery objective remains unmet.
