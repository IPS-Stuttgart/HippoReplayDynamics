# Burst-Time Recruitment Control: Results

## Decision

The proposed cross-dataset stereotyped recruitment explanation is not
established. Retain this as an audited alternative-model control, not a new
biological mechanism or a high-importance discovery. Do not tune the knot
count, drop weak animals or change normalization to rescue it.

The five-knot model has a small positive predictive effect in Pfeiffer/Foster,
but fails the full predictive criterion in Tanni. Both datasets are sensitive
to shuffling even for this simple time-only predictor. Spatial predictors remain
better than this comparator in every animal. Their advantage is not uniquely
attributable to spatial content here: unlike the burst-time model, they also
receive target-event training-cell observations.

## Frozen Experiment

- 4,001 Pfeiffer/Foster immobile MUA events, eight sessions, four animals.
- 5,224 Tanni immobile MUA events, 25 recordings, five animals.
- All 9,225 frozen candidates; no selection by continuity or model evidence.
- Five chronological event folds, one-second guards, five fixed neural splits.
- Fit cell-composition profiles versus relative burst time on other events.
- Five time-profile knots primary; three and ten fixed sensitivities.
- Twenty original-parent whole-bin permutations at five knots.
- Reuse existing spatial and neural-HMM scores, without refitting or rescoring.
- Proper count-conditional multinomial prediction of held-out cell identities.

This is normalized elapsed event time, not theta or ripple oscillation phase.
It is also not a posterior trajectory smoothness or fuzzy-continuity experiment.

## Primary Results

Units: paired predictive difference in nats per held-out spike. Median over
neural splits within event, then equal events within session, sessions within
animal, and animals within dataset. Intervals are exact animal-bootstrap
percentiles, conditional on the existing fits/folds. No broad-search correction.

| Contrast | Pfeiffer/Foster mean [95% CI]; positive rats | Tanni mean [95% CI]; positive rats |
|---|---|---|
| Burst time minus global composition | +0.00670 [+0.00163, +0.01177]; 4/4 | -0.00211 [-0.01349, +0.00850]; 3/5 |
| Burst time minus own time average | +0.00802 [+0.00212, +0.01393]; 4/4 | +0.00326 [-0.00284, +0.01067]; 3/5 |
| Burst time original minus shuffled | +0.03851 [+0.02901, +0.04801]; 4/4 | +0.04560 [+0.03572, +0.05568]; 5/5 |
| Spatial IMM minus burst time | +0.70251 [+0.64714, +0.75101]; 4/4 | +0.34654 [+0.25958, +0.45064]; 5/5 |
| Spatial IID minus burst time | +0.59081 [+0.52913, +0.65249]; 4/4 | +0.26834 [+0.19213, +0.36397]; 5/5 |
| Learned K50 HMM minus burst time | +0.73859 [+0.69725, +0.77994]; 4/4 | +0.20136 [+0.07114, +0.30918]; 4/5 |

At three knots Tanni beats the model's own time average in all five animals,
but still does not reliably beat global composition. Ten knots lose reliable
gain over global composition in both datasets. Thus no prespecified setting
passes the complete recruitment criterion in both datasets. These are
sensitivity analyses, not alternative primary outcomes.

Raw nats/event do not support all-animal uniformity even for Pfeiffer/Foster:
burst time beats global composition in three of four rats. Preserve both raw
and normalized tables; do not present the per-spike result as universal.

## Interpretation Boundaries

1. Original-minus-shuffle tests alignment. It does not establish superiority
   to an independently trained time-independent predictor.
2. Failure of this particular time-only model does not establish biological
   absence of stereotyped recruitment. A common template may miss variable,
   context-dependent recruitment or suffer limited calibration support.
3. Spatial IID also beats the burst-time model, so this comparison is not
   specific to IMM, switching or temporal coupling.
4. Spatial models and the learned HMM condition on target training cells;
   the burst-time model does not. This is a deliberately simple alternative,
   not a matched-information causal decomposition.
5. Calibration includes all neurons on other events. No target event spikes
   fit the burst-time model. Event detection nevertheless used all cells,
   so inference is conditional on the frozen candidate selection/duration.
6. Scores sum bin-marginal count-conditional log predictions. They are not
   predictions of total firing rate or a joint held-out sequence likelihood.
7. Independent numerical auditing is not independent biological replication.

Stereotyped recruitment and replay-detector validation have close precedents:
[Liu et al., 2019](https://doi.org/10.1016/j.neuron.2019.05.040),
[spontaneous brain-wide cascades, 2024](https://academic.oup.com/pnasnexus/article/3/4/pgae078/7609348),
and [Takigawa et al., 2024](https://elifesciences.org/articles/85635).
This control does not establish that a new mechanistic paper exists.

## Provenance And Verification

Server: gpuserver6000. Branch: `test-burst-phase-prediction`.

- Frozen scoring commit: `ff1971d4525159cfc2534aaa70660ce8b0ebbd2e`.
- Reporter commit: `00c7d46b`.
- Run: `/mnt/seagate10tb/florianpfaff/burst-phase-prediction-all9225-20260909`.
- Run manifest SHA256:
  `167722f20a9fd2d5c804fa6b22f8d5af51bc726bbb26b4fb6a804d0255f89e86`.
- Independent audit: same run path plus `-audit/burst_phase_audit.json`.
- Report: same run path plus `-report/`.

The independent audit passed: all 165 fold-specific calibration profiles at
three knot settings; all 138,375 original split/knot predictions; 46,125 first
shuffle predictions; 184,500 independently reconstructed predictions total.
Maximum discrepancy was 1.71e-13 nats. It also checked source/output hashes,
chronological folds, neural partitions, counts, 922,500 shuffle-row coverage
and means, reused comparator scores, paired event medians, session/animal
summaries and exact bootstrap intervals. It did not reopen native recordings,
repeat source-model optimization, or independently recalculate the other
19 shuffle scores for each event/split.

All 18 new kernel/reporter tests and 40 related scoring tests passed. Ruff and
staged whitespace checks passed. This experiment does not alter existing
replay detection, criteria, published primary datasets or artifacts.
