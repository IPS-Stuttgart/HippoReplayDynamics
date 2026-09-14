# Cell-predictive temporal context: failed remedy and diagnostic

## Frozen experiment and decision

Producer and protocol commit: `132960d6ca0d16a6992652528e84b1c018d2ab8e`.
Scoring, independent audit and report completed successfully on gpuserver6000
in detached systemd user services. No scientific thresholds changed after the
run. The primary method failed the frozen development criteria and was NOT
advanced to HC-11. The overall remedy objective remains unachieved.

This is not an independent-data success. PF is development because prior PF
experiments informed the rule. The existing eight-session HC-11 input benchmark
was checked for availability only, not scored with this method. The original
targeted Home-coverage contrast has not been remedied.

## Method and denominator

Eight PF sessions, four rats, 200 previously hash-selected real candidates per
session: 1,600 real events from a 4,069-candidate source pool. Three frozen,
equal-size disjoint A/B splits were evaluated, with split 0 primary and splits
1/2 sensitivity only. Repeated splits are not independent animals.

Each population independently compared two endpoint posteriors: flat-prior
decoding of the original last 20 ms, versus the previously frozen diffusion +
25% reset model with up to 200 ms of preceding disjoint-bin context. Neither
method changed event identities or endpoint timestamps. No future bins were used.

Three cell folds within A or B tested the joint held-out-cell endpoint Poisson
likelihood integrated over a posterior inferred from the other two folds.
Validation cells never re-inferred the latent posterior. Context was chosen
only for a positive median per-cell predictive score difference above the
1e-10 numerical guard, with at least 3 endpoint spikes and 2 active cells.
Otherwise the original independent posterior was retained. No event was dropped.
The final chosen posterior used all cells of its own population; this refitted
posterior is not itself a held-out predictive score. A never conditioned on B.

First-half RUN data trained maps and unit inclusion; fourth-quarter RUN was
held out. Four known-path simulations tested stationary, continuous moving,
moving with gains/map drift, and a distant change in the final 20 ms. Simulations
preserve original whole-population 5-ms spike totals, not exact active-cell
patterns or biological noise correlations. Real replay locations remain unknown.

## Primary result: equal-animal means

Each session contributes its event-average metric, sessions average within rat,
and rats receive equal weight. The physical errors below average A and B for
display; the frozen truth gates test each side separately, including mean error,
90th-percentile error and regional Brier score.

| Measurement | Independent | Predictive context | Change |
|---|---:|---:|---:|
| Real A/B endpoint separation | 45.450 cm | 44.795 cm | 1.44% reduction |
| Real regional total variation | 0.50973 | 0.50369 | 1.18% reduction |
| Held-out RUN physical error | 58.762 cm | 57.682 cm | 1.84% reduction |
| Stationary simulation physical error | 45.706 cm | 41.874 cm | 8.39% reduction |
| Moving simulation physical error | 39.432 cm | 36.615 cm | 7.14% reduction |
| Moving + gains physical error | 39.191 cm | 36.646 cm | 6.49% reduction |
| Late-jump simulation physical error | 82.631 cm | 84.363 cm | 2.10% increase |

Real separation and regional-TV improvements occurred in only 2/4 rats.
Descriptive 5,000-resample rat-bootstrap intervals for reductions were
[-0.369, +2.171] cm and [-0.00853, +0.02724], respectively. These small-rat
intervals are not evidence for a broad population-level guarantee.

The entropy-matched independent control had separation 46.180 cm and regional
TV 0.51905. The primary is about 3% better than that control, below the frozen
5% requirement, and far below the 10% baseline-agreement requirement.
Both population entropies decreased, so entropy inflation is not the cause of
the small improvement. Sharper posteriors did not ensure stable content.

All six late-jump truth safeguards failed. Mean errors increased from
79.348 to 81.095 cm for A and 85.915 to 87.631 cm for B. Per-session p90 errors
averaged within rat increased from 136.596 to 140.340 cm and 137.945 to
141.400 cm. Regional Brier scores increased from 0.50976 to 0.53091 and
0.56798 to 0.58945. Each of these changes was adverse in all four rats.

## Did predictive scores identify useful context?

The target is reduction of known physical error from unconditional context
relative to independent decoding. Labels were used only for evaluation.
AUROC uses only the predeclared spike/active-cell-supported endpoints, with
session AUROCs averaged within rat and rats equally weighted.

| Known-truth source | A AUROC | B AUROC | Required diagnostic result |
|---|---:|---:|---|
| Held-out RUN | 0.556 | 0.464 | Both fail |
| Late-jump simulation | 0.585 | 0.635 | A fails; B passes |

All these AUROCs were defined; failures are not missing-data artifacts. Other
simulation-source AUROCs ranged approximately 0.431-0.504. The positive
late-jump B result is not a replacement for the full primary requirement.

Context was chosen on 32.25% of real A endpoints and 35.19% of real B endpoints.
Eligibility was 43.00%/47.06%. For RUN, context choices were 15.94%/16.31%.
For late jumps they were only 7.44%/7.81%, versus eligibility 47.94%/43.81%.
Thus the rule rejects most abrupt-jump cases but does not prevent remaining
choices from worsening mean, tail and regional accuracy. It is not a reliable
localization-benefit diagnostic merely because it predicts some held-out spikes.

Sensitivity splits agree with the failure: real separation reductions were
1.45% and 0.83%, regional-TV reductions 1.48% and 1.11%; the improvement
intervals included zero. Late-jump mean errors worsened on both sides and in
all four rats in each split. No sensitivity split replaces the primary.

## Verification and artifacts

- 75 relevant regression tests passed from the repository root. An initial
  test launch from the remote home directory failed import collection; rerunning
  in the correct worktree resolved it without source changes. The optional
  Matplotlib Axes3D warning did not affect the 2D figure.
- The separate dense auditor rebuilt 115,200 readouts, 230,400 population
  posteriors, all 345,600 raw fold-predictive scores, and 26,073 native-clock
  real/RUN context blocks. All input and output hashes matched.
- An additional standalone checker imported neither producer nor reporter. It
  independently reproduced 6,144 session metrics, 3,072 animal metrics, all
  768 aggregate rows, 288 diagnostic rows, bootstrap intervals and all 43 gates.
  AUROC was separately computed from the Mann-Whitney rank formula.
- A modified summary statistic was rejected even after its file checksum was
  updated in a disposable copied report. Original artifacts remained unchanged.
- The four-panel summary figure was visually inspected. It shows the small,
  rat-inconsistent real gains and the late-jump error increase without omitting
  the unfavorable control.

Server root:
`/mnt/seagate10tb/florianpfaff/predictive-context-content-20260914/`

Subdirectories: `pf/`, `pf-audit/`, `pf-report/`.
Standalone verification: `check_report_independently.py`,
`independent_summary_check.json`, `independent_summary_tamper_check.json`.
No source recordings or posterior arrays are included in the compact export.

## Consequence for the next experiment

Do not loosen the threshold, omit the jump control, promote unconditional
context, or call a within-PF result independent validation. This particular
held-out-spike selection rule does not solve the content-instability problem.
The next diagnostic must be tested against actual localization benefit, not
assume predictive spike likelihood is a reliable surrogate. Any new rule must
be frozen before its independent evaluation and must still reduce the original
fixed-endpoint Home contrast without harming known regional RUN recovery.
