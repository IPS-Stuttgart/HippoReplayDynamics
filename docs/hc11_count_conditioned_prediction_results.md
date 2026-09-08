# hc-11: conditional prediction does not replicate the PF pattern

Date: 2026-09-08. Producer `cb048fe16b9561bfa31e4a13d9af16782506c183`;
independent verifier `4e641cd344c88db83093b602a6ea46b8a7a2815f`.
Protocol: `hc11_count_conditioned_prediction_protocol.md`.

## Result

The existing four-rat hc-11 PRE/POST cohort was scored with proper conditional
cross-cell prediction, the missing observation-level test identified after
the PF audit. All 320 candidates, eight sessions and five splits were retained.
153600 model/observation/map/encoding/temperature rows succeeded. No new
selection, scoring thresholds, event windows or dynamic priors were chosen
after reading scores. The primary and sensitivity maps use native MAZE data.

Primary POST, direction-conditioned mixture, inference T=1:

| Held-out identity-score contrast | Equal-animal mean, nats/event | Pointwise 95% bootstrap interval | Positive animals |
| --- | ---: | --- | ---: |
| IMM - independent positions | +0.0273 | [-0.1776, +0.3162] | 1/4 |
| IMM - static location | +1.5099 | [+0.5177, +3.4614] | 4/4 |
| IMM - diffusion | +0.5359 | [+0.2244, +0.9149] | 4/4 |
| IMM real - permuted map | -0.0189 | [-0.1497, +0.1599] | 1/4 |
| Identity-only - total-rate-only IMM inference | +0.2288 | [-0.6440, +1.6957] | 2/4 |

The decisive distinction is between excluding a static location and improving
on independent spatial snapshots. The former is positive; the latter is not
robustly supported. Fixed diffusion is also below the independent comparator:
-0.5359 [-1.0434, +0.0124], 0/4 animals positive. IMM beating diffusion therefore
does not establish a successful model of temporal trajectory structure here.
It can avoid an overly restrictive comparator without improving independent
positions. These comparisons do not identify a neuronal mechanism.

## Sensitivities and PRE/POST

POST IMM-minus-independent:

| Encoding | Inference T | Mean | Interval | Positive animals |
| --- | ---: | ---: | --- | ---: |
| Direction mixture | 1.0 | +0.0273 | [-0.1776, +0.3162] | 1/4 |
| Pooled | 1.0 | +0.0681 | [-0.2308, +0.4544] | 2/4 |
| Direction mixture | 0.3 | +0.0691 | [-0.0580, +0.1923] | 4/4 |
| Pooled | 0.3 | +0.0775 | [-0.1042, +0.2755] | 3/4 |

Every declared interval includes zero. The positive 4/4 count at T=0.3 does
not satisfy the frozen robust-pattern requirement or justify choosing that
temperature after seeing the result. Every held-out likelihood is untempered
and normalized; only the training inference changes temperature.

Primary PRE IMM-minus-independent is -0.0810 [-0.2149, +0.0422]. The matched
POST-minus-PRE shift is +0.1082 [-0.1612, +0.4293], 3/4 animals positive.
That is not robust evidence for experience-dependent predictive improvement.

Per-animal primary POST means: Achilles +0.3264, Buddy -0.1077, Cicero -0.0668,
Gatsby -0.0429. No Achilles-only follow-up is promoted as confirmation.

## Boundaries

- This previously inspected event cohort is an external replication of a
  prediction method, not a prospective new-animal confirmation experiment.
- Failure here is not proof that hc-11 lacks replay, temporally organized
  neural activity, learning, or meaningful directional sequences. Frozen
  encoding/dynamic assumptions and sparse information can limit the test.
- Selected encoding populations span 18-85 cells/session. Median split-level
  POST held-out spikes are 5.5 and training spikes 14. Fifty of 1600 event/split
  combinations have no held-out spikes: retain their zero conditional score,
  with undefined per-spike normalization. Do not make them disappear.
- Do not attribute the PF/hc-11 difference to 2D versus 1D geometry: epochs,
  animals, information, grid, time bins and native dynamic priors also differ.
  Raw score magnitudes across those datasets are not directly comparable.
- Four animals imply a minimum exact one-sided sign-flip p of 0.0625.
  Bootstrap intervals are pointwise/exploratory, not a multiplicity-adjusted
  substitute for a larger animal sample.
- Direction-mixture iid is independent in position conditional on a common
  direction inferred from training data. The pooled-map sensitivity, without
  that common direction, also fails to establish the primary predictive gain.

## Verification

58 focused tests passed. Technical scorer runtime: 65.4 seconds on
gpuserver6000, eight single-thread workers. Raw spike totals match the original
selection for all 320 events; end-boundary convention was explicitly audited.

Verifier reconstructs every selected event from raw spike timestamps; verifies
16 raw-file hashes, all five splits, all eight refitted map caches, and the
frozen population-code permutations. It independently recomputes 1536 score
rows using a separate dense log-domain recursion: maximum discrepancy
1.1027623258996755e-11. It also reconstructs 51200 split contrasts, 10240 event
contrasts and all 96 equal-animal aggregate point estimates. Refitting reuses
the frozen native RUN helper; bootstrap confidence intervals are not
independently recomputed. Do not claim a broader verification scope.

Run: `/mnt/seagate10tb/florianpfaff/hc11-conditional-cross-cell-prediction-320x5-20260908/`.
Audit: `/mnt/seagate10tb/florianpfaff/hc11-conditional-cross-cell-prediction-320x5-audit-20260908/`.
Run-manifest SHA256:
`bc846d25e27b9400d2870b2cad534c781d1fdee12f27496e5a8d96db153d3d07`.

## Decision for the paper search

Close this frozen hc-11 conditional-prediction replication as unsupported,
not as a replicated IMM positive. Keep the existing negative strict ladder
unchanged. Do not tune event selection or priors to turn this test positive.

This adds a useful external constraint to the PF interpretation and to the
methods study: outperforming static or fixed-diffusion comparators is weaker
than demonstrating held-out predictive value over independent positions.
That distinction is not itself a new general statistical principle. It does
not fulfill the high-importance biological-discovery goal, which remains open.
