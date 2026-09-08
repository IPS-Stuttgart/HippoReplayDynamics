# hc-11: sensitivity under known generators, not an external biological positive

2026-09-08. Protocol: `hc11_count_matched_recovery_protocol.md`.

## Decision

The frozen scorer can detect cohort-level temporal predictive structure in
specified known-map simulations with the real hc-11 count profiles. This
narrows the explanation of the real negative: raw spike scarcity alone does
not force a negative result under those assumptions. It does not show that
real sleep events contain no replay or that their maps/noise match RUN.

The real POST IMM-minus-independent effect remains +0.0273 nats/event,
pointwise 95% hierarchical bootstrap CI [-0.1776, +0.3162], positive in one
of four animals. No real scoring, selection, prior, or threshold was changed.
The PF conditional-prediction result remains unreplicated in this hc-11 cohort.

This is a capability/identification boundary, not a newly established neural
mechanism or completion of the high-importance-paper search.

## Frozen experiment and audit

- All 320 templates: 160 PRE and 160 POST; eight sessions, four animals.
- Four generators, 50 independent conditional datasets each: 64000 full
  population draws. One draw is reused across all five frozen neural splits
  and both encoding variants. Total 640000 wide split-score rows.
- Each time bin retains exactly its observed full-population spike total.
  Identities are sampled from the normalized native directional RUN map at
  the generated position. Split-specific counts and active units can change.
- Generator families are independent positions, static location, diffusion,
  and first-order IMM with the previously frozen native kernels. They are
  stochastic paths, not constant-speed or biophysical RatInABox simulations.
- Proper conditional held-out probabilities, untempered T=1, training-only
  latent inference. True direction/path are not arguments to the prediction
  function. The oracle is calculated separately after inference.
- Event medians across splits, then equal-animal averages. Both primary
  contrasts must have positive means, positive lower bootstrap limits and
  positive effects in all four animals for a positive pattern.

Independent verification passes: all 64000 bin-total profiles/valid paths and
split count partitions; 192 independently regenerated draws; 1536 separate
dense log-domain predictive calculations (max discrepancy 1.42e-12); all
128000 event contrasts/ranks and 4800 bootstrap panels (max point/CI error
1.78e-15); all 16 positive-pattern counts. Full condition coverage and
normalized finite scores are checked non-vacuously. The reporter separately
reconstructs confusion-table counts from the audited event ranks.

This does not independently recompute every model score, establish realistic
neural noise, or turn simulations into additional biological subjects.

## Primary POST result

Direction-mixture encoding, T=1. Effects are the median across 50 simulated
cohort-level equal-animal estimates, in nats/event.

| True generator | IMM - independent | IMM - static | Both-contrast positive pattern |
| --- | ---: | ---: | ---: |
| Independent positions | -0.070 | +2.522 | 0/50 |
| Static location | +1.105 | -0.109 | 0/50 |
| Diffusion | +0.752 | +0.570 | 34/50 |
| First-order IMM | +0.542 | +1.035 | 48/50 |
| Actual POST cohort | +0.027 | +1.510 | Not supported |

For true IMM, 48/50 is 96%, two-sided exact binomial 95% interval
[86.3%, 99.5%]. For diffusion, 34/50 is 68% [53.3%, 80.5%]. Zero of 50
has upper interval bound 7.1%, not a certified zero error rate. These are
conditional simulation frequencies, not biological sensitivity/specificity.

The p05-p95 range of the simulated IMM-minus-independent effect is
[+0.416, +0.677] for IMM and [+0.621, +0.854] for diffusion. The actual point
estimate is below every one of these simulated pure-generator estimates.
This cannot identify a biological mixture or exclude weaker/slower/sparser
latent structure, alternative priors, or RUN-to-sleep encoding mismatch.

Both baselines are essential. A static generator readily beats independent
positions; independent positions readily beat a static location. Neither
single contrast alone is evidence for moving replay. This is a demonstration
within the benchmark, not a new general statistical principle.

## Preserved sensitivities

| Phase / encoding | Diffusion positive patterns | IMM positive patterns | IID / static |
| --- | ---: | ---: | ---: |
| POST / direction mixture (primary) | 34/50 | 48/50 | 0/50 each |
| POST / pooled | 41/50 | 48/50 | 0/50 each |
| PRE / direction mixture | 38/50 | 48/50 | 0/50 each |
| PRE / pooled | 39/50 | 50/50 | 0/50 each |

These do not constitute independent replications: encodings/splits share
population draws, and every simulation reuses the same four-animal templates.

## Cohort detection is not event identification

Primary POST raw per-event best-model recovery is 61.5% for IID, 59.8% for
static, 42.6% for diffusion, and only 13.1% for IMM. Even IID-generated events
rank IMM first in 14.4% of draws. All ranks use the same event-median paired
gain definition; ties are explicit, and no confidence threshold is applied.

The low IMM raw recovery is compatible with positive mean predictive gains:
short sparse draws from a switching process often resemble individual modes,
and the size of a score difference is not captured by winner counts. It does
not prove an implementation error or absence of switching. Conversely, the
96% cohort-pattern frequency must never be called 96% event classification
accuracy. Model winner fractions alone cannot support a mechanistic taxonomy.

## Counts matched, information only approximately matched

Observed POST medians: total 20 spikes, train 14, held out 5.5, active units
11. Across simulated IMM cohorts the corresponding median-of-medians is
20, 13, 6, 11. Diffusion is similar. PRE held-out median is 5 observed versus
6 simulated. These checks do not establish identical event-level information.

The benchmark assumes known stable RUN maps and conditional independence.
It omits sleep-specific cell gains, correlated noise, map changes and the
original event-detection mechanism. It conditions on selected real event
count profiles without reapplying the biological event detector to simulations.
Do not extrapolate the null frequencies to a full detection pipeline.

## Next research decision

Do not scale or retune the weak sleep evidence to obtain an external IMM
positive. If investigating the transfer failure, first require an empirical
held-out MAZE spike prediction control using separately fitted RUN maps and
the same count-conditioned scoring. Known-generator recovery alone cannot
certify that native observed spikes follow the assumed encoding distribution.
That would distinguish measurement/model transfer from insufficient total
counts; it would not itself establish a replay learning mechanism.

## Artifacts

Producer commit: `627699930fd75175b2697ba1e8ed52cba1a70f46`.
Verifier commit: `16383c5ba54ee84c867b0d8744ed93523737fd79`.
Recovery manifest SHA256:
`0f7db3b9932a204617e1a62fade770e764aeabb6abba28d89dc826d9113111ae`.

Run: `/mnt/seagate10tb/florianpfaff/hc11-count-coherent-recovery-50x320x4-20260908/`.
Audit: `/mnt/seagate10tb/florianpfaff/hc11-count-coherent-recovery-50x320x4-audit-20260908/`.
Report: `/mnt/seagate10tb/florianpfaff/hc11-count-coherent-recovery-50x320x4-report-v2-20260908/`.
Figure: `hc11_conditional_recovery.png` (and PDF).
Readout: `recovery_readout.csv`; participation: `recovery_information_comparison.csv`.
