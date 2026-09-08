# Matched 2D MUA rate-transfer results

2026-09-08. Completed exploratory diagnostic, not an independent confirmation
cohort or a new biological gain mechanism. No new event selection or dynamics
tuning. The frozen protocol is `2d_mua_rate_transfer_protocol.md`.

## Provenance and verification

- Producer: `0e8551b62e37510b84fa910f98b0d040b00e5add`, clean at launch.
- Audit/report: `0601e0b3e61c41444b0a5280df9beb61252ee595`.
- Run: `/mnt/seagate10tb/florianpfaff/2d-mua-rate-transfer-all9225-k20-20260908`.
- Manifest SHA256: `84418e7db33df939b9609dae312eed19d72b7dc6e8d382d7cbdc91e4c1744943`.
- Audit and report use the same run path with `-audit` and `-report` suffixes.
- Production runtime: 1,463.40 s on gpuserver6000, 12 workers.
- 44 relevant tests pass; Ruff and whitespace checks pass.
- Independent audit passes: 184,500 original analytic global scores,
  1,845,000 shuffled score rows, all calibration folds/counts/gains and frozen
  permutations, 1,291,500 split contrasts, 258,300 event contrasts, and 56
  independently reconstructed hierarchical confidence-interval panels.
- A separate dynamic-inference implementation verifies 3,168 original/shuffled
  predictions across every session; maximum discrepancy is 1.16e-10 nats.
  This is a stratified dynamic-score audit, not independent recomputation of
  every dynamic score. Native counts and RUN maps reuse the audited parent.

## Primary results

All 9,225 frozen candidates: PF 4,001/eight sessions/four animals; Tanni
5,224/25 sessions/five animals. Five 70/30 cell splits and 20 whole-bin shuffles
per event. Differences are paired within split and median-aggregated per event;
session means are averaged within animal, then animals equally within dataset.
Intervals are the frozen 5,000-draw animal/session/event bootstrap, not 9,225
independent biological replicates. They do not include exploratory-choice
uncertainty.

Scores are proper held-out cell-identity multinomial marginal log scores,
conditional on each held-out time bin's spike total. They are not joint event
log evidence or predictions of total spike count. Held-out target spikes never
update the training posterior. Gains use only guarded other-event folds.

| Primary contrast | PF mean [95% CI] | Positive animals | Tanni mean [95% CI] | Positive animals |
|---|---:|---:|---:|---:|
| Recalibrated minus original IMM | +0.613 [0.088, 1.249] | 3/4 | +1.932 [0.439, 3.794] | 5/5 |
| IMM minus independent positions | +1.396 [1.240, 1.578] | 4/4 | +0.817 [0.637, 1.032] | 5/5 |
| IMM minus matched other-event global rates | +11.953 [7.593, 17.052] | 4/4 | +3.696 [2.711, 4.959] | 5/5 |
| IMM real minus permuted map | +0.440 [0.302, 0.572] | 4/4 | +0.133 [0.096, 0.169] | 5/5 |
| IMM original minus mean shuffled order | +0.895 [0.692, 1.085] | 4/4 | +0.547 [0.395, 0.707] | 5/5 |
| IMM order-by-map interaction | +0.394 [0.270, 0.518] | 4/4 | +0.167 [0.118, 0.219] | 5/5 |

All units above are paired nats/event. The complete six-part rule passes for
Tanni and fails for PF because recalibration does not help Rat2 (-0.0496
nats/event). This failure is retained; the other five PF contrasts pass.
The Tanni adaptation gain is positive in all five animals in the declared raw
endpoint, but R2481's per-held-out-spike adaptation gain is negative (-0.0201).
Do not describe calibration improvement as uniformly positive under every
normalization.

## What changed, and what did not

The prior RUN-only Tanni IMM failed the stronger other-event global comparator
(+1.741, CI [-0.237, 3.812], 3/5 animals positive). Giving the spatial maps and
that baseline equal calibration exposure removes this particular failure:
all five Tanni animal means are now positive (+2.49 to +5.10 nats/event).
This is consistent with observation-model mismatch limiting that comparison;
it does not measure the biological source of a gain change.

Recalibration does not create a larger temporal-order effect: the Tanni
order advantage changes from +0.565 to +0.547, and the interaction from +0.172
to +0.167. Independent-position prediction also beats the matched global
baseline (+2.930 [2.004, 4.056], 5/5). Thus much of the gain is observation fit,
not a newly amplified switching mechanism. Event medians of different
contrasts are not algebraically additive.

The stronger-shrinkage alpha=1000 sensitivity retains Tanni IMM minus global
(+3.768 [2.762, 5.037], 5/5) and IMM minus IID (+0.848 [0.654, 1.077], 5/5).
It was frozen in advance and is not substituted for the primary endpoint.
Tanni diffusion beats global (+2.996 [2.113, 4.255], 5/5), but diffusion minus
IID crosses zero (+0.099 [-0.583, 0.643], 4/5). Model preference alone does not
establish a unique neural implementation of switching.

## Publication relevance and next boundary

This is a promising extension of the measurement study: the appearance of a
decoded trajectory and held-out map/order-dependent predictability are not
interchangeable, and an observation-model mismatch can obscure the latter.
It does not establish how many geometrically rejected events are genuine
replays: this new calibration has not yet been joined to those split-specific
labels. Do not silently transfer a previous subgroup finding to these scores.

Candidate detection used all cells; results are conditional on that frozen
ascertainment. Other-event calibration is retrospective, can use later data,
and can reflect sampled spatial content rather than physiological gain.
This is a follow-up motivated by earlier failures in the same events, not
prospective external confirmation. No thresholds or favorable subgroups were
chosen after seeing this run.

Before a biological mechanism claim, freeze the procedure on independent
events/recordings and compare against adequately validated nonspatial temporal
models, not just an event-global baseline. Replay rate codes, state-dependent
firing, HMM decoding and predictive validation have prior literature; none is
new merely because this run passes. The prior hc-11 rate-transfer failure
remains a separate negative result. Physical speed, uniformity, Bayesian
smoothing, and a high-importance mechanism are not established by this test.
