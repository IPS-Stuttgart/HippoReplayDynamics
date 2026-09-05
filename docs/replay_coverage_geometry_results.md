# Controlled Field Geometry and Decoder Resolution

Status: completed, reconstructed synthetic experiment; the full study is NOT
yet paper-ready. No biological speed-uniformity or replay-prevalence conclusion.

## Provenance

- Compute: gpuserver6000.
- Artifact: `/mnt/seagate10tb/florianpfaff/replay-coverage-geometry-full-20260905`.
- Scorer clean commit: `5b1eb81a47a40741364525f3fe769efb0eb4b4a8`.
- Reconstruction clean commit: `90f305e35d56e39a7969b5f66a7b01daef6e866e`.
- Final non-rescoring report commit: `ce0d58f3`; outputs in `report-final/`.
- RatInABox 1.15.3 Gaussian field responses; known prescribed paths. RatInABox
  does not supply a neural replay-generation mechanism in this experiment.
- Eight independent population seeds, 24 paired path seeds each, six truth
  classes: 192 paired source draws / 1,152 truth arrays, reused across fields.
- 24 spatial configurations (N64/128, sigma20/30/40 cm, area3.7/8.75 m2,
  aspect1/1.4); targeted 4/8/16 cm x 10/20/40 ms x 5/10 ms resolution factorial.
- 96 batches, 55,296 observation arrays, 1,400,832 repeated metric rows.
  Rows/settings are not independent samples or biological animals.
- Runtime 890.9 seconds. All technical gates passed. All path/count/grid/rate
  hashes reconstructed; every metric row independently recounted for support
  and true geometric eligibility; 9,728 rows additionally checked against an
  independent analytic Gaussian likelihood/position-error calculation. No
  discrepancies. This is not a second independent redecoding of every row.
- The focused suite has 118 passing tests; Ruff and whitespace checks pass.
  Separate smoke data are excluded. Source code and inputs are archived.
- All ten final-report output hashes were verified. The three production
  figures were visually inspected: field factorial, resolution factorial, and
  continuous-recovery/null-acceptance trade-off. No blank or clipped panels.

All effect summaries average paths within population, then populations equally.
Intervals bootstrap eight synthetic population seeds, not animals. The sweep
is descriptive; individual intervals are not multiplicity-adjusted discoveries.

Production figures in `report-final/`:

- `geometry_field_factorial.png`
- `geometry_resolution_factorial.png`
- `geometry_recovery_null_tradeoff.png`

## 1. Arena Size Alone Is Not a Sufficient Explanation

Primary illustrated slice: N128, Gaussian sigma30 cm, aspect1.4, full-arena
support, 8 cm grid, 20 ms windows/5 ms stride, MAP, no bin-support filter,
constant true speed 1000 cm/s, conditional on true geometric eligibility.
The same physical paths stay in a common central 140 x 140 cm domain.

| Observation family | 3.7 m2 recovery | 8.75 m2 recovery | Large minus small, pp (95% CI) |
|---|---:|---:|---:|
| Native Poisson | 40.94% | 11.34% | -29.61 [-37.98, -20.80] |
| Common total schedule / conditional likelihood | 44.39% | 53.74% | +9.35 [-1.80, +21.73] |

Native counts average 67.9 versus 32.8 spikes/event. The separate conditional
generator uses identical total schedules (about 59.7 spikes/event) in both
arenas. It assigns identities using true position; it is NOT a count-only
intervention on the native observations and does not overturn the earlier
data-only pooled-cell result. Its interval crossing zero is not equivalence.
It demonstrates that a larger-arena penalty is not universal in these models.

The native large-arena penalty remains with the decoder restricted to the
common support (49.24% versus 14.77% recovery), so expanded decoder search space
alone does not account for it. Restriction itself is optimistic known-support
information unavailable during biological replay. Aspect effects are mixed;
the complete paired-factor table retains both aspect ratios.

## 2. Broader Fields Have Competing Effects

Full-arena N128, aspect1.4, 8 cm /20 ms /5 ms, MAP without support exclusion:

| Sigma (cm) | Native, 3.7 m2 | Native, 8.75 m2 | Common count, 3.7 m2 | Common count, 8.75 m2 |
|---|---:|---:|---:|---:|
| 20 | 40.79% | 6.48% | 77.59% | 77.15% |
| 30 | 40.94% | 11.34% | 44.39% | 53.74% |
| 40 | 51.17% | 16.76% | 17.30% | 12.26% |

Wider fixed-peak fields supply more native spikes, but make location tuning
broader. Under the fixed-total generative experiment, recovery declines strongly
with width. For sigma40 versus30 the conditional differences are -27.09 pp
[-35.54,-19.24] and -41.48 pp [-48.01,-35.14]. This is a trade-off between
information quantity and spatial specificity, not a universal density formula.
Native cell removal still hurts: N64 minus N128 is -22.80 pp in 3.7 m2 and
-7.19 pp in 8.75 m2 at sigma30 (paired intervals exclude zero).

## 3. Longer Windows Recover Paths AND Admit Nulls

Same N128/sigma30/aspect1.4/full-support/native-Poisson slice, 8 cm grid,
5 ms stride, unfiltered MAP. Each setting reuses EXACTLY the same fine spikes.

| Arena | Window | Eligible continuous recovery | Stationary acceptance | Independent snapshots acceptance | Whole-bin shuffled acceptance |
|---|---:|---:|---:|---:|---:|
| 3.7 m2 | 10 ms | 1.17% | 0.52% | 0.00% | 0.00% |
| 3.7 m2 | 20 ms | 40.94% | 15.63% | 0.52% | 4.17% |
| 3.7 m2 | 40 ms | 89.42% | 4.17% | 23.44% | 17.19% |
| 8.75 m2 | 10 ms | 0.00% | 0.00% | 0.00% | 0.00% |
| 8.75 m2 | 20 ms | 11.34% | 7.29% | 0.00% | 0.52% |
| 8.75 m2 | 40 ms | 64.34% | 18.23% | 10.42% | 18.23% |

These are geometric acceptances of specific simulated classes, not biological
replay FPRs or shuffle p-values. Zero observed acceptance is not zero population
error. Stationary acceptance is nonmonotonic and configuration-dependent.

The continuous improvement is not just a changing truth-eligibility denominator:
on the SAME 169 paths eligible at 20 and 40 ms, recovery differences are +48.48 pp
[44.73,52.14] and +53.01 pp [44.58,60.83]. Thus longer averaging really changes
the inferred geometry, but its higher sensitivity is not an unqualified win.

At 20 ms, refining the grid from 8 to 4 cm changes MAP recovery by -4.71 pp
[-14.75,+5.42] and -5.35 pp [-8.69,-2.38]. Coarsening to 16 cm reduces it by
-20.28 pp and -7.68 pp. A fixed 20 cm jump threshold interacts with discretization;
more bins do not automatically mean more true paths accepted. The separate
10 ms-stride time-scaled rule avoids silently changing the implied speed/span
threshold; its results are retained, not substituted for the literal heuristic.

## 4. Speed-Gradient Recovery Is Information-Limited, Not Fixed by a Finer Grid

Posterior mean, before selection, native Poisson, known horizontal coordinate
(an optimistic analysis), other settings as above. Injected gradient contrast
is 1.0. The true arclength response is approximately 1.0 at each window length.

| Window | Window-mean truth response | Decoded, 3.7 m2 (95% CI) | Decoded, 8.75 m2 (95% CI) |
|---|---:|---:|---:|
| 10 ms | 0.903 | 0.059 [-0.097,0.239] | -0.156 [-0.393,0.087] |
| 20 ms | 0.816 | 0.442 [0.341,0.560] | 0.195 [0.086,0.315] |
| 40 ms | 0.647 | 0.556 [0.502,0.611] | 0.423 [0.348,0.493] |

The 4/8/16 cm curves are nearly identical: at 20 ms, responses are
0.4421/0.4422/0.4423 (small) and 0.1949/0.1948/0.1947 (large). Refining spatial
bins does not restore the missing gradient. Longer windows recover more of the
time-averaged signal, while the averaging itself removes real kinematic detail.

Continuity selection does not guarantee identifiable gradients. At 10 ms neither
arena has any population with both selected gradients available. At 20 ms only
1/8 large-arena populations qualifies; at 40 ms all eight do. The reporting rule
remains >=5 events and coordinate variance>=0.01. Do not treat the selected
large-arena 20 ms response from one population as a replicated positive.

## Interpretation and Remaining Requirements

The strongest new result is a recovery/specificity trade-off under unchanged
spikes: window length changes both genuine-path recovery and null acceptance,
while finer grids barely affect speed-gradient recovery. This makes increasing
trajectory counts an inadequate validation target. Geometry, tuning width,
information quantity and analysis resolution must be evaluated jointly.

This fulfills the controlled factor-isolation experiment within its frozen
ranges. It does NOT establish optimal settings for PF/Tanni, biological speed
uniformity, or that recording coverage explains their observed replay-rate
difference. The arenas here are synthetic size conditions, not new animal data.
Gaussian fields, independent fine-bin Poisson counts, fixed 200 ms duration,
known maps and a prescribed common path domain are important limitations.
Even the matched observation family uses a static-position likelihood inside
each window although the true path moves; the truth-chord reference exposes
only part of that approximation, not a complete correction.

Next required experiments are map-estimation/observation mismatch and independent
calibration/abstention validation, followed by event-definition sensitivity and
comparison with published baselines. Only then can the complete claim matrix
support a paper-ready result. Do not retune thresholds to force a positive or
uniformity finding. The full study goal remains active.
