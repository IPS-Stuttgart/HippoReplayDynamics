# Paired Information-Loss and Recovery Results

Status: completed and reconstructed experiment; full paper-ready study remains
incomplete. No biological speed-uniformity or replay-prevalence conclusion.

## Provenance

- Server: gpuserver6000 (`workstation2`).
- Artifact: `/mnt/seagate10tb/florianpfaff/replay-coverage-counterfactual-all33-20260905`.
- Clean scorer: `36484da3e443fb4641b9ba3da0e823bc150e5a1a`.
- Audit and final non-rescoring report invoked from clean commit `163696c2`.
- 33 sessions, nine animals, 971 frozen source duration profiles, 2,913
  source/randomization units, 17,478 truth-trial records including shuffles,
  489,384 observation arrays, and 1,957,536 repeated metric rows.
- Three synthetic randomizations on the SAME empirical maps and profiles are
  not independent biological replications. Truth conditions share starting
  randomization within each source/randomization unit.
- All six technical gates pass. Every path/count-array hash was reconstructed;
  all output/input/cache/snapshot hashes verified; all metric rows independently
  recounted for spike support. Zero discrepancies. Pooled counts were separately
  verified as sums of removed spikes, excluding that channel from cell-support
  gates. The focused suite has 96 passing tests; Ruff and whitespace checks pass.
- Scoring code and inputs are archived. Three-session smoke artifacts are
  separate and not pooled into this production result.

## 1. Keeping Counts Without Identities Does Not Restore Continuity

Primary: constant true speed 1,000 cm/s, truth-geometrically-eligible paths,
Poisson decoding, MAP geometric criterion, no per-bin support exclusion,
rate multiplier 3, all three randomizations. Average events within session/
randomization, randomizations within session, sessions within animal, then
animals equally. CIs bootstrap four/five animals conditional on frozen maps.

| Dataset | Full native | Half native | Half + pooled removed spikes | Half + oracle-restored labels |
|---|---:|---:|---:|---:|
| PF | 29.30% | 10.21% | 10.50% | 28.72% |
| Tanni, all arenas | 7.78% | 1.94% | 2.98% | 7.73% |

| Dataset | Paired contrast | Percentage points, 95% CI |
|---|---|---:|
| PF | Native half minus full | -19.09 [-29.11, -8.84] |
| PF | Pooled half minus full | -18.80 [-28.67, -8.93] |
| PF | Native half minus pooled half | -0.29 [-1.42, +0.90] |
| Tanni | Native half minus full | -5.84 [-7.63, -4.08] |
| Tanni | Pooled half minus full | -4.80 [-5.63, -4.14] |
| Tanni | Native half minus pooled half | -1.04 [-2.79, +0.32] |

Pooling preserves EVERY spike in EVERY fine-time bin without using true
position to generate labels. Its loss is negative in all nine animals and the
dataset-average effect is negative in all three randomizations. Conditional
decoding gives pooled-half losses of -18.51 pp PF and -5.13 pp Tanni.
Intervals spanning zero for native-minus-pooled are not equivalence tests.

Preserving aggregate spike counts is insufficient: losing individual spatially
tuned identities retains most of the continuity penalty. This is not yet an
isolation of coverage holes from field size, tuning redundancy, or other
population information. Dropping the pool loses both counts and aggregate
spatial tuning, not counts alone.

### Correction to the Earlier Restoration Interpretation

Oracle restoration nearly returns recovery to full-population levels, but
assigns labels according to KNOWN true position. It can add spatial information
and is not a deterministic reduction of recorded spikes. Neither this arm nor
the earlier imposed-count benchmark establishes that count loss rather than
cell-identity information explains the subsampling effect. The pooled arm is
the primary control. See `replay_coverage_information_controls.md`.

## 2. Gradient Attenuation at Comparable Spike Budgets

Opposite horizontal fields vary true speed between 500 and 1,500 cm/s, with
injected normalized-slope contrast 1.0. These are NOT wall-distance gradients;
PF physical walls remain unverified. Primary full-population Poisson posterior-
mean decoding uses all three randomizations, all windows before continuity
selection, and the KNOWN spatial coordinate as an optimistic diagnostic.

| Dataset | Injected arclength response | Window-mean truth response | Decoded response, 95% CI |
|---|---:|---:|---:|
| PF | 1.000 | 0.847 | 0.351 [0.210, 0.526] |
| Tanni | 1.000 | 0.810 | 0.107 [-0.011, 0.225] |

Attenuation exceeds the finite-window/reflection effect. Tanni's interval
includes zero despite a large known nonzero contrast; this does not establish
a zero biological effect.

The primary simulated/source population-rate ratios are about 1.14 PF and
1.33 Tanni, from paired profile ratios averaged through sessions and animals.
Population-rate summaries are 421 vs 364 Hz (PF) and 269 vs 208 Hz (Tanni).
These are aggregate rates across cells, not individual-neuron rates. The
primary simulation is not simply much lower in overall count than source MUA
candidates, but does not match their fine-time gain, correlations, spatial
content, or cell-activation patterns.

## 3. Higher Exposure Partly Restores Gradient Recovery

The separate paired dose series uses replicate zero ONLY at every multiplier.
Do not substitute the three-replicate primary point into this dose series.

| Multiplier | PF decoded gradient response | Tanni decoded response |
|---|---:|---:|
| 1 | 0.053 | -0.019 |
| 3 | 0.311 | 0.168 |
| 10 | 0.667 | 0.485 |
| 30 | 0.773 | 0.716 |

Truth window-mean responses remain 0.839 PF and 0.809 Tanni. High exposures
approach them, but this is not calibrated equivalence. Multiplier-30 simulated/
source rate ratios are about 11.3 and 13.1: these are information-limit stress
tests, not physiological replay-rate claims. Temporal/grid approximations
still remain at high exposure.

## 4. Nulls and Selection Denominators

Primary full-population Poisson MAP, unfiltered, geometric null acceptance:

| Dataset | Stationary | Independent snapshots | Whole-bin shuffled continuous |
|---|---:|---:|---:|
| PF | 2.4% | 0.0% | 2.5% |
| Tanni | 3.6% | 0.4% | 1.6% |

These are specific synthetic-null acceptance rates, not biological replay FPR
or event-level shuffle significance. Zero observed acceptance does not imply
zero population probability. Shuffles preserve within-5-ms-bin snapshots;
overlapping 20 ms windows average across them.

Only 13/24 PF and 2/75 Tanni session/randomization combinations support BOTH
selected-core gradients at the primary dose. They cover five of eight PF
sessions (three animals) and two of 25 Tanni sessions (two animals). The gates
of >=5 events and spatial variance >=0.01 were not relaxed. Selecting smooth
events does not automatically create sufficient or representative coverage.

## Figure and Remaining Scope

`coverage_counterfactual.png` was visually inspected: four nonblank panels,
readable labels/intervals, no clipped legends. Both rows show the paired dose
replicate. Top: conditional MAP continuity loss; bottom: Poisson posterior-mean
gradient response with known coordinates. They are not the primary three-
randomization analysis. CIs condition on the empirical maps/simulations.

The result strengthens the recording-information explanation, corrects the
oracle-restoration interpretation, and replicates poor gradient recovery at
roughly comparable counts. Still incomplete: broader cell/field/arena and
temporal/grid isolation; map-estimation and observation-mismatch validation;
independent calibration/evaluation; event-definition sensitivity; verified
wall coordinates; meaningful equivalence testing only where recovery permits.
