# Independent RUN-Map and Shared-Gain Recovery

Status: scoring, production reconstruction and final report verification pass.
No biological uniformity claim.

## Frozen Experiment

Scorer commit: `9eeeb64a13c68f6a9918bc703d044914528ca616`, clean at launch.
Server: gpuserver6000. Artifact:
`/mnt/seagate10tb/florianpfaff/replay-coverage-map-mismatch-all33-20260905`.
Protocol: `replay_coverage_map_mismatch_protocol.md`.

All 33 sessions, nine animals and 66 chronological half directions scored.
The 971 frozen source duration profiles produce 11,652 truth trials, 46,608
observation arrays and 745,728 metric rows. These are repeated synthetic
conditions, not additional biological events or animals. All technical gates
and the separate full-cohort encoding-availability gate pass. Scoring took
701 seconds. The four-direction, two-profile technical smoke is excluded.

Audit and reporter also ran at clean commit `9eeeb64a`. Every half-map array
and all 46,608 observation arrays reproduced. Direct support/truth-eligibility
checks cover all 745,728 metric rows; independent likelihood, posterior-inclusion,
continuity and speed checks cover 25,344 sampled rows, not every decoded row.
All 15 final report file hashes verify, and the four-panel recovery figure was
visually inspected. The focused suite has 128 passing tests. Code archive,
installed dependency metadata, input snapshots and hashes accompany the run.

The non-rescoring `report/` directory contains equal-session/animal endpoints,
paired map contrasts, gradient response/availability, source/simulated spike
budgets, the figure and its manifest. Local copy:
`results/map-mismatch/report/` in the local study mirror. The scorer, audit and
report commands are the corresponding `scripts/*replay_coverage_map_mismatch.py`
programs using this frozen artifact directory and the protocol inputs.

Decoder A uses only one RUN half for cell QC, occupancy, grid and rate fitting;
the disjoint B half supplies the surrogate generator. Both directions are run.
Generator-known B rates and independent A rates decode identical simulated
spikes on identical A-defined cells, spatial states, support and flat priors.
The experiment does not use an HMM or temporal transition prior. Both position
extents constrain the synthetic domain; this is not real held-out replay truth.

There are no missing B-cell identities. Training-selected cells range from
66 to 216 in PF and 44 to 170 in Tanni. Median per-direction half-map correlation
is 0.886 PF and 0.872 Tanni. Correlated maps are not identical maps, and these
correlations are descriptive, not a post-hoc inclusion gate. The comparison
includes sampling, estimation noise and possible neural nonstationarity.

## Primary Paired Comparison

Native Poisson observations and Poisson decoding; full training-selected cells;
MAP, unfiltered 20 ms windows stepped by 5 ms; constant true speed 1000 cm/s.
Recovery is restricted to truth paths meeting the frozen geometric criterion.
Directions average within session, sessions within animal, animals equally.
Position error is the corresponding average of event-median errors, not a
pooled median across windows. Tanni includes all arena sizes.

| Metric | PF: known generator | PF: independent RUN | Tanni: known generator | Tanni: independent RUN |
|---|---:|---:|---:|---:|
| Eligible continuous paths recovered | 35.82% | 19.60% | 7.94% | 6.80% |
| Position error, cm | 15.16 | 21.81 | 28.68 | 34.72 |
| Nominal 95% HPD inclusion of true position | 85.49% | 76.15% | 90.57% | 84.82% |

The PF recovery decrease is present in all four animals. The Tanni decrease
is small on average and is not uniform across animals. Do not turn this into
a universal recovery-loss claim merely because position error worsens.
The paired independent-minus-known differences are -16.21 percentage points
(animal-bootstrap 95% interval [-22.65, -8.66]) PF and -1.14 points
([-2.45, +0.09]) Tanni. These intervals condition on the fixed maps and synthetic
draws; four PF/five Tanni animals, not 745,728 independent biological samples.

HPD inclusion counts truth outside A's state support as a miss. Mean true-position
support is 91.42% PF and 96.35% Tanni in this slice, identical across map
conditions. Therefore the nominal-95% shortfall combines support restriction,
within-window motion/discretization and model mismatch; it is not a pure
calibration-error estimate. The generator-known decoder is rate-oracle only.

## Speed-Gradient Recovery

For an injected positive-minus-negative gradient contrast of 1.0, full-cell
posterior-mean speed at known synthetic coordinates, before continuity selection:

| Readout | PF | Tanni |
|---|---:|---:|
| True arclength response | 1.000 | 1.000 |
| True window-chord response | 0.815 | 0.807 |
| Generator-known decoded response | 0.469 | 0.145 |
| Independent RUN-map decoded response | 0.321 | 0.121 |

Independent-minus-known response: PF -0.147 [-0.204, -0.081]; Tanni -0.024
[-0.116, +0.030]. The independent-map responses themselves have intervals
[0.204, 0.392] PF and [0.007, 0.243] Tanni. A small positive response is not
accurate recovery of the injected contrast of 1.0.

Window averaging itself loses part of the gradient; decoding loses substantially
more. These are known-coordinate recovery tests, not observed wall-distance
correlations or biological equivalence estimates. Independent maps do not
restore identifiability. Neither does failure to detect an imposed gradient
demonstrate that a biological gradient is absent.

Selected-core availability is a separate limitation: full-cell paired positive
and negative gradient estimates exist in 9/16 PF directions with known maps and
6/16 with independent maps; Tanni has 2/50 and 1/50, respectively. The latter
independent-map estimate represents only one animal. Selected-only averages
cannot be presented as a complete-cohort improvement or replicated result.

## Observation Stress

The declared shared-gain stress uses mean-one lognormal gain, CV=1, constant
within 20 ms blocks and independent of location. This is not fitted replay
covariance. With independent maps and Poisson decoding, continuous recovery
is 11.26% PF and 3.92% Tanni. Conditional decoding gives 12.63% and 3.99%.
Conditional decoding reduces intensity sensitivity but does not provide a
general recovery or calibration solution. Native/gain observations share paths
but use separate random count draws; only the map comparison uses identical
spikes. Full tables retain every likelihood, estimator and support condition.

Under shared gain plus independent maps, the full-cell posterior-mean gradient
response falls to 0.042 [-0.076, 0.120] PF and -0.056 [-0.165, 0.017] Tanni
with Poisson decoding. This is a declared observation-mismatch sensitivity,
not evidence that biological shared gain has this magnitude or effect.
Mean simulated population rates are 439/442 Hz (Poisson/gain) versus 387 Hz
in the matched retained-unit source profiles for PF; 247/245 versus 201 Hz for
Tanni. These are comparable aggregate budgets, not matched per-event counts.
They do not match covariance, tuning drift or all characteristics of replay.

Native-Poisson/full-cell MAP geometric null acceptances (known to independent
maps) are: stationary 1.88% to 2.71% PF and 3.63% to 2.47% Tanni;
independent snapshots 0% to 0% PF and 0.53% to 0.80% Tanni; shuffled paths
2.08% to 2.08% PF and 1.47% to 1.18% Tanni. These are simulated heuristic
acceptances, not biological replay false-positive rates or shuffled p-values.

## Interpretation and Next Requirement

Known-map surrogates can overstate recoverability, substantially in the PF
primary slice. Separately estimated maps worsen localization and posterior
inclusion in both datasets, while Tanni continuity recovery remains low under
either map. Strong speed-gradient attenuation therefore survives this more
demanding surrogate test; this does not identify the true speeds of real replay.

The next useful contribution is a frozen calibration or abstention procedure,
tested on independent populations and observations, with explicit failure to
identify meaningful gradients. Event-definition sensitivity and comparisons
against established replay/calibration baselines remain incomplete. No
threshold was relaxed and no biological uniformity claim is authorized.
