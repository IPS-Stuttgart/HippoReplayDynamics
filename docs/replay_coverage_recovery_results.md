# Empirical-Map Recovery Results

Status: completed development experiment; full paper-ready study incomplete.
No biological speed-uniformity or replay-prevalence conclusion.

## Provenance and Verification

All science ran on gpuserver6000 (`workstation2`). Authoritative artifact:
`/mnt/seagate10tb/florianpfaff/replay-coverage-recovery-all33-20260905`.

- Generator/scorer: clean commit `925443f5edde5de3ff3f856958c7a210afd8cca3`.
- Reconstruction/support auditor: clean commit
  `08b9c2a40c36863be8662f70a65ba97c4b9676b7`.
- Non-rescoring paired reporter: commit `757a9d92`.
- 77 focused tests pass; Ruff and committed whitespace checks pass.
- All seven production technical gates pass. No short or failed simulation
  draws in this completed cohort.
- 33 sessions, nine animals, 971 source profiles, 4,855 truth draws, 43,695
  observation arrays, and 349,560 repeated analysis rows. These are not 349,560
  independent events. Tanni includes all arena sizes, not just the large arenas.
- One small-arena session, R2481 / 2019-04-29_16-26-33, had only 11 candidates;
  all 11 were used. The other 32 sessions contributed 30 uniformly sampled
  candidates each. No duplication to reach the cap and no decoded-success retry.
- The auditor reconstructed every path/count-array hash and independently
  recounted overlapping windows and non-overlapping spike-support pairs for
  every metric row. Zero mismatches. This verifies implementation/artifact
  consistency, not biological realism or an independent validation of the model.
- Inputs and code are snapshotted; scoring commit archive is
  `scoring-code-925443f5.tar.gz`. A separate three-session dirty-tree smoke was
  a technical check only and is not pooled into this result.

The six-panel `coverage_recovery_gradients.png` was visually inspected. Its
shaded intervals are animal-bootstrap uncertainty conditional on these fixed
simulation draws, not confidence intervals on biological speed.

## 1. Cell Removal Is Not Pure Spatial-Coverage Loss

Constant true speed 1,000 cm/s, MAP geometric criterion, no per-bin support
filter. Recovery here means the fraction of continuous draws whose known
window-mean path passes the geometric rule that also pass after decoding.
The comparison is paired within session, then averaged within animal and
equally across animals. CIs bootstrap the four/five animals.

| Dataset | Intervention / decoder | Half-minus-full recovery (percentage points), 95% CI |
|---|---|---:|
| PF | Poisson generation, remove cells and their spikes; Poisson decoder | -20.64 [-33.20, -8.82] |
| PF | Same generation/removal; conditional decoder | -19.68 [-30.78, -7.46] |
| PF | Preserve each source 5 ms population count across subsets; conditional decoder | +5.93 [+0.42, +11.54] |
| Tanni | Poisson generation, remove cells and their spikes; Poisson decoder | -6.73 [-11.20, -3.32] |
| Tanni | Same generation/removal; conditional decoder | -5.75 [-10.64, -1.52] |
| Tanni | Preserve each source 5 ms population count across subsets; conditional decoder | +0.85 [-1.52, +3.76] |

Full/half recovery under primary Poisson generation/decoding is 31.5%/10.8%
for PF and 8.6%/1.9% for Tanni. Under fixed counts and conditional decoding it
is 28.0%/34.0% and 5.7%/6.5%, respectively.

The large loss under native cell removal is absent under the fixed-count
intervention, including when the likelihood is held constant. Therefore the
earlier real-cell thinning result must not be attributed automatically to holes
in spatial cell coverage. Available spike information matters substantially in
this surrogate setting. This is NOT a complete causal decomposition of the
real-data effect: count restoration redistributes spikes, the regimes have
different total-count processes, maps are surrogates, and only one nested
population ordering/session was tested. It does not prove spatial coverage is
unimportant, or that fewer neurons improve true biological replay decoding.

Using an unconditional Poisson likelihood on the imposed fixed totals changes
even the PF direction (-3.41 pp). That intentionally misspecified sensitivity
must not replace the conditional-likelihood result.

## 2. Strong Spatial Speed Gradients Can Look Nearly Flat

Truth is v(x) = 1,000 * (1 + g*q(x)) cm/s with g = -0.5, 0, +0.5 and q running
from -1 to +1 horizontally across the simulation domain. Thus the nonuniform
conditions span 500 to 1,500 cm/s, with opposite directions. These are NOT wall
gradients. PF domain is encoding-grid extent, not verified physical walls.

Primary Poisson generation/decoding, full cells, posterior-mean speed,
unfiltered and before continuity selection. Regression uses the KNOWN spatial
coordinate as an optimistic diagnostic, not an available real-data covariate.
The table is the paired fitted-slope response to switching g from -0.5 to +0.5.
The injected slope difference is 1.0.

| Dataset | True arclength response | Truth window-mean chord response | Decoded response, 95% animal-bootstrap CI |
|---|---:|---:|---:|
| PF | 1.000 | 0.860 | 0.477 [0.248, 0.635] |
| Tanni, all arenas | 1.000 | 0.806 | 0.066 [0.003, 0.129] |

The numerical true-field regression matches its injected gradient to within
1.14e-5 across unfiltered session conditions. Reflection and window averaging
attenuate chord gradients somewhat, but do not explain the much greater
attenuation after decoding. Tanni has a small detectable paired response under
this Poisson setting; it is not accurate recovery of the injected magnitude.

In the fixed-count/conditional setting, responses are PF 0.331 [0.214, 0.449]
and Tanni -0.016 [-0.168, 0.108]. Shared-gain mismatch weakens them further:
PF 0.177 [0.033, 0.403], Tanni -0.042 [-0.143, 0.102] under Poisson decoding.
These are development sensitivities, not independently replicated calibration.

Both individual nonzero injected Tanni gradients can yield fitted slopes near
zero. A flat decoded profile therefore cannot be interpreted as physical-speed
uniformity under the current pipeline. This is a limit of this measurement
setup, not proof that real replay speeds are nonuniform.

## 3. Selection Does Not Automatically Fix Identifiability

The reporter separately examines only steps inside accepted continuity cores.
It abstains at session level when fewer than five contributing events survive
or normalized spatial variance is below 0.01.

With full cells, posterior mean, no per-bin support exclusion, and Poisson
generation/decoding, only 5/8, 5/8, and 6/8 PF sessions support a selected-core
gradient for g=-0.5, 0, +0.5. Tanni has only 3/25, 3/25, and 1/25, respectively.
Shared-gain Tanni has 0/25 at all three gradients. Do not turn surviving-session
averages into all-session evidence of uniformity. More source profiles and
independent draws are required before a selected-event recovery claim.

## 4. Decoded Step Speed Has a Noise Contribution

Full-cell, Poisson, posterior-mean, unfiltered BEFORE trajectory selection:

| Dataset | Stationary truth: decoded median-speed summary | Constant 1,000 cm/s truth: decoded summary |
|---|---:|---:|
| PF | 1,096 cm/s | 1,466 cm/s |
| Tanni, all arenas | 1,760 cm/s | 1,872 cm/s |

Summaries average session medians within animal and animals equally. They are
not medians of the selected trajectory subset. Bin-to-bin estimation noise
generates substantial apparent movement even with stationary truth. This does
not mean the criterion calls all stationary events replay: full-cell Poisson
MAP false geometric acceptance is 3.3% PF and 3.4% Tanni; discontinuous-null
acceptance is 0% and 0.7%. Other estimators/regimes are recorded separately.
These are false acceptance rates for these specific synthetic nulls, not an
empirical false-positive rate for biological replay or a shuffle significance
test. Finite-window averaging of discontinuous snapshots is itself a caveat.

## Remaining Requirements and Next Decision

The most useful immediate continuation is a frozen independent recovery run:
new paths/seeds and population orderings, more source profiles where available,
and rate/spike-information sensitivity. Repeat the paired contrasts, not a
new threshold chosen to make continuity or uniformity pass. Then test map
estimation/mismatch and any proposed bias correction on separate evaluation
draws. Controlled field density/width, arena area/aspect, bins/strides, spatial
grid, event definitions, verified wall coordinates, and calibrated equivalence
remain unfinished requirements in the study plan.

Supported now: this pipeline's continuity yield and speed readouts are strongly
information-dependent; cell-removal effects are not automatically spatial
coverage effects; and the current surrogate benchmark can substantially hide
large spatial speed gradients. Not supported: biological uniformity, complete
explanation of PF/Tanni differences, a new neuronal replay mechanism, or a
finished calibration method. The full goal remains active.
