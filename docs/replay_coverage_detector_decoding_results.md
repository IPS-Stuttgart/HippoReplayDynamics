# Detector and Window Coverage Sensitivity

Date: 2026-09-05. This completes the frozen detector/window perturbation and
selected LFP/MUA trace inspection. It does not complete the full study or turn
geometric continuity into a shuffle-significant replay label.

## Provenance and Verification

All computation ran on gpuserver6000 in
`/home/florianpfaff/HippoReplayDynamics-recording-coverage`.

- Frozen decoding/trace commit: `13afa9403cf6240404aa78d14490004fd5473790`.
- Corrected independent-auditor commit: `dc6c2c9d48f84204569661b10c34010770823866`.
- Each invocation recorded a clean tree. The decoder and its inputs were not
  changed or rescored after results were available.
- Input: `/mnt/seagate10tb/florianpfaff/replay-coverage-event-definitions-all33-20260905-v2`.
- Decode: `/mnt/seagate10tb/florianpfaff/replay-coverage-detector-decoding-all33-20260905`.
- Non-rescoring report: `report-v2/` inside that decoding artifact, with detector
  sources explicit in the figure. The original `report/` remains unchanged.
- Traces: `/mnt/seagate10tb/florianpfaff/replay-coverage-detector-traces-20260905`.
- All 171 relevant tests and focused Ruff checks passed.

All 33 sessions completed: 28,451 eligible core/fixed windows and 910,432
condition rows. The independent audit directly recounted all 955,845 scoring
frames from cached unbinned spikes, reconstructed every path-metric row with
separate segmentation and gap logic, checked 3,580 analytic posterior frames,
verified all source/output hashes and reconstructed all summaries. Posterior
checks are sampled, not an independent analytic reconstruction of every frame.
Window variants and recording replicates are not independent biological samples.
All 132 full/half population definitions were also matched directly against
the original real-subsampling manifest: identical cell IDs, seed and replicates.

The first audit attempt stopped on 139 PF Rat4/Open1 windows because its frame
count formula omitted the frozen helper's 1 ns near-full-bin tolerance. Source
durations differ from exact multiples of 5 ms by subnanosecond amounts at large
absolute acquisition times; saved final frames end at the original event edge.
The auditor now honors that existing tolerance, with a real-number regression
fixture. No source times, decoder settings, decoded outputs or scientific
thresholds were changed. The full audit subsequently passed.

## Trace Inspection

Fifteen examples were selected before decoding: per Tanni animal, one each
from eligible MUA-only, ripple-only and overlapping cores, using median RUN-QC
spike-count rank with deterministic ties. All five four-row figures were viewed.
They show raw LFP on the lowest QC-passing native channel, its 150-250 Hz signal,
the full-recording pooled envelope z, and RUN-QC cell spikes. No time offset was
applied. Figure geometry and labels were inspected and are legible.

The MUA-only examples contain visible population bursts with subthreshold
pooled ripple envelope; ripple-only examples can contain strong band-limited
episodes with few retained-cell spikes. The examples make these assay
differences plausible but do not establish false-positive rates, prove hardware
synchronization, exclude artifacts, or validate every channel/session. Some
large raw transients remain conspicuous and need not be biological ripples.
The displayed single channel is not chosen to maximize ripple amplitude and
need not reflect the pooled detector. PF raw LFP is unavailable in this cache.

Trace generation preceded decoding; visual review occurred while the fixed
decoding run was executing, before interpreting its results. No examples or
thresholds were changed in response to the decodes.

## Primary Continuity Result

Independent flat-prior Poisson decoding, MAP, 20 ms windows/5 ms stride,
>=2 cells and >=3 spikes per frame. Same three half-cell populations and maps
across detectors/window variants. Average subsets within session, sessions within
animal, and animals equally; CIs bootstrap animals. Four PF and five Tanni
animals contribute. Tanni includes all arena sizes, not just the large arena.

| Dataset/cohort | Eligible windows | Full (%) | Half (%) | Half-minus-full pp (95% CI) |
|---|---:|---:|---:|---:|
| PF MUA core | 4,001 | 26.59 | 8.81 | -17.79 [-21.72, -12.72] |
| PF MUA 200 ms | 3,707 | 27.32 | 9.43 | -17.89 [-22.84, -12.93] |
| PF native ripple core | 2,581 | 31.93 | 13.57 | -18.36 [-23.04, -13.68] |
| PF native ripple 200 ms | 2,567 | 30.91 | 12.67 | -18.25 [-24.04, -12.46] |
| Tanni MUA core | 5,224 | 7.44 | 0.83 | -6.61 [-7.91, -5.27] |
| Tanni MUA 200 ms | 5,170 | 7.24 | 0.73 | -6.51 [-7.77, -5.04] |
| Tanni LFP ripple core, z3 | 2,635 | 2.29 | 0.31 | -1.98 [-3.25, -0.79] |
| Tanni LFP ripple 200 ms, z3 | 2,566 | 2.51 | 0.22 | -2.29 [-4.20, -0.86] |

Every animal has a negative primary half-cell effect in all four cohorts. The
effect is therefore not exclusive to the original MUA window definition.
Tanni ripple acceptance starts near the floor: its smaller absolute loss is
not evidence of greater robustness to sparse recording or different biology.
Two Tanni ripple-unavailable sessions remain explicit; MUA uses 25/25 sessions,
ripple uses 23/25, with all five animals still represented.

Within-session differences of removal effects are near zero for PF ripple vs
MUA (-0.58 pp [-1.51, +0.81] for core windows). Tanni ripple losses are smaller
than MUA losses (+4.99 pp [+3.23, +6.48], 23 common sessions). Fixed-vs-core
interactions cross zero in both datasets. These compare different candidate
sets within sessions, not randomized causal detector effects or equivalence.

The negative mean MAP removal effect persists without support filtering and
under count-conditioned decoding. However, not every sensitivity interval
excludes zero: e.g. Tanni z5 conditional/core and z3 conditional/fixed supported
results are uncertain. z4/z5 and both/only/unknown summaries are retained in the
root tables. Overlap labels use primary z3, not a recomputed z4/z5 overlap.
Sensitivity intervals are descriptive, not multiplicity-adjusted discoveries.

## Paired Speed and Measurability

Posterior-mean Poisson, same bin-support rule. Speeds compare adjacent
non-overlapping 20 ms frames and require all intermediate 5 ms frames. The
effect uses identical supported steps in full and half populations, then event
medians, subset/session averages and equal animal weights.

| Dataset/cohort | Half-minus-full speed cm/s (95% CI) | Events with common measurable steps (%) |
|---|---:|---:|
| PF MUA core | +146.27 [+122.42, +168.24] | 76.21 |
| PF MUA 200 ms | +142.91 [+118.58, +168.24] | 80.33 |
| PF native ripple core | +145.79 [+118.57, +178.28] | 79.87 |
| PF native ripple 200 ms | +143.94 [+113.76, +174.11] | 79.72 |
| Tanni MUA core | +94.02 [+36.60, +155.00] | 48.84 |
| Tanni MUA 200 ms | +54.38 [+31.34, +79.81] | 50.91 |
| Tanni LFP ripple core | -93.44 [-252.89, +49.76] | 26.42 |
| Tanni LFP ripple 200 ms | +59.32 [-30.39, +167.03] | 26.69 |

PF shows a positive paired decoded-speed shift under cell removal across the
four definitions. Tanni MUA shifts are positive but its ripple estimates are
uncertain and window-sensitive. There is no universal direction of speed
distortion established across these cohorts. These are candidate-window
readouts, not speeds of confirmed continuous replay or known true speed error.

Selected-core speed is especially sparse in Tanni half-cell decodes: only
about 0.57-0.58% of MUA events and 0.14-0.22% of ripple events have measurable
selected-core speed. Conditional means cannot be generalized to all candidates.
This strict-gap paired estimand is not the old endpoint-only speed statistic;
do not present differences from the earlier speed table as new biology.

## What This Adds, and What It Does Not

Supported: the primary real-population continuity effect survives both tested
candidate definitions and both window variants, with the same sign in all nine
animals. It is a recording/analysis sensitivity, not just a peculiarity of the
initial MUA cohort. The PF paired speed shift also survives these definitions;
Tanni speed inference remains less stable and much less available.

Still not established: shuffle-significant replay robustness, true replay
prevalence, which real speed estimate is accurate, biological uniformity,
coverage as the complete explanation of PF/Tanni differences, or a calibration
procedure transferable to new recording populations. Next is the established
shuffle-significant baseline comparison, then new-population validation and the
integrated methods/results/limitations pack. Do not retune thresholds to make
these remaining tests pass.
