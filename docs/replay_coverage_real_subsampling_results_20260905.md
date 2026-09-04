# Real-Recording Coverage Perturbation

Date: 2026-09-05. Status: substantive progress; full calibration study incomplete.

## Provenance and Scope

All computation used gpuserver6000 (`workstation2`).

- Worktree: `/home/florianpfaff/HippoReplayDynamics-recording-coverage`.
- Input-cache commit: `038aadb5828bfa23e172bbc89afc75180b961f70`.
- Paired-decoding commit: `eaf567b504c4ef4a16d295e1c0fef4eb0ce564b0`.
- Both production manifests record clean trees at invocation and unchanged
  hashed input code/data. The later documentation commit is not the run commit.
- Input artifact: `/mnt/seagate10tb/florianpfaff/replay-coverage-real-inputs-all33-20260905`.
- Decoding artifact: `/mnt/seagate10tb/florianpfaff/replay-coverage-subsampling-all33-20260905`.
- Focused verification: 42 tests, Ruff, and staged whitespace check passed.
- Every input-readiness and subsampling technical gate passed.

The cache retains ALL supplied immobile high-MUA core windows: 4,069 PF events
from eight sessions/four rats, and 8,072 Tanni events from 25 sessions/five rats.
The latter includes several arena sizes per animal and repeated small-arena
sessions; only 3,554 of these candidates belong to the five large-arena sessions.
It is not an 8,072-event large-arena cohort. No source candidate was removed on
the basis of continuity, posterior content, or replay evidence.

An independent check compared cached counts against direct counts from the
unbinned sorted spike tables, and summed bin durations against source event
durations. There were zero discrepancies across all 12,141 candidates. There
were also no overlapping source core windows within a session.

RUN-only QC retained 63-213 units/session in PF and 48-165 in Tanni. All sorted
units, failed-unit QC rows, spike times, and rates remain cached. This common
pipeline intentionally differs from the old dataset-specific decoders; these
numbers are not a bitwise reproduction of the earlier 1,264/226 trajectory counts.

## Experiment

Score the same events with all QC units or nested 75%, 50%, and 25% recording
subsets, using three deterministic session-level population permutations. The
detector and place maps are not refitted or retuned after removing cells.
Use independent uniform-prior Poisson decoding and a count-conditioned
multinomial sensitivity, MAP and posterior mean, with/without the >=2 active
cells / >=3 spikes per-bin rule. No temporal model is used.

There are 971,280 metric rows: 12,141 distinct candidates times 80 analyses.
These rows are NOT independent events or biological replicates.

Continuity uses overlapping 20 ms windows at 5 ms strides; speeds use adjacent
non-overlapping 20 ms windows. Missing support is never bridged. The continuity
rule is geometric, not a shuffle-significant replay definition.

## Continuity Result

Poisson likelihood, MAP criterion. Percentages are session-averaged within
animal and then equally animal-weighted, NOT simple pooled event fractions.
Differences are paired to each event's full-population reference before
aggregation. Intervals bootstrap animals after averaging recording replicates.

| Dataset | Per-bin support rule | Full units | Half units | Paired difference, percentage points (95% CI) |
|---|---|---:|---:|---:|
| PF | None | 28.71% | 11.79% | -16.92 [-20.54, -12.58] |
| PF | >=2 cells, >=3 spikes | 26.39% | 8.73% | -17.66 [-21.64, -12.66] |
| Tanni, all arenas | None | 9.80% | 2.87% | -6.94 [-8.36, -5.51] |
| Tanni, all arenas | >=2 cells, >=3 spikes | 7.09% | 0.74% | -6.35 [-7.99, -4.70] |

Half-population continuity loss is negative in all four PF rats and all five
Tanni rats, both with and without the per-bin support rule. It therefore cannot
be explained solely by more windows falling below that rule. The result also
depends on decoding/continuity estimation under reduced observations.

With the support rule and half the units, the PF loss of originally passing
candidates is 19.64 percentage points while newly passing candidates contribute
1.98 points; the net change is -17.66. Corresponding Tanni quantities are 6.65
and 0.30 points. A newly passing candidate is not a newly established true replay.

## Speed Result and Denominator Warning

Posterior-mean, Poisson, NO per-bin support filter, non-overlapping windows.
Entries are equally animal-weighted means of session medians of event-median
speeds. They describe candidate-event decoded readouts, not actual latent
replay velocities. These summary differences are not paired effect-size CIs.

| Dataset | Full units | Half units | Quarter units |
|---|---:|---:|---:|
| PF | 1,141 cm/s | 1,482 cm/s | 1,392 cm/s |
| Tanni, all arenas | 1,795 cm/s | 1,670 cm/s | 1,144 cm/s |

Apparent speed does not respond monotonically or in the same direction across
the datasets. The underlying source candidate windows have not changed. This
is measurement sensitivity, not evidence that the biological trajectories
changed speed. Latent truth is unavailable for these recordings, so this
intervention does not establish which estimate is closer to the true speed.

The per-bin support filter changes which speed windows/events remain measurable.
At full/half/quarter populations the measurable-event fractions are about
97%/78%/44% in PF and 93%/54%/15% in Tanni. Accordingly, the filtered speed
curves combine altered decoding with altered measurement support. Do not
interpret them as paired comparisons of identical time-bin sets.

The figure `coverage_subsampling.png` shows the >=2 cells/>=3 spikes condition:
top panels are MAP continuity fractions; bottom panels are full-candidate
posterior-mean speed summaries (not speeds restricted to selected replay).
Bands are 95% animal-bootstrap intervals; only four/five animals are available.
The count-conditioned likelihood is a sensitivity, not a demonstrated preferred
observation model for actual replay. The figure was visually inspected.

## Data/Method Limits Found

- Some occupied Tanni edge bins have centers outside the physical arena. The
  decoder excludes those centers using native arena metadata and records their
  number. This mask is fixed across cell subsets. A boundary-centroid or grid
  alignment sensitivity is still necessary for a wall-distance study.
- PF physical wall coordinates remain unverified here. Tracking extrema are
  explicitly not treated as walls; no PF wall-distance claim is made.
- There are rare extreme finite-difference RUN speeds. Fractions above 200 cm/s
  are at most 0.010% of moving frames in PF and 0.056% in Tanni. Quantiles and
  maxima are exposed, not silently filtered; tracking/speed sensitivity remains.
- Tanni's firing-rate/stability filter is not equivalent to PF's additionally
  supplied excitatory-cell label. Waveform-type and inclusion sensitivity remain.
- All-RUN unit stability must not be used to claim leakage-free held-out RUN
  decoding. A training-only inclusion/support procedure is still required.
- Subsampling removes both spatially distributed cells and total spikes. It
  does not isolate coverage geometry from spike-count information.
- The candidate detector is held fixed. Detection-rate changes under a smaller
  recorded population have not been measured by this intervention.

## Current Claim Boundary

Supported: reducing the recorded population changes continuity decisions on
the SAME real candidate events, across both datasets and every recorded animal;
the continuity loss persists without per-bin spike-support filtering. Estimated
candidate-event speeds are also sensitive to population sampling and analysis
selection. This makes naive cross-dataset comparisons unsafe.

Not established: that coverage explains the observed PF/Tanni biological
difference, that rejected candidates are genuine continuous replays, the sign
of real speed bias, physical-speed uniformity, or a new replay-generation
mechanism. The study is not yet paper-ready.

Next required work: training-only RUN decoder validation; common absolute cell
counts and within-animal arena-size contrasts; one-draw, actual-map simulations
of continuous/stationary/discontinuous paths; matched observation likelihoods;
and recovery of both positive and negative spatial speed gradients on independent
trials. Only after those controls should biological equivalence be evaluated.
