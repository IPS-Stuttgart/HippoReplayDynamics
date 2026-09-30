# Detector versus decoder population sampling: outcome

Date: 2026-09-17. Status: completed methodological stress test, not biological ground truth.

## Question and frozen design

Does changing the recorded population alter which population events are detected,
what a decoder says about unchanged windows, or both? This run crosses the full
RUN-qualified cell population and each subset in both roles. The underlying
spike train, RUN maps, valid spatial grid and detector rule are unchanged across
arms. Each detector uses its own population activity and its own baseline.

The primary region is the central tile of a 3-by-3 partition of valid-grid bounds.
It is NOT Home, a reward well, or an independently established replay target.
Targeted high/low halves are ranked using RUN regional coverage only. Two random
half-partitions and two whole-tetrode partitions are controls. These targeted
halves intentionally impose extreme coverage differences; they do not represent
typical missing-cell patterns. Whole-tetrode halves can differ in cell count.

Cohort: all 8 PF recordings / 4 rats and all 25 Tanni recordings / 5 rats in the
frozen cache, including different Tanni arena sizes, not just the large arena.
All 33 recordings support all 11 population configurations. Full-population
detectors found 3,734 real PF and 8,583 real Tanni MUA candidates. These are not
verified continuous trajectory events and need not reproduce historical counts.
Here, full means RUN-qualified sorted cells, not every recorded unit or all
unsorted MUA. Readout is mean central posterior mass in the last full 20-ms window
on the native event's 5-ms decoding grid. Silent windows are retained.

Known-content simulations use the actual RUN maps, region-blind stationary or
moving paths, independent Poisson spikes, gains 3 and 6, and 3 replicas per panel.
There are 118,800 two-second synthetic epochs, with an identical burst-gain time
profile regardless of region. The moving generator's 500 cm/s is a declared
stress parameter, not an inferred biological speed. All population arms use the
same generated spike train. The 429 sources comprise 33 real + 396 synthetic
panels. Parameters were frozen in `detector_decoder_cross_protocol.md` before
the production run.

## Real-data crossed result

Equal-weight animals; repeated subsets/replicas average within recording, then
recordings within animal. Units are percentage points of mean central posterior
mass, not percentage-point changes in verified replay prevalence.

For the central-coverage-poor half and the Poisson likelihood:

| Dataset | Detector only | Decoder only | Interaction | Both changed |
| --- | ---: | ---: | ---: | ---: |
| PF | -10.22 | -6.33 | +4.28 | -12.27 |
| Tanni | -3.37 | -1.52 | +1.07 | -3.82 |

Detector only means subset detector/full decoder versus full/full. Decoder only
means full detector/subset decoder versus full/full, using identical windows.
The positive interaction means the two losses must NOT simply be added.
Both main effects and the combined decrease are negative in all 4 PF and all
5 Tanni animal averages. PF's full/full mass is 15.13%, versus 2.86% for low/low;
Tanni's is 6.74% versus 2.92%.

The descriptive animal-bootstrap 95% intervals for the combined contrast are
[-15.34, -9.21] pp (PF) and [-4.62, -3.23] pp (Tanni). There are only 4/5 animals;
these are not a multiplicity-adjusted biological significance claim.

Random and whole-tetrode controls are smaller on average. Mean absolute combined
contrasts at the recording/partition-side level are 1.34-2.21 pp for PF and
0.95-1.32 pp for Tanni, depending on control family/side, versus 12.27 and 3.82 pp
for the targeted-poor halves. These absolute summaries average the two repeats
within recording before taking absolute values; they are not worst-case values.

## Known truth: selection is genuinely spatially biased

Stationary generator, gain 6; successful detection means a native candidate peak
within 150 ms of the synthetic gain peak. These rates use independently known
inside/outside content, not full-population decoded labels.

| Dataset | Full: central burst detected | Poor half: central detected | Poor half: outside detected |
| --- | ---: | ---: | ---: |
| PF | 97.81% | 2.45% | 76.71% |
| Tanni | 96.04% | 13.19% | 76.51% |

Random halves detect central bursts at 90.69-91.68% (PF) and 88.30-90.41%
(Tanni); whole-tetrode controls at 85.79-94.54% and 87.11-90.48%, respectively.
Poor central coverage does not merely lose events indiscriminately.

With no change in the generated content, the true central fraction of selected
stationary endpoints falls from 13.60% to 0.40% in PF and from 9.47% to 1.43% in
Tanni. Thus a perfect decoder of the detected windows would still see a biased
event sample. This is selection, not a decoder misclassification.

The decomposition of the combined Poisson difference is:

| Dataset | True selected-content shift | Change in decoding bias | Combined decoded shift |
| --- | ---: | ---: | ---: |
| PF | -13.20 pp | +1.65 pp | -11.55 pp |
| Tanni | -8.04 pp | +1.43 pp | -6.61 pp |

The positive second term partly masks the selection loss; it does not correct it.
True selection shifts remain negative in every animal for both gains and both
generators: PF -13.20 to -20.15 pp, Tanni -8.04 to -14.99 pp across these panels.
For moving trajectories, changed endpoint timing contributes to true selection
shift as well as changed event membership.

## Important sensitivities

- Conditional-multinomial decoding removes the location-dependent total-rate
  term but does not alter detection. On real data the poor-half detector effects
  remain -10.24 pp PF / -3.40 pp Tanni, but the decoder-only effects become
  -3.53 / +1.04 pp. The combined effects become -4.60 / +0.62 pp. Tanni's latter
  descriptive interval spans zero [-0.07, +1.31] pp. Do not claim an invariant
  direction or size of decoding bias across observation models.
- The synthetic true-content selection effect is independent of which decoder
  likelihood is subsequently applied. This is the strongest conclusion here.
- Among uniquely overlap-matched real events, the full-decoder change caused by
  altered boundaries alone is -2.50 pp PF / -1.27 pp Tanni. Median endpoint shifts
  averaged across recording/animal are +0.38 / +0.80 ms. Small median shifts do
  not prove negligible effects in individual sparse terminal windows. Matched
  events form a selected subgroup, not an unbiased substitute for the factorial.
- Real low/low silent-terminal-window rates average 0.20% PF / 1.49% Tanni;
  mean active cells are 2.18 / 1.79. Neither silence nor weak support was hidden
  by a post-decoding filter. Strict endpoints remain sparse measurements.
- The Poisson decoder does not fit the simulated shared burst gain. The
  conditional likelihood is an explicit sensitivity, not a calibrated remedy.
  Simulations assume independent spiking given position/gain and frozen RUN
  tuning. They establish possibility and mechanism under those assumptions,
  not the true generative process of biological replay.
- The targeted halves also differ in tuning/rate properties; equal cell counts
  alone do not isolate one causal feature of coverage from every correlated
  population property. The extreme effect sizes are not estimates for ordinary
  recordings. Full real-data decoding is a reference, never spatial truth.

## Verification and provenance

All 33 session jobs and all 429 source panels passed. There are no missing
factorial contrasts. The independent audit reran every native detector, recounted
raw spikes, reconstructed truth occupancy and likelihoods, and checked hashes,
population identities, grid support, denominators and factorial algebra.
It checked 1,384,370 detector-event rows and 10,356,332 decoded window readouts.
Additional checks verified all true-content decompositions, maximum-overlap
links and 1,306,800 epoch-detection records. Twelve links had tied overlaps within
1e-8 s; one argmax changed under CSV float round-trip. The timing-only reporter
excludes ties and non-one-to-one matches; the main contrasts do not use matches.
Regression tests: 39 passed; scoped Ruff checks clean. Final figure inspected.

Research root on gpuserver6000:
`/home/florianpfaff/HippoReplayDynamics-content-stability-20260914`

Base commit: `8b8473018da6ab5c293b13600399be41686e6c15`, branch
`test-cross-dataset-content-stability`, dirty with explicit script/input hashes.
No claim that the new files are represented by that base commit alone.

Artifacts under `/mnt/seagate10tb/florianpfaff/`:

- `detector-decoder-cross-all33-20260917`: frozen cohort, maps/populations, raw
  simulated trains, all native candidates/readouts, truth and manifests (2.3 GB).
- `detector-decoder-cross-all33-audit-20260917`: independent reconstruction audit.
- `detector-decoder-cross-all33-report-20260917`: non-rescoring tables and figure.

## Decision

This is a positive methodological diagnostic: recording composition can change
regional content through both candidate selection and decoding, and the
known-content experiment confirms genuinely selective event loss in both sets
of empirical maps. The more robust message is detection bias; decoder effects
are likelihood-dependent. Improving a decoder alone cannot reconstruct events
that its sampled population did not detect.

Do not claim corrected Home preference, calibrated real regional prevalence,
new replay biology, or a demonstrated first-in-literature contribution. This
experiment does not reopen the correction stopgate. A paper argument would
need to position this quantitative decomposition against prior sampling-bias
work and establish its size under more realistic recording losses, rather than
presenting these deliberately extreme halves as typical populations.
