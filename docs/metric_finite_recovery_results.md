# Finite-Spike Metric Recovery: Not Ready

2026-09-09. This is progress on the research search, not a high-importance
discovery or a completed biological mechanism test. No real replay event was
rescored. The preceding idealized screen passed; this finite-spike follow-up
fails its frozen practical readiness criteria.

## Provenance And Verification

- Producer/protocol commit: `304cc6e56b9d2668e9f16fa26cf92c5be80368a7`.
- Independent verifier commit: `ec0fc918aaf1170d34cee79ab96ab81d558175a8`.
- Server: gpuserver6000.
- Run: `/mnt/seagate10tb/florianpfaff/metric-finite-recovery-all33-20260909`.
- Run manifest SHA256:
  `ad202b19f9b187f85f83cb5700f46860b43ecd679764d6278cc5cabb0c5099f8`.
- Passing audit: sibling `-audit-v2/metric_finite_recovery_audit.json`.
- Report/figures: sibling `-report/`.
- 33 original RUN maps, 9 animals, five saved cell partitions.
- 8,448 latent path/profile families: 64 profiles per recording, each with
  physical, neural, stationary and iid-position generators.
- 506,880 score rows across partitions, observation conditions, count support
  and origin arms. These are not independent biological events.
- All 2,027,520 model scores independently reconstructed; maximum error
  `6.963318810448982e-13` nats. All 1,320 saved observation arrays, 396 kernels,
  101,376 split-reduced trial records, source/output hashes, calibration
  thresholds, bootstrap summaries and decisions checked.

The first audit completed all score checks but stopped because CSV drops the
pandas column-axis name `winner`. The verifier now ignores that metadata only,
retaining checks of actual labels, keys and values. A regression test covers
the distinction. The failed audit log is preserved; simulation outputs were
not altered or regenerated. The final relevant suite passed 129 tests; Ruff,
format and whitespace checks passed. Logs are in sibling `-validation/`.
Both PNGs were inspected visually and checked for nonblank pixels: dimensions
2340x1260 and 1980x720, RGB standard deviations approximately 40-41.

## Primary Results

Native observed count profiles, independently decoded origin, matched maps:

| Metric | Pfeiffer/Foster | Tanni |
|---|---:|---:|
| Physical/neural balanced accuracy | 54.88% | 51.63% |
| Conditional Monte Carlo 95% interval | 50.88-58.89% | 49.25-53.88% |
| Lowest animal balanced accuracy | 51.17% | 49.22% |
| Detection power: physical generator | 8.20% | 2.63% |
| Detection power: neural generator | 6.25% | 2.13% |
| False positives: stationary generator | 3.91% | 1.88% |
| False positives: iid positions | 0.39% | 0.75% |
| Frozen readiness | fail | fail |

Calibration uses separate stationary/iid simulations; detection power is on
all evaluation windows, not a selected subset. The 50% power requirement is a
predeclared practical floor, not a universal identifiability theorem. Empirical
null rates below 5% are not a confidence guarantee of population FPR below 5%.

PF's small matched-map accuracy advantage disappears under the fixed
observation stresses: 50.78% [46.97,54.79] with cell-gain drift and 51.37%
[47.36,55.47] with map error. Tanni remains near chance: 52.09% [49.72,54.41]
and 51.09% [48.84,53.38], respectively. All six native-condition readiness rows
are false. No threshold, animal set or event-profile selection was changed.

## Information Sensitivities

Matched-map physical/neural balanced accuracy:

| Origin/support | PF | Tanni |
|---|---:|---:|
| Decoded, native | 54.88% | 51.63% |
| Decoded, fourfold counts | 53.91% | 52.06% |
| Known origin, native | 57.32% | 51.22% |
| Known origin, fourfold counts | 54.00% | 55.66% |

Do not interpret the nonmonotonic Monte Carlo estimates as evidence that more
spikes hurt. Fourfold counts have fresh identity draws, and the training-only
neural kernel is still an imperfect estimate of the full-population neural
teacher. Neither sensitivity provides robust high recovery. Median evaluation
held-target count is 10 for PF and 7 for Tanni; corresponding training-origin
counts are 18 and 15. These are sums across scored targets/origins, not per-bin
counts or counts of distinct cells.

## What This Does And Does Not Say

The proposed 40-ms conditional-identity forecast is not ready to assign a
physical-distance versus neural-similarity mechanism to real events. Soft or
fuzzy classification cannot by itself create the missing discrimination.
This simulation does not directly test a fuzzy continuity rule, constant
physical speed, anatomical neural-sheet speed, or speed uniformity.

The neural metric is Hellinger distance between RUN conditional cell-identity
distributions. Physical and neural transitions match dwell, uniform equilibrium
occupancy and mean off-diagonal entropy, not every physical jump-length moment.
They generate stochastic paths. The observation stresses are controlled
perturbations, not measured uncertainty from independently refitted RUN halves.

The intervals condition on the sampled maps/perturbations/calibration and
resample profile/trial indices jointly across generators. They do not support
a rat-population biological claim. The full source cohort is not selected on
trajectory continuity or model preference, but only the sampled count profiles
enter these synthetic experiments.

## Next Discriminator, Not A Rescue

Before calling this an information limit, distinguish three remaining causes:

1. Teacher separation: classify the saved true latent paths using their exact
   physical versus full-population neural transition probabilities. This gives
   a latent-path diagnostic, not a spike-based result.
2. Observation limitation: integrate whole-event spike probabilities under the
   exact matched generating models, using all simulated identities and every
   time bin. With equal priors this is the optimal binary classifier for these
   specified generators; its empirical accuracy can still have Monte Carlo
   uncertainty. Compare native/fourfold support, including static/iid controls.
3. Forecast/metric-estimation loss: compare the existing forecast with one
   using full-population RUN geometry. RUN maps and held replay spikes are
   different information sources; using held cells' independent RUN maps need
   not leak held replay activity. Label this geometry condition explicitly.

Freeze those simulation-only diagnostics before computing them. Do not loosen
the current readiness gates, interpret a favorable sensitivity as the primary,
or select real replay events to obtain a cleaner outcome. A weak oracle result
would bound this particular generator comparison, not all possible neural versus
physical dynamics models. A strong oracle result would instead motivate a better
estimator and a new held-out recovery experiment.

## Novelty Boundary

Associative/neural versus physical geometry in replay is not a new broad idea.
Evans and Burgess (2019) already relate place-field associations and
discriminability/Bhattacharyya-like distances to metric structure:
https://proceedings.neurips.cc/paper_files/paper/2019/file/aa68c75c4a77c87f97fb686b2f068676-Paper.pdf

Diekmann and Cheng (2023) already examine replay driven by structural knowledge
and experience statistics:
https://elifesciences.org/articles/82301

These sources do not establish novelty for the present implementation, nor does
the present failed recovery establish a publishable impossibility result. A
substantive future contribution would require an independently recoverable,
robust empirical discriminator beyond these existing concepts. The research
goal remains open.
