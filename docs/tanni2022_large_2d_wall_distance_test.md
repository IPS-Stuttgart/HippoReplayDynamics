# Tanni et al. 2022 large-2D wall-distance test

## Question

Does decoded motion during awake ripple-associated events vary with represented distance from the nearest wall in the 350 x 250 cm arena?

The test was designed around the main confound created by this dataset: spatial coding resolution changes with wall distance. A raw decoded speed in cm/s can therefore vary even when the underlying population-code dynamics do not.

## Data and frozen event definition

- Dataset DOI: `10.5522/04/18128891.v1`
- Paper DOI: `10.1016/j.cub.2022.06.046`
- Animals: R2470, R2474, R2478, R2481, R2482
- Environment: each animal's largest 350 x 250 cm foraging arena
- Decoder cells: 628 manually sorted, paper-aligned place-like units across the five sessions
- LFP: all 32 CA1 channels, filtered independently at 150-250 Hz; Hilbert envelopes were robust-z-scored per channel and averaged only after envelope extraction
- Candidate event: ripple-envelope core with aggregate z >= 3, peak z >= 10, duration 15-250 ms, gaps <= 30 ms merged
- Decoding window: fixed 200 ms window centered on the ripple peak
- Event support: animal speed < 5 cm/s, at least 5 spikes from at least 3 decoder cells

This produced 9,496 spectral candidates and 3,131 immobile, spike-supported decoded events across all five animals. The full-channel detector was stable against the four-channel pilot on R2470: spectral-event precision/recall within 50 ms were 0.888/0.905; selected-event precision/recall were 0.775/0.912.

## Decoder validation

Held-out RUN decoding was finite in all five animals. Median position errors were:

| Animal | Median error (cm) |
| --- | ---: |
| R2470 | 36.35 |
| R2474 | 27.15 |
| R2478 | 26.78 |
| R2481 | 20.30 |
| R2482 | 19.86 |

The median across sessions was 26.78 cm. The retained unit total, 628, nearly reproduces the paper's reported 629 CA1 place cells and provides an independent adapter/QC check.

## Estimators and controls

The primary path estimator was the independent-bin Poisson emission posterior mean, so no diffusion, momentum, or IMM motion prior could create a wall-speed relation.

Wall distance was computed as the posterior expectation of each spatial bin's nearest-wall distance. It was not computed from the posterior mean coordinate: diffuse posteriors average toward the arena center, which would create a far-from-wall/slow relationship by construction.

Reported sensitivities include:

- posterior-mean physical speed in cm/s;
- MAP physical speed;
- independent-posterior RMS displacement speed;
- Poisson/Hellinger population-code speed;
- posterior entropy and spatial spread;
- local code gradient, occupancy, event spike/cell counts, and ripple peak;
- a matched decoder simulation with constant true physical speed.

Primary correlations use one median row per event and then balance across animals. Segment-level correlations are sensitivity rows only.

## Broad ripple-candidate result

Physical posterior-mean speed decreased with posterior-expected wall distance in every animal:

- raw animal-median Spearman rho: -0.291;
- quality-adjusted partial rho: -0.147;
- animal-bootstrap interval: [-0.338, -0.071].

That association is not specific to replay dynamics. The constant-true-speed decoder null produced a median decoded-wall/decoded-speed rho of -0.349 with interval [-0.416, -0.248], which contains the observed raw effect.

The estimator sensitivities also fail to converge:

- code-space speed adjusted rho: -0.016, interval spanning zero;
- MAP speed adjusted rho: +0.003, interval spanning zero;
- independent-posterior RMS speed adjusted rho: -0.373.

The deepest represented-wall quartile is sparse (51 event-quartile observations), so it must not carry a monotonic environment-wide claim.

**Broad-event verdict:** no robust biological wall-distance speed association. The observed physical-speed pattern is compatible with decoder geometry and depends materially on the path estimator.

## Exact-model subset stop gate

A deterministic, pre-model subset was sampled within animal and represented-wall strata and scored under stationary, diffusion, fragmented, and first-order IMM models on a coarse exact 2D grid. There were 141 scoreable selected events because no animal had event-median posterior-expected wall distance in the nominal farthest quartile.

Best-model counts were:

| Model | Best rows |
| --- | ---: |
| stationary | 75 |
| diffusion | 36 |
| fragmented | 16 |
| first-order IMM | 14 |

Only 5/141 events confidently favored ordered dynamics (diffusion or first-order IMM) over both stationary and fragmented at the 5.5 log-evidence margin, spanning 3/5 animals. This is too small for an ordered-replay wall-speed analysis.

Fifty-three events had first-order IMM at least 5.5 above fragmented, but this is not a rescue criterion: stationary remained competitive or best for many of them. The clean comparison must beat both stationary and fragmented.

## Claim boundary

The large-arena dataset is technically supported and provides a useful specificity result, but it does not currently support the claim that replay slows or accelerates as a function of wall distance.

The result also illustrates why the original place-field scaling finding does not already imply a replay-speed finding: physical decoded speed, code-space speed, MAP speed, posterior-mean speed, and uncertainty-aware displacement are distinct quantities, and the decoder itself can generate a wall-dependent physical-speed correlation under constant true speed.
