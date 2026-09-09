# Directional Versus Reversible Forecasts: Verified Results

2026-09-09. Follow-up to the frozen occupancy/dwell control, not independent
confirmation. The primary two-dataset directional criterion FAILED. No
threshold, horizon, model, or event selection was changed after scoring.

## Population And Provenance

- 9,225 frozen immobile high-MUA candidates, not 9,225 validated replays.
- Pfeiffer/Foster: 4,001 candidates, eight recordings, four rats.
- Tanni: 5,224 candidates, 25 recordings, five rats.
- Five chronological event folds and five neural partitions; existing K50
  neural HMM fits, RUN maps, calibration gains, and partitions reused.
- 691,875 event/split/model/horizon rows, including explicit too-short rows.
- Primary endpoint: proper future held-cell identity log-score advantage
  per held-out target spike, at a 40-ms center lag with a 20-ms unobserved
  gap. This is not prediction of total burst amplitude or physical location.
- Event medians across partitions precede equal session and animal means.
  Intervals are conditional animal bootstraps, not independent-event CIs.
- Scoring commit: `683340a0021e96012934dda3602488eddaab095d`, clean tree.
- Audit/report commit: `fe5a380caae5314e2aedb111016f7a423ed86732`, clean tree.
- Run: `/mnt/seagate10tb/florianpfaff/reversible-neural-forecasts-all9225-v2-20260909`.
- Run manifest SHA256:
  `707dcdb7ac9ca840b62736ea717fd76c53e4c81707dafe2dc314778113ff15ab`.
- Parent occupancy-control manifest SHA256:
  `da514a563fc1ac1b9e3704cdf6b8a95b14b88049e14eebaf84901739fd07d980`.
- Scoring runtime: 178.65 seconds, eight workers on gpuserver6000.

## What Was Changed

The original transition matrix A was compared with its stationary reverse
R and additive reversiblization B = (A + R)/2. B preserves the original
equilibrium distribution, every state self-transition, and every undirected
stationary connection strength, while removing net directional currents.

These are changed forecast kernels, NOT reversed observations, time-bin
shuffles, or backward inference. All start from the same original
forward-filtered training-cell state. No intervening/target training spikes
or held-out spikes update that state. The original model and reversible
alternative were not independently refitted or optimally filtered.

## Primary Results

All numbers below are nats per held-out target spike; brackets are 95%
conditional animal-bootstrap intervals.

| Learned neural HMM contrast | Pfeiffer/Foster | Tanni |
| --- | --- | --- |
| Original minus reversible | +0.006384 [0.003828, 0.008940]; 4/4 positive | -0.001254 [-0.003436, 0.001045]; 1/5 positive |
| Original minus stationary-reversed kernel | +0.042318 [0.031474, 0.048895]; 4/4 positive | +0.013494 [0.008196, 0.017580]; 5/5 positive |
| Reversible minus occupancy/dwell null | +0.083177 [0.071621, 0.094733]; 4/4 positive | +0.015333 [0.003568, 0.026400]; 4/5 positive |

Thus an inversion of learned direction hurts prediction in every animal,
but retaining direction does not consistently improve on a direction-neutral
kernel in Tanni. Reversible connections retain substantial predictive value
in PF and a heterogeneous positive pooled contrast in Tanni. The latter is
not a uniform two-dataset result: Tanni R2478 remains negative.

The two direction contrasts are not interchangeable. Worse reversed-kernel
prediction does not establish that directional flow is needed; averaging
directions can retain or improve prediction. Reversible processes can still
produce ordered individual paths in either direction. Conversely, failure
to beat B does not prove biological time-reversibility or absence of replay.
The possibility of regularization by reversiblization is not excluded.

## Animal And Normalization Sensitivity

Primary original-minus-reversible estimates by animal:

| Dataset | Animal | Nats / held-out spike |
| --- | --- | ---: |
| Pfeiffer/Foster | Rat1 | +0.009045 |
| Pfeiffer/Foster | Rat2 | +0.004031 |
| Pfeiffer/Foster | Rat3 | +0.003625 |
| Pfeiffer/Foster | Rat4 | +0.008836 |
| Tanni | R2470 | -0.000986 |
| Tanni | R2474 | +0.003039 |
| Tanni | R2478 | -0.004429 |
| Tanni | R2481 | -0.003352 |
| Tanni | R2482 | -0.000541 |

Raw-event scoring is a material qualification, not a reason to select a
different endpoint. PF original-minus-reversible is +0.026808 nats/event
[-0.022523, 0.076139], with only 2/4 positive rats. Tanni is +0.011054
[-0.013577, 0.038625], 3/5 positive. The small PF directional increment
is therefore not sign-uniform across normalization choices. Original minus
reversed is positive in all animals using either metric.

The 20/80-ms horizons remain sensitivity endpoints with different eligible
populations. At 20 ms, original minus reversible is inconclusive in PF and
negative in all five Tanni rats. At 80 ms it is positive in all four PF rats
and four of five Tanni rats, but this cannot replace the fixed 40-ms primary.
Medians of paired contrasts need not add after aggregation; do not infer a
percentage decomposition by dividing independently aggregated contrasts.

## Spatial Diagnostics

Spatial diffusion is reversible by construction: original/reversible/reversed
contrasts are numerical zero (pooled magnitudes at most about 5.4e-14 nats
per spike). Tiny signed numerical values are not biological animal votes.
The check passed for real and permuted maps.

Spatial IMM's original-minus-reversible contrast is near zero in PF
(-0.000053) and small/heterogeneous in Tanni (+0.001653, 4/5 positive).
Its joint position/mode dynamics have built-in asymmetry, not learned
biological direction. These diagnostic rows cannot replace the neural-HMM
primary. The previous broad forecasting criterion remains failed.

## Independent Verification And Run History

The first run, `reversible-neural-forecasts-all9225-20260909`, terminated
with `BrokenPipeError(32, 'Broken pipe')` after 20 session outputs when its
connection was lost. It is preserved with status failed; its scientific
outputs were not interpreted. The v2 rerun used the same frozen scientific
code, detached logging, and a new directory; no threshold/model changes.

The independent audit passed:

- All input/output hashes and original rows checked.
- All 231 parameter sets checked for equilibrium, diagonal, undirected
  flux, and detailed-balance constraints. Spatial adjoint checks use 16
  deterministic positions per mode, with all mode-block flux constraints.
- 23,450 new forecasts reconstructed using separate likelihood, filtering,
  and reverse-transition implementations, from the first chronological
  event in each fold across all its splits/models/eligible horizons.
- Maximum independent score discrepancy: 5.6843e-14 nats.
- All 2,075,625 split contrasts and 415,125 event contrasts, session/animal
  summaries, bootstrap intervals, and final decisions reconstructed.
- Native raw files and source fitting were not repeated by this audit.
- 85 tests passed across the new kernel/reporter, prior lagged and
  occupancy controls, learned assemblies, rate-transfer and related
  predictive controls. Ruff and whitespace checks passed.
- Non-rescoring report's six artifact hashes checked; its 2340 x 1080
  figure inspected visually and checked nonblank (RGB standard deviation
  33.44). Panels use different horizontal scales; compare numeric values.

## Claim Boundary And Research Decision

The defensible finding is narrower than a general replay mechanism:

> Reversing learned transition preferences reduces held-out neural forecast
> scores in both datasets, but removing net direction while preserving
> reversible connections does not consistently reduce prediction in Tanni.
> Evidence against reversed dynamics alone is therefore insufficient to
> establish a required directional predictive component.

This is a model-control result on the already explored candidate population,
not independent replication, a causal test, or evidence that the animal
implements an HMM. Event detection used all cells, and calibration events
can occur later chronologically than test events. Predictive evaluation
does not establish a biological learning schedule or rule out structured
burst-phase effects. Four/five animals and extensive prior hypothesis
search limit confidence in a population-general mechanism.

HMM discovery of neural sequences and spatial relationships is established
in [Maboudi et al., 2018](https://elifesciences.org/articles/34467).
Hippocampal irreversibility is also not a new concept: the
[Shi and Lynn 2026 preprint](https://arxiv.org/abs/2601.05284) links neural
flows during movement to behavior and encoding resolution. Neither source
establishes this exact forecast contrast, but neither a failed primary
nor absence of an identical published control establishes a novel discovery.

The high-importance-paper goal remains unmet. Do not rescue the result by
selecting the positive horizon, excluding an animal, applying fuzzy event
weights after seeing scores, or calling all MUA candidates replay. Preserve
the failed primary and the bounded predictive-association result together.
