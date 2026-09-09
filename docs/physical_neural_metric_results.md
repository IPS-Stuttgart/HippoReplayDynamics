# Physical Versus Neural Geometry: Verified Operating Screen

2026-09-09. Simulation-only identifiability screen using existing RUN maps.
No real replay candidate was rescored. The high-importance-paper goal is
not achieved, and previous failed biological/forecasting criteria are unchanged.

## Why This Is The Next Experiment

The preceding directional study separated useful reversible associations
from a non-replicated directional increment. Instead of choosing another
favorable directional threshold, this screen returns to the substantive
physical-versus-neural propagation question. It asks whether that comparison
is measurable under ideal conditions before interpreting any real preference.

Physical distance and similarity between neural population firing patterns
are related, not identical. A predictor based on only a subset of neurons
could estimate neural geometry poorly and then falsely make physical
geometry look preferable. The known-generator screen checks this failure
mode while matching both models' equilibrium occupancy, state dwell, and
mean off-diagonal transition entropy. It is not a new continuity criterion.

The neuronal metric is Hellinger distance between normalized cell-identity
probabilities from RUN rates. This is dimensionless, not the earlier
unconditioned Poisson-rate distance or a measurement of neural-sheet speed.
The generators are reversible random walks, not constant-velocity paths.

## Frozen Cohort And Computation

- All 33 cached RUN maps: PF eight recordings/four rats; Tanni 25/five.
- PF 63-213 encoding cells and 539-584 spatial states per recording.
- Tanni 48-165 encoding cells and 160-1,400 spatial states per recording.
- All five saved 70/30 neural partitions; no event or animal exclusions.
- Full population defines the neural generator. Training cells alone define
  the estimated neural metric; held-cell RUN maps define target identities.
- Known origin sampled uniformly over valid bins. Exactly integrate every
  destination and possible future held-cell identity at 20/40/80 ms.
- Fixed primary 40 ms. One held-cell spike by construction, not a Poisson
  count process or an estimate of actual event recovery probability.
- 231 matched transition kernels, 495 split/horizon rows, 1,980 expected
  score contrasts. Runtime 13.69 seconds on gpuserver6000, eight workers.
- Split medians within recording, equal recording means within animal, then
  equal animal means. No biological p-values or confidence intervals from
  these exact synthetic expectations.

## Primary Results

Expected nats per future held-cell identity under the stated known generator:

| Contrast | PF | Tanni |
| --- | ---: | ---: |
| Neural generator: training-neural minus physical | +0.000883 | +0.000484 |
| Neural generator: oracle-neural minus physical | +0.001090 | +0.000637 |
| Neural generator: oracle minus training-neural | +0.000197 | +0.000150 |
| Physical generator: physical minus training-neural | +0.001299 | +0.000788 |

The first contrast is positive in all nine animals and all 33 recording
medians at 40 ms. The frozen known-origin readiness screen passes. Animal
values are:

| Dataset | Animal | Training-neural minus physical |
| --- | --- | ---: |
| PF | Rat1 | +0.000931 |
| PF | Rat2 | +0.000903 |
| PF | Rat3 | +0.000839 |
| PF | Rat4 | +0.000858 |
| Tanni | R2470 | +0.000428 |
| Tanni | R2474 | +0.000477 |
| Tanni | R2478 | +0.000345 |
| Tanni | R2481 | +0.000709 |
| Tanni | R2482 | +0.000460 |

The sign is also positive in every animal at 20 and 80 ms. These are
sensitivity endpoints, not new biological replications. The oracle and
physical-generator contrasts are KL divergences and nonnegative by
construction: their positivity must not be counted as discoveries.
Differences of separately median-aggregated quantities need not add exactly.

The measurable distinction is small in this single-identity experiment.
Do not translate its reciprocal into required event or neuron counts: actual
spikes share latent states, finite-bin joint likelihoods are not independent
single-spike mixtures, and future origins are correlated. Proper finite-spike
recovery and operating power remain to be measured.

## Matched Does Not Mean Identical

Uniform equilibrium and every diagonal are shared. Dwell is exp(-1/3), and
mean conditional off-diagonal entropy is 0.5 log(n_states - 1). Individual
row entropies and physical jumps are not fixed. Median per-recording RMS
physical step size, including stays, is 7.95 cm for the PF physical kernel
versus 8.49 cm for its full-neural kernel; Tanni values are 7.18 and 7.64 cm.
These describe simulated kernels, not observed speed or constant motion.
The underflow guard clips log affinities at -700; it is frozen in code.

The protocol calls this a necessary screen as a chosen readiness criterion,
not a mathematical theorem ruling out all other conditioning populations.
Knowing the origin exactly is deliberately optimistic. Both generator and
predictor reuse fitted RUN maps. The full recorded population is not all
neurons in the animal, and map estimation uncertainty is not tested here.

## Verification And Reproducibility

- Initial science/protocol commit: `03605e012e87921ffb5032a6ffe89bf7911e2475`.
- Clean scoring/auditing/reporting commit:
  `5bbfe48b713cd94988759170944148d5f1f36243`.
- Run: `/mnt/seagate10tb/florianpfaff/physical-neural-metric-screen-all33-20260909`.
- Run manifest SHA256:
  `71882248599186e0e790b408d1524a98d713d4c3220225004e00aafc05fde937`.
- Parent cache manifest SHA256:
  `d16542d0bf935929a05a335f84cd2060167c76dd8e255378beb202fb84e97386`.
- Independent audit in the sibling `-audit` directory reconstructs all costs,
  all 231 scaled kernels, all 1,980 expected scores, cell counts, physical
  step diagnostics, split/recording/animal reductions, and the decision.
- Maximum independent expected-score discrepancy: 2.7102e-14 nats.
- Maximum row-sum residual: 2.8293e-12; symmetry residual 4.1634e-17;
  diagonal residual zero; entropy discrepancy <= 9.9986e-8. At most 152
  balancing iterations were needed per evaluated kernel.
- All source/output hashes verified. Raw timestamp ingestion and RUN map
  estimation were not repeated; candidate spike arrays were not read.
- 102 relevant tests and Ruff passed. The report's four artifact hashes
  were checked, and its 2160 x 900 figure was inspected visually and for
  nonblank pixels (RGB standard deviation 37.79).
- Report: sibling `-report/physical_neural_metric_report.md`.

## Research Decision

The screen establishes that the two controlled geometries are distinguishable
in expected held-neuron predictions with known positions, including when
the neural metric is estimated without the held neurons. It does not establish
which geometry real activity follows, or even adequate individual-event
classification at actual spike counts.

The next substantive step is a frozen finite-spike recovery experiment with
hidden positions, using both known generators, all 33 maps and all five
partitions. It must assess observation/map mismatch as well as the optimistic
same-map case before any real-event preference is interpreted. Include
static/unstructured controls, retain all failed conditions, and do not select
favorable animals or a better horizon after results. No such finite-spike or
real-event result is claimed in this note.

This remains a possible measurement route to the physical-versus-neural
propagation question. Associative geometry in replay already has theoretical
precedent ([2019 structural-inference model](https://proceedings.neurips.cc/paper_files/paper/2019/file/aa68c75c4a77c87f97fb686b2f068676-Paper.pdf);
[2023 experience/structure model](https://pmc.ncbi.nlm.nih.gov/articles/PMC10076035/)).
The screen's numerical success alone neither establishes novelty nor supplies
a high-importance biological paper.
