# Physical Versus Neural Metric: Identifiability Screen

Frozen before computing the new empirical-map results, 2026-09-09.
Exploratory follow-up, not independent confirmation. The preceding directional
forecast primary failed; its thresholds and conclusion remain unchanged.

## Scientific Question

Can the recorded populations distinguish propagation organized by physical
distance from propagation organized by population-code similarity? This is
related to the neural-versus-physical speed hypothesis, but does not test
constant velocity, directed trajectories, or biological replay speed.

Before scoring real events, quantify a best-case necessary operating condition:
with the represented origin known exactly, can a metric constructed from only
training neurons predict future held-neuron identities generated using the
full recorded population's neural geometry? Failure would make a subsequent
physical-model preference ambiguous because the neural comparator is inadequate.

## Frozen Inputs

All 33 RUN-map caches from
`/mnt/seagate10tb/florianpfaff/conditional-2d-mua-pf-tanni-all33-20260908`.
Four PF rats/eight recordings and five Tanni rats/25 recordings. Reuse all
five saved 70/30 cell partitions. Do not inspect or rescore candidate spikes,
change maps, exclude low-quality animals, or select favorable recordings.
Hash-check every cache against the source manifest. Rate maps are already
estimated and previously inspected; this is not an independent data sample.

## Matched Transition Families

Physical cost: squared Euclidean distance between valid spatial-bin centers.
Neural cost: squared Hellinger distance between conditional cell-identity
distributions p(cell | position), computed from RUN rates. Normalize each
cost by its median positive off-diagonal value to set numerical scale only.

For each cost D, form symmetric off-diagonal affinities exp(-beta D). Use
symmetric matrix scaling to make the off-diagonal transition doubly stochastic.
Fit beta to mean off-diagonal row entropy 0.5 log(n_states - 1), identical
for both families. Set the diagonal to exp(-0.02/0.06) in both. Thus both
families share uniform equilibrium over valid grid states, every state dwell
probability and mean conditional transition entropy. They need not share
individual edge weights, physical jump distances or each row's entropy.
This is a controlled reversible random-walk comparison, not literal
constant-speed motion. Entropy matching is a modeling convention, not a
biological estimate. No replay score selects its parameters.

Construct the neural generator from all recorded RUN cells and its predictive
approximation from training RUN cells only, separately for each partition.
The physical generator does not depend on the cell split. All likelihood
maps for held cells come from RUN, which is allowed; no held replay activity
or target observations are used to infer an origin or geometry.

## Exact Oracle-Origin Prediction

The origin is a known grid state sampled uniformly. Evolve 1, 2 or 4 steps
(20/40/80 ms), with 40 ms fixed as primary. Draw one future held-cell identity
from the held population's normalized RUN rates at the generated destination.
The spike count is one by construction: this is a conditional identity model,
not a Poisson process or a prediction of population amplitude.

Integrate all origins, destinations and possible future identities exactly.
No Monte Carlo event sampling or real replay scoring is necessary. Report:

- Physical generator: physical minus training-neural expected log score.
- Full-neural generator: training-neural minus physical expected log score.
- Full-neural generator: oracle-full-neural minus physical (KL benchmark).
- Full-neural generator: oracle minus training-neural (approximation deficit).

The first and oracle contrasts are nonnegative by the proper-score identity;
their positivity is NOT a discovery. The second can be negative if partial
population geometry is misleading, even with a known origin. The deficit
must be nonnegative. Also report physical jumps, achieved entropy, stationary
and dwell residuals, cell counts and numerical convergence.

Reduce split medians within recording and equal recording means within animal.
Do not use bootstrap significance or claim independent biological observations
from exact synthetic expectations. The necessary screen passes only if the
training-neural model beats physical under its own full-neural generator in
every animal in both datasets at 40 ms, with no missing fits or scores.
Passing is necessary, not sufficient: unknown origins, finite spike counts,
unrecorded neurons and encoding mismatch remain for subsequent recovery.
No new real-event scoring is authorized by this screen alone.

## Validation And Limits

Before empirical maps: symmetry/stochasticity/equilibrium/diagonal/entropy
tests, identical metrics, relabeling invariance, invalid maps and singular
geometry fail explicitly, direct dense expected-score reference, and proper
KL inequalities. Afterward independently reconstruct costs, kernel constraints,
all forecast expectations and all reductions; record exact scope and failures.
Any numerical failure stays visible; no fallback replaces a neural metric
with a physical one. A source cache failing a hash aborts the run.

Fitting both generator and decoder from the same estimated maps is an
optimistic operating check. The full recorded population is not the brain's
full population. A pass cannot validate biological mechanism or resolve
place-field uncertainty. A failure rejects this comparator under these
conditions, not neural-space propagation itself.

## Novelty Boundary

Associative versus physical geometry in replay has substantial precedent,
including [Coordinated hippocampal-entorhinal replay as structural inference
(2019)](https://proceedings.neurips.cc/paper_files/paper/2019/file/aa68c75c4a77c87f97fb686b2f068676-Paper.pdf)
and [Diekmann and Cheng (2023)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10076035/).
This screen checks whether an empirical discriminator is possible; it does
not invent neural geometry, establish novelty by absent search results, or
satisfy the high-importance-paper objective.
