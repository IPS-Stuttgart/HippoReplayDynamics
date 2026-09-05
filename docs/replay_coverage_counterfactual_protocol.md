# Paired Spike-Loss and Cell-Identity Counterfactual

Status: frozen before new real-map simulation results. This extends, not
replaces, the complete scope in `replay_recording_coverage_study.md`.

## Why This Experiment

The first empirical-map benchmark compared native Poisson spike loss to
imposed totals from real candidates. That was informative, but not a direct
partition on identical parent observations. The primary control pools removed
cell counts into one aggregate channel, preserving the exact population total
of EACH parent in EVERY 1 ms bin without changing any retained-cell spike.
This is a deterministic transformation of the observations: no true position
or replacement labels are used. A secondary oracle-restoration stress test
reassigns removed spikes using the simulated position and is NOT a pure
information-loss intervention. Conditional and unconditional
likelihoods can both be correctly specified at the observation-family level.
Moving-within-window and spatial-discretization approximations still remain.

## Frozen Sources and Replication

Use the 971 source duration profiles already frozen in
`replay-coverage-recovery-all33-20260905`, covering 33 sessions and nine animals.
No new selection on any decoding result. Root seed 20260909, three independent
source-level randomizations of initial position/direction, nested cell subsets,
and observations. These are independent synthetic randomizations on SAME
empirical maps/profiles, not independent animals or held-out empirical maps.
All paired truth classes share starting randomization within source/replicate.

Primary replication uses firing-rate multiplier 3 for all three randomizations.
A separate paired dose series uses multipliers 1, 3, 10, 30 on replicate zero
ONLY. Its paths, cell subsets, and observations are coupled across every dose.
Do not combine the three-replicate primary point with one-replicate points in
the dose curve. Larger multipliers are information-limit stress tests, not
assertions about physiological replay rates. No dose is chosen from outcomes.

Truth conditions retain g=-0.5, 0, +0.5 horizontal speed fields, stationary,
and discontinuous snapshots. Add a whole-5-ms-bin permutation of the constant
continuous path AND its exact population spike vectors. Preserve within-bin
position/spiking structure and each cell's total spikes. No physical arclength
speed is assigned to the discontinuous or shuffled controls.

PF simulation extent is the encoding grid, not verified walls. Tanni uses
native arena bounds. The empirical bilinear maps remain surrogate truth.

## Exact Coupling and Likelihood

Generate full-population Poisson counts at highest multiplier (30), then
binomially thin through 10, 3, 1. This preserves nested counts in each cell
and fine-time bin while keeping correct Poisson marginals at every exposure.

Primary pooled control, for each half/quarter subset at every exposure:

1. Keep every retained-cell count exactly.
2. Append one channel equal to the sum of all removed-cell counts.
3. Decode using retained rate maps plus the sum of removed rate maps.

Pooling preserves all population totals, but removes individual identities of
the removed cells. The aggregate still carries its own spatial tuning. The
per-bin >=2-cells/>=3-spikes gate counts only retained sorted-cell spikes and
cells, never the aggregate channel. Thus pooled and native-subset observations
use exactly the same support masks. Record cell and observation-channel counts
separately. Unfiltered results avoid this support-selection difference versus
the full reference.

Secondary oracle-restoration control at maximum exposure:

1. Keep every retained-cell count exactly.
2. Count removed-cell spikes in the same fine-time bin.
3. Reassign those removed labels multinomially among retained cells, with
   probabilities proportional to retained-cell rates at the known position.
4. At lower exposures, hypergeometrically thin reassigned counts to the exact
   number of removed spikes at that exposure. Native kept spikes remain fixed.

The restored population has effective rates

lambda'_i(x) = sum_all lambda_j(x) * lambda_i(x) / sum_retained lambda_j(x).

Use these effective rates for restored Poisson decoding. Conditional decoding
uses the same retained-cell probabilities; the shared rate factor cancels.
Verify both the Poisson marginal mean/variance and conditional counts in a
stationary-on-grid synthetic unit test. This secondary counterfactual uses the
full map AND true position to assign replacement labels. It can add spatial
information and is not a data-processing degradation or a practical correction
for unknown unrecorded neurons.

Full/native is the shared reference. Decode seven population conditions:
full/native, half/native, quarter/native, half/pooled, quarter/pooled,
half/restored, quarter/restored. No duplicate full-population controls.

## Measurements and Inference

Reuse the same 20 ms independent uniform-prior decoder, MAP/posterior mean,
overlapping 5 ms continuity stride, non-overlapping 20 ms speed stride, fixed
geometric criterion, and optional >=2-cells/>=3-spikes support rule. Do not
bridge missing bins. Keep all failures and selection denominators.

For each truth-eligible constant-speed source/replicate and each estimator,
likelihood and support setting, calculate the exact algebraic partition:

native_subset - full = (pooled_subset - full) + (native_subset - pooled_subset).

The first component removes individual cell identities while preserving the
exact parent totals. The second discards the pooled channel, losing both its
spikes AND its aggregate spatial information; it is NOT a spike-count-only
effect. These are controlled measurement effects, not causal components of
brain function. Report oracle-restored contrasts separately without interpreting
them as pure information loss. All comparisons use the same truth and
are paired before event/session/replicate/animal averaging.

Report dose/replicate-specific continuity, null acceptance, errors and gradient
responses. Average events within session/replicate, randomizations within
session, sessions within animal, animals equally. Bootstrap animals, conditional
on the maps/simulation design. Keep per-replicate summaries to expose simulation
variability. Gradient availability retains >=5 events and spatial variance
>=0.01; no relaxing those thresholds because selected events are sparse.

## Gates and Scope

Expected full run: 971 source profiles x 3 randomizations x 6 truth classes =
17,478 trial records (including shuffled derivatives). Score six exposure/
replicate combinations per source (four doses at replicate zero, primary dose
at replicates one and two), producing 1,957,536 metric rows. Generate 489,384
observation arrays across all doses, including unscored doses of replicates
one and two, to retain the same nested generation scheme. These rows are not
independent biological observations. Save per-batch artifacts, path/count
hashes, exact populations, dependency snapshot, and clean code provenance.

Technical gates verify complete source/truth/replicate/condition coverage,
preserved population totals, retained spikes, nested exposure counts, whole-bin
shuffle invariants, and unchanged inputs/code. Scientific outcomes are never
technical pass criteria. A tiny smoke cap is deterministic and explicitly
excluded from the production cohort.

This tests synthetic replication and a more precise information-loss partition.
It does not finish field-width/density/arena factorial isolation, map-estimation
and observation-mismatch validation, independent correction evaluation,
event-definition sensitivity, verified wall-gradient analysis, or biologically
meaningful equivalence testing. Those requirements remain active.
