# Reversible-Connection Forecast Control: Frozen Protocol

2026-09-09, before new real-data scoring. Subsequent exploratory mechanism
control, not independent confirmation or a change to earlier failed gates.

## Question And Primary Model

Does the learned neural HMM forecast future held-out cell identities better
when its learned transition direction is retained, compared with the SAME
undirected connections and equilibrium occupancy without directional flow?
This separates preferred direction from reversible associations. It is not
a new definition of replay or evidence for an animal planning its future.

Primary model: frozen K50 learned neural HMM. It is the only model here whose
directional transition preferences were learned from separate events. Include
both real/permuted spatial IMM and diffusion as diagnostics, not alternative
primary models selected after inspecting the results. Spatial diffusion should
be reversible up to numerical tolerance; the IMM's joint mode-position process
may have built-in asymmetry. That asymmetry is not learned biological direction.

Use the complete independently audited occupancy-control run, SHA256
`da514a563fc1ac1b9e3704cdf6b8a95b14b88049e14eebaf84901739fd07d980`.
All 9,225 immobile MUA events, 33 recordings/nine animals, five event folds,
five neural partitions, maps, gains, learned models and original forecasts
remain unchanged. No event classification or cell/animal exclusion is added.
Read the model equilibrium distributions already independently verified in
that run. Do not refit anything on new predictive scores.

## Nulls

For row-stochastic original A with positive stationary pi, define

    R_ij = pi_j * A_ji / pi_i
    B = (A + R) / 2

R is the stationary time-reversed KERNEL, not a time-bin shuffle and not
backward filtering. Both R and B are used to forecast forwards from an origin.
B preserves pi, every A_ii, and each undirected equilibrium edge flux:

    pi_i B_ij + pi_j B_ji = pi_i A_ij + pi_j A_ji

B obeys detailed balance. Thus it removes net directional currents while
preserving the strength of each unordered connection. R reverses currents
and preserves their magnitudes. These properties follow exactly from the
formula, aside from the specified stationary-distribution numerical tolerance.

For spatial operators, use their full joint mode/position pi. Apply R using
the adjoint operator rather than materializing dense fragmented transitions:
R(q) = pi * A_transpose(q/pi), in the row-distribution action convention.
Unlike the previous occupancy null, B need not preserve conditional mode
probabilities separately for every position. This limits the spatial-IMM
interpretation; it does not affect the primary learned-neural comparison.

All three propagation rules start from the identical ORIGINAL forward-filtered
origin posterior. Match the original likelihood/permutation order, array-copy
semantics and saved forecast hashes exactly. No held-out or intervening spikes
update any forecasts. No alternative model is re-inferred with target data.

## Endpoints And Decisions

Full 20-ms bins, fixed primary 40-ms center lag (20-ms unobserved gap).
20/80-ms horizons are sensitivities with potentially different event coverage.
Proper held-out multinomial cell-identity scores conditional on target spike
count, not powered likelihoods or prediction of total firing rate.

Three contrasts for every original model:

- original minus reversible B;
- original minus time-reversed R;
- reversible B minus the previous occupancy/dwell-matched null.

The last contrast asks how much prediction remains from unordered connections.
Individual split-level differences decompose exactly, but medians of contrasts
need not sum after aggregation. Keep old source scores alongside the new rows.

Primary aggregation: event medians over neural splits, equal session means,
then equal animal means. Per-held-out-target-spike is primary; raw event nats
secondary. Exact conditional animal-bootstrap intervals as before.
A directional predictive lead requires positive lower CIs and positive
estimates in every animal for BOTH primary neural-HMM contrasts in BOTH
datasets at 40 ms. No spatial model or different horizon can substitute.
Retain the prior failed broad forecasting criterion as failed. Do not infer
biological absence from a failure, or causality/novelty from a pass.

## Verification

Before real scoring: dense-reference R/B equality, stochasticity, stationary
occupancy, identical diagonal, fixed unordered flux and B detailed balance;
directed synthetic sequence recovered; reversible generator produces zero
contrast; future/held-out mutation invariance; original hashes reproduced;
complete model/split/horizon coverage and non-vacuous failed/missing cases.

After scoring: verify all source/output hashes, original rows, all parameter
constraints, all aggregates and decisions. Reconstruct first event per fold,
all five splits and all eligible horizons/models with separately implemented
filtering, likelihoods and reverse propagation. Record verification scope:
native raw files and original model fitting are not repeated.

## Literature And Novelty Boundary

Broken detailed balance is established in neural dynamics, for example
[Lynn et al., 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8617485/).
The January 2026 preprint
[Shi and Lynn, Irreversible behavior drives neural flows in the hippocampus](https://arxiv.org/abs/2601.05284)
examines movement-related flows in calcium recordings on a virtual track and
relates them to velocity, velocity variation and encoding resolution. That
is not this immobile-event predictive test, but it precludes claiming that
hippocampal time-asymmetry or its connection to resolution is new. A positive
result here would remain a mechanistic lead needing independent confirmation.

This estimates predictive value of fitted directional transitions, not
physical entropy production. The original filter, event selection using all
cells, retrospective calibration folds and extensive preceding exploration
remain limitations. Symmetric associations can generate ordered individual
paths in either direction; failure of a net-direction control does not mean
that individual events contain no sequences.
