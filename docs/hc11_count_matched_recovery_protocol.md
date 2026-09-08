# Count-matched hc-11 conditional-prediction recovery

Frozen before simulation scoring, 2026-09-08. This is a capability diagnostic
for the negative external conditional-prediction result, not a retuned real
analysis, biological absence claim, or new event selection.

## Inputs and sample units

Use all 320 PRE/POST candidates and eight session map/count caches from
`hc11-conditional-cross-cell-prediction-320x5-20260908` (producer cb048fe1).
Its raw-count/predictive audit passed. Parent manifest SHA256:
`bc846d25e27b9400d2870b2cad534c781d1fdee12f27496e5a8d96db153d3d07`.
Four animals remain the biological sample, not the number of simulations.

Generate 50 independent conditional datasets for each of four generators:
independent positions, one static location, fixed diffusion, first-order IMM.
Every dataset uses all 320 templates. Primary summaries use POST; PRE is
retained as the predeclared sensitivity. No event is dropped for an empty
held-out subset or an unhelpful model rank.

## Generator

Uniform initial position and a global direction drawn with probability 1/2
for each native direction map. IID draws new uniform positions each bin;
static repeats the initial position; diffusion and IMM use the SAME frozen
linear/circular transition kernels as the hc-11 scorer: diffusion sigma
85 cm/sqrt(s), stationary IMM sigma 2 cm, four-sigma support, mode stickiness
0.95. IMM starts with uniform mode mass. These are stochastic position
processes, not constant-speed trajectories or biophysical network models.

For every bin preserve the observed total count N across all encoding units.
Draw cell identities from Multinomial(N, rates_i(x)/sum_i rates_i(x)), using
the selected direction's real RUN map. Draw ONE complete population event,
then apply the five existing cell splits to that same event. Do not generate
new observations separately for each split. Durations, partial final bins,
topology, unit IDs and total-count profiles match the real event templates.
Split-specific totals and active-cell counts may change and must be reported;
matching them all would generally be incompatible with a single population
draw. No true position is used to restore missing spikes during decoding.

This is a known-map, conditionally independent observation benchmark. It
does not recreate shared gain, cell-specific sleep changes, or RUN-to-sleep
encoding mismatch. Conditional training inference intentionally ignores the
spatial information in how the full total is allocated to each cell subset.
Passing therefore establishes capacity under these explicit assumptions,
not that real sleep activity must be equally decodable.

Seed: stable SHA256 of (20260908, hc11_count_recovery, session, replicate,
generator, phase, event ID). The latent path, direction and full count matrix
are saved. Replicates 0..49; compute in ten-replicate shards without changing
seeds. Identical spike draws feed all scored models and encoding variants.

## Prediction and summaries

Reuse frozen training-only inference and proper per-bin held-out multinomial
prediction at T=1. Score iid, static, diffusion and IMM, with direction-mixture
maps primary and pooled maps sensitivity. No wrong-map or prior sweep here.
True latent position/direction are used ONLY for an oracle score diagnostic,
never for model inference. The oracle is an expected-information reference,
not a guarantee of superiority for every finite observed event.

Paired differences are computed within each neural split, then median across
five splits for each event. Equal-animal means follow event aggregation.
Primary contrasts are IMM-iid and IMM-static; also report IMM-diffusion,
diffusion-iid, static-iid, and oracle-iid.

Precompute 2000 hierarchical bootstrap weight vectors per phase (animals,
sessions within animals, events within sessions), seed 20260908. Reuse weights
across simulated datasets to reduce Monte Carlo comparison noise. This is
a simulation diagnostic, not an additional independent-animal confidence
guarantee. The parent real-data CIs used 5000 draws and are left unchanged.

For each generator report effect distributions across 50 datasets, fractions
with all four animal effects positive, and fractions meeting the declared
positive pattern: both primary contrasts have positive equal-animal means,
positive lower bootstrap limits and positive effects in all four animals.
Report this for ALL generators, including IID and static nulls. Name it
`positive_pattern_fraction`, not unconditional biological power or false
discovery rate. Report binomial uncertainty; zero of 50 is not a zero error rate.

Raw predictive model ranks use event-median gains relative to iid; exact
ties are ambiguous. They are recovery diagnostics, not calibrated biological
model claims. Real effects may be shown against simulation distributions,
but pure-generator incompatibility cannot identify the true biological
mixture or diagnose a mechanism. Every family, phase and variant remains
visible; do not select a favorable simulation to reinterpret the real result.

## Verification and decisions

Check exact counts per bin, one draw reused across splits, deterministic
seeds, valid paths, disjoint cells, frozen posteriors, normalized held-out
probabilities, and non-vacuous completeness. Independently regenerate a
phase/session/generator-balanced set of simulations and their scores using
the separate dense log-domain verifier; reconstruct aggregate point estimates.
Report the actual audit scope, including any unverified bootstrap outputs.

If specified temporal generators reliably produce predictive gains at these
counts, scarcity alone is not sufficient to explain the real result under
the known-map observation assumptions. If they do not, the real negative
cannot distinguish weak temporal content from inadequate sensitivity.
Neither outcome validates biological absence, uniform replay speed, or an
IMM-specific brain mechanism. No real-data threshold/selection rescue follows
automatically from this benchmark.
