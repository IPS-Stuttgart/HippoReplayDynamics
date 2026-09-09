# Literal Replay Clock Recovery: Frozen First Screen

Simulation only. No real replay score, selection threshold or biological claim
is produced. This tests a different hypothesis from entropy-matched random-walk
kernels: actual constant physical arc-length speed versus constant conditional
population-code arc-length speed along the SAME continuous geometric path.

## Design

Use all 33 frozen RUN encoders (PF: eight recordings/four rats; Tanni: 25/five).
Source: `conditional-2d-mua-pf-tanni-all33-20260908`, all source hashes checked.
For each recording draw 32 paths, alternating straight and sinusoidally curved.
Endpoints are sampled uniformly from valid grid centers, 40-120 cm apart. Curve
amplitude is 20% of endpoint separation, with random sign. Reject unsupported
geometry only: no likelihood, event acceptance or decoder quality selection.
Triangular rate interpolation must stay in triangles with diameter <= the
8 cm grid diagonal; no bridging large missing regions. Record sampling attempts.
Geometry is represented by 801 nodes. Degenerate code geometry fails explicitly.

Draw a native full-20-ms-bin count profile per path, using only profiles with
at least three bins. Retain its duration and total spikes in EACH bin. Generate
fresh conditional-multinomial cell identities, not copies of recorded identities.
Five independent observation realizations per path/generator/condition. New
SHA256 RNG namespace `literal_clock_v1|20260909`.

For location x, define p_i(x)=lambda_i(x)/sum_j lambda_j(x). Neural code arc
length is accumulated Hellinger distance between neighboring p vectors. It is
not physical distance between neurons and does not establish how a neural sheet
is wired. Physical clock is normalized physical arc length; neural clock is
normalized code arc length. Both traverse the full same path in the same time.

Integrate rates over each time bin using 128-point Gauss-Legendre quadrature;
normalize the integrated rates to generate conditional identities. Do not treat
fast motion as a stationary location at the bin midpoint. Exact and gain-drift
conditions: gains are exp(.35*Z-.35^2/2), fixed per cell/recording. Both clocks
remain the original RUN-defined clocks; drift changes emissions, not the path.

## What The Scorer Knows

Primary test is an OPTIMISTIC KNOWN-PATH ORACLE. It compares the two candidate
clock likelihoods with the true geometry, start, end, duration and rate map
provided. Drift is not provided to the scorer. Chance is 50% with equal classes;
numerical ties receive half credit. This is NOT a deployable real-event decoder,
not a replay detector and not held-out prediction. A positive result permits a
future blind recovery study only. A negative is not a universal information bound.

Secondary: independent uniform-prior conditional-multinomial decoding on 8 cm
and deterministically coarsened 16 cm grids, without a temporal prior. Record
posterior mean/MAP errors and speed, posterior RMS uncertainty, and retained
hard-continuity step fraction (<20 cm). These are diagnostic summaries, not
Foster's full trajectory-event rule and not a newly calibrated fuzzy classifier.
Coarse bins average supported fine-bin rates/centers; partial edge groups exist.

Aggregate observation repeats within path, then paths within recording, then
recordings within animal, then animals within dataset. Do not present thousands
of synthetic repeats as biological sample size. Primary practical screen:
mean exact oracle accuracy >=.80 in BOTH datasets and each animal >.50.
Same target for gain-drift robustness, separately. Thresholds are operating
targets, not a novelty criterion or a biological confidence threshold.

## Verification And Interpretation

Save geometry, native totals, clock coordinates, averaged rates, all simulated
count arrays and all scores. Independently reconstruct arc lengths, integrate
with 256 nodes, verify count totals and regenerate sampled identities, recompute
likelihoods/decoding and summary reductions. Report quadrature disagreement;
max probability error >1e-4 fails numerical readiness rather than silently
removing events. Inspect figures and attach code/input/output hashes.

True clocks may be similar along some paths; retain those paths. Constant code
speed implies physical speed inversely proportional to local code distance per
cm by construction; that is not a discovered biological effect. RUN maps are
treated as known; no map-estimation-error arm yet. Strong cell correlations,
rate changes beyond this gain model, alternative anatomical metrics, unknown
geometric paths, stationary/fragmented nulls and fuzzy event selection all remain
outside this first oracle screen. They are required before real inference.

Broad constant-speed replay has precedent in
[Davidson et al. 2009](https://pmc.ncbi.nlm.nih.gov/articles/PMC4364032/), and
variable-speed models in
[Denovellis et al. 2021](https://elifesciences.org/articles/64505).
Any paper contribution must discriminate mechanisms beyond those precedents,
not merely show that a constant-speed simulation produces constant speed.
