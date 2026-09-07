# Count-conditioned cross-cell prediction

Frozen 2026-09-08 before inspecting results. This is an exploratory mechanistic
discriminator motivated by previous PF findings, not independent confirmation.

## Question

Does spatial temporal inference predict which held-out neurons fire, beyond the
total-population rate term and beyond fixed spatial dynamics? A positive result
would motivate replication, not establish a behavioral function or novelty.

Use all 160 events in the frozen PF evidence table, without selecting on winners.
Fit RUN maps with the archived evidence settings. Use the same rectangular grid
and uniform prior as the historical scorer, including rate-floor bins; do not
silently change spatial support. Make five deterministic 70/30 cell splits per
session (seeds 20260804 through 20260808). Report event medians across splits,
then equal-animal summaries. The one-event-per-session run is technical only.

## Likelihood decomposition

For rates r_i(x), counts k_i and N=sum_i k_i, independent Poisson likelihood
factors exactly into Poisson(N; dt*sum_i r_i(x)) and
Multinomial(k; N, r_i(x)/sum_j r_j(x)). Compare training inference using:

- full_poisson: both factors;
- count_conditioned: multinomial identities only;
- total_rate_only: population-total Poisson factor only.

Conditioning removes the total-rate likelihood term and is invariant to a common
positive position-dependent scaling of all cell rates. It does NOT equalize
information across bins: bins with more spikes carry more identity information.
This is a shared-gain nuisance elimination, not proof all firing-rate confounds
or real replay correlations have been removed.

In every case infer latent posteriors from training cells only. Score held-out
cell identities conditional on their own observed total using normalized
multinomial probabilities; also report proper full Poisson predictive scores.
Held-out counts never update the latent posterior. Zero-held-out-spike bins have
conditional log score zero and are retained. Predictive scores are sums of
marginal log scores, not the joint path likelihood.

Primary inference temperature is 1.0. Historical 0.3 is sensitivity only;
held-out likelihood temperature is ALWAYS 1.0. Powered historical held-out
emissions are not normalized predictive probabilities and are not reproduced.

## Comparators and maps

Models: iid_position (fragmented), static_location (one position per event),
stationary (historical local small-step kernel), diffusion (historical fixed
diffusion), first_order_imm (historical three-mode model). Use no fitted replay
hyperparameters. The iid and static models are computed analytically.

Real maps versus the same seeded shared occupied-bin population-code permutation
as the preceding map audit. This wrong map preserves neural population snapshots
but destroys spatial adjacency. It is not a different-context map. Include
explicit iid/static invariance checks under this permutation.

Primary contrasts on conditional held-out scores at temperature 1:
IMM minus iid, IMM minus fixed diffusion, IMM minus static location,
count-conditioned IMM minus total-rate-only IMM, and real-minus-wrong IMM
predictive score. Report all directions, not just favorable contrasts. Compare
full-Poisson inference on the SAME conditional held-out target, not raw evidence
across different observation families. Selecting the best model from held-out
scores and calling it confirmatory is prohibited.

Hierarchical bootstrap: rats, sessions within rat, events within session;
5000 replicates, seed 20260908. Also report per-rat effects and exact rat-level
sign-flip p (only four rats, minimum one-sided p=0.0625). A positive bootstrap
interval does not override this small-animal limitation. No arbitrary logZ
threshold is a biological gate. Technical gates require nonempty exact coverage,
finite proper scores, disjoint cells, unchanged training posterior hashes, and
map-permutation invariance for analytic baselines.

## Next decisions

If identities add prediction but IMM does not exceed fixed diffusion, retain
spatial coordination and drop switching-specific interpretation. If total-rate
inference is equally effective, investigate population envelope rather than
claiming trajectory mechanisms. If conditional IMM exceeds comparators and is
map-specific across rats, test independent events/dataset before a paper claim.
Do not scale datasets or add biological hypotheses based on a favorable subset.

Relevant precedent: Denovellis et al. 2021 (eLife 64505) already uses switching
replay dynamics; Tirole et al. 2022 (eLife 79031) addresses rate coding and decoder
selection; this experiment is not a claim to invent either conditional decoding
or rate-confound checks.
