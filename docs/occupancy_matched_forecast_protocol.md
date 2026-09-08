# Occupancy-Matched Forecast Null: Frozen Protocol

2026-09-09, before new real-data forecast scores. Exploratory control on
previously inspected data; not an independent replication or a threshold change.

## Question

Does the earlier forecast advantage survive removal of preferred destinations
when the null also preserves the ORIGINAL MODEL's stationary state frequencies?
The previous dwell null did not preserve them. Its compound two-dataset gate
remains failed; this experiment cannot retrospectively change that verdict.

## Fixed Data And Endpoints

Use exactly the completed lagged forecast run, manifest SHA256
`342d5670f77706ad784af7b1f98368f8eaa66e88247cfe78bb188d2daa79195a`,
with its matching passing independent audit. All 9,225 events, 33 recordings,
five event folds and five neural partitions remain fixed. Source parameters
are reused, not refitted to observations. Hash the source run, original input
files and all original forecast outputs used for pairing.

Keep full 20-ms bins, original too-short and zero-count policies, original
forward-only origin posterior, and horizons 20/40/80 ms center-to-center.
40 ms is primary. No held-out or intervening spikes update any forecast.
Score held-out multinomial cell identities conditional on total target spikes,
not total firing rate. Reconstruct the original dynamic score and forecast
hash for every row before accepting its pairing with the old result.

All five model rows: real/permuted spatial IMM, real/permuted diffusion,
and K50 learned neural HMM. The new null is the only new scored condition.
Primary contrasts are original dynamic minus occupancy/dwell-matched null
for real IMM, real diffusion and neural HMM. Also report the old dwell-null
score minus the new null, and the real-versus-permuted difference of the new
dynamic-minus-null contrast. No favorable model, horizon or animal subset
will be selected after scoring.

Report event medians across splits; equal event, session, then animal weights
as previously frozen. Per-held-out-spike is primary, raw nats/event secondary.
Exact conditional animal-bootstrap intervals. A destination-structure lead
requires positive lower CI and all animals positive in BOTH datasets for the
same model at 40 ms. The broader forecasting claim also retains the original
frozen/no-history/global controls. A positive new contrast alone cannot rescue
a failure of those controls. Neither criterion establishes novelty or causality.

## Maximum-Entropy Null

For the neural model, compute its stationary distribution pi from its frozen
transition matrix A. Preserve B_ii=A_ii, B*1=1 and pi*B=pi. Maximize conditional
entropy -sum_i pi_i sum_j B_ij log(B_ij) over the remaining entries. In flux
coordinates F_ij=pi_i B_ij, this is maximum entropy with fixed off-diagonal
row/column masses pi_i(1-A_ii) and structural diagonal zeros. The solution has
F_ij=u_i v_j off diagonal, fitted by iterative proportional scaling. This is
a deterministic constraint construction, not a fit to target spike likelihoods.

For the spatial models, index each state by mode m and position x. Preserve:

- the full joint stationary distribution pi[m,x];
- each original conditional mode-switch probability M[m,n], for every source x;
- the probability of unchanged position for EVERY source/destination mode:
  B[(m,x),(n,x)] = M[m,n] * K_n[x,x].

This includes full-state dwell, within-mode spatial dwell, and position dwell
on mode switches. It retains every corresponding constraint of the previous
spatial dwell null, while adding stationary joint-state frequencies.

Fixed flux: F[(m,x),(n,x)] = pi[m,x] M[m,n] K_n[x,x]. For y!=x, the
maximum-entropy remaining flux has form F[(m,x),(n,y)] = u[(m,x),n] v[n,y].
Row-group totals are pi[m,x] M[m,n] (1-K_n[x,x]); column totals are pi[n,y]
minus all fixed incoming flux. Scale u and v to satisfy both. This factor form
allows linear-time null propagation without materializing the dense matrix.

Existence is guaranteed by the original transition operator satisfying the
same constraints, aside from numerical approximation of pi. Stationary-only
destination groups can have exactly zero remaining flow. Handle these zeros
explicitly; reject residual inconsistencies above numerical tolerance. Do not
silently alter model probabilities to force feasibility or convergence.

Validate row sums, fixed dwell, conditional mode totals, stationary-frequency
preservation, nonnegative entries, and scaling residuals. Preserve pi to
absolute 1e-11 and row/mode probabilities to 1e-10 or fail the session. Record
iterations and all residuals. Store pi, diagonal kernels, mode matrix and null
factors for independent reconstruction. pi is model equilibrium, not empirical
burst occupancy, which can be nonstationary.

## Verification And Interpretation

Before real scoring: compare the factorized null against a separately scaled
dense flux matrix on tiny states; check all constraints; recover a known
directed sequence; obtain zero contrast when the generating transition is the
maximum-entropy null; test suffix and held-out mutation invariance. Check
temporal eligibility, complete factors and no-vacuous summaries.

After scoring: audit hashes, all saved null constraints and deterministic
first-event-per-fold forecasts across all splits/horizons/models. Reconstruct
paired event/session/animal contrasts and intervals independently. All source
dynamic scores must match the prior run, not merely a sampled subset.

A positive effect means preferred destinations improve prediction beyond
these matched properties in these models. It does not prove a planned route,
a unique neural mechanism, or a spatially specific IMM advantage. Spatial
diffusion kernels remain symmetric. Learned neural states may encode space.
The same original filter supplies both origin distributions; this does not
compare two optimally refitted inference models. Broad exploratory selection
and limited animal counts remain limitations even if numerical gates pass.

## Numerical Reconstruction Amendment

The first run, `occupancy-matched-forecasts-all9225-20260909`, failed its
original-forecast hash check and is retained unchanged. No new null contrasts
were interpreted. A targeted reconstruction of Rat1/Open2 event 1 found that
permuting rate-map columns before likelihood evaluation changed likelihoods
by 1.42e-14 and forecast probabilities by 6.66e-16. Computing likelihoods on
the original maps and then permuting their columns, exactly as the source
scorer did, reproduced the source forecast hash exactly. The real-map case
also retains the original advanced-indexing operation. The new diffusion
regression additionally required the original explicit copies of filtered
and collapsed probabilities to preserve array layout and reduction order.
This amendment fixes
only numerical operation order, adds multi-cell permutation regression tests,
and retains exact hash matching and every scientific parameter and decision
rule. The full rerun uses a new v2 directory and a separately frozen commit.
