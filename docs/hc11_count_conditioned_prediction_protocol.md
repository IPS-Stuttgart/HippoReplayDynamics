# hc-11 conditional cross-cell prediction

Frozen before the new real-data scores on 2026-09-08. This is an external
replication of a predictive target, not a new biological mechanism or a rescue
of the earlier negative hc-11 strict replay ladder.

## Question and cohort

Does a temporal model inferred from training neurons predict the held-out
cell-identity pattern beyond independent locations, static location, and
inference from population totals alone?

Use the existing `hc11-pre-post-paper-events-primary8-p20-pre-rate` selection:
320 events (160 PRE, 160 POST), eight sessions, four animals. Candidates are
paper-style population synchrony events overlapping published or previously
validated LFP ripple detections. PRE/POST pairs were strength-matched before
model evidence. This is a previously inspected cohort, not new confirmation
data. No new winner, duration, decoder, animal, or event-strength selection.
Native and generated ripple sources remain individually traceable.

Primary: POST, direction-conditioned map mixture, inference temperature 1.
Pooled maps and inference temperature 0.3 are declared sensitivities; PRE and
matched POST-minus-PRE are exploratory. Do not choose a favorable setting.

## Frozen method

- Reuse the native hc-11 MAZE encoding loader, mechanically extracted from
  `6a491825:scripts/score_hc11_webshare_native_ripple_evidence.py`. Its existing
  CA1 place-like unit selection uses RUN data only. No slow/fast subgrouping.
- Keep native 4 cm spatial bins, 20 ms event bins with a final partial bin,
  no padding, RUN speed >=5 cm/s, smoothing 1.5 bins, >=20 RUN spikes,
  >=0.1 bits/spike spatial information, >=1 Hz peak rate, >=5 encoding units.
- Linear/circular topology is taken from native maze metadata. Uniform
  spatial initial prior; no occupancy prior. Diffusion sigma 85 cm/sqrt(s),
  near-stationary IMM sigma 2 cm, four-sigma support, mode stickiness 0.95.
  These retain the native hc-11 settings, not PF's grid or temporal resolution.
- Five deterministic 70/30 splits, seeds 20260804..20260808, shared across
  conditions within each session. All cell encoding maps learned during RUN.
- Training observations: full Poisson, normalized multinomial cell identities
  conditional on each bin's spike total, or Poisson population totals alone.
  Reuse the exact likelihood factorization from the proper PF prediction audit.
- Models: iid positions, one static position, fixed diffusion, first-order
  stationary/diffusion/fragmented IMM. The IMM is implemented as a block
  transition using the existing forward-backward engine and tested against the
  PF implementation on a linear grid and explicit path enumeration.
- Direction is a global latent variable with equal prior weights. Compute its
  posterior from training observations only. At every time bin sum prediction
  over training-weighted direction AND position before taking the logarithm.
  The iid baseline in this variant has independent positions conditional on a
  shared direction; pooled maps avoid that additional shared latent variable.
- Freeze all posteriors before computing held-out likelihoods. The primary
  test likelihood is normalized multinomial at temperature 1 in every
  condition. Held-out spike totals specify the conditional prediction target;
  they do not update the position/direction posterior. Also record proper
  untempered full-Poisson scores as a descriptive sensitivity.
- Scores sum per-bin marginal log predictions. They are NOT joint sequence
  likelihoods, Bayes factors, or predictions of future time bins.
- One shared population-code position permutation per session, seed derived
  from SHA256(20260908|session); applied identically to training and held-out
  maps and both directions. Geometry stays fixed. This is not a cross-day map.

## Reporting and decision

Per-split paired differences -> event medians across splits -> equal-animal
means. Report per-spike normalization, both phases, each animal/session, and
matched POST-minus-PRE. Pointwise hierarchical bootstrap 5000 replicates,
seed 20260908; exact animal sign-flip alongside, with four animals' minimum
one-sided p=0.0625. No universal significance claim or multiplicity-free
confirmation based on bootstrap intervals alone.

Primary contrasts: IMM-iid, IMM-static, IMM-diffusion, real-permuted IMM,
identity-only minus total-rate-only inference. Identity-only minus full
inference, diffusion-iid, static-iid are explanatory controls. Do not pool raw
nats across PF and hc-11: event duration, cells, dt and priors differ.

A robust external predictive pattern would have positive primary POST
IMM-iid and IMM-static effects in all four animals, positive bootstrap lower
bounds, and consistent directions in pooled maps and T=0.3. This is NOT an
IMM-specific mechanism: examine IMM-diffusion separately. Any mixed/negative
result stays mixed/negative; no subset or prior sweep to obtain a replication.
The earlier strict ladder remains a distinct, negative result.

## Verification

Fail on missing events, condition/split rows, nonfinite probabilities,
overlapping neural splits, test-temperature changes, or posterior updates.
Preserve zero-held-out-spike rows (score zero, per-spike undefined). Cache
maps/counts/edges/splits, hash actual raw MAT inputs and outputs, and record
the commit. Before interpretation independently verify raw counts and a
deterministic, phase/session-balanced score subset, plus all paired summaries.
The audit must state its scope; a technical gate alone is not a biological test.
