# Exact sparse momentum state-space decoder

`sorted-spike-state-space-momentum-exact-sparse` is a paper-facing second-order
state-space decoder for momentum replay.  It differs from
`sorted-spike-state-space-momentum` in one important way: the state support is
not selected by emission top-k candidates.  Instead, the decoder runs an exact
forward/backward recursion over sparse pair states `(x[t-1], x[t])` induced by a
finite-radius Gaussian transition model.

This makes its evidence comparable to the exact first-order state-space rows.
The model reports

```text
diagnostic_state_space_sparse_momentum_evidence_support=exact_full_grid
diagnostic_state_space_sparse_momentum_state_support=finite_radius_pair_grid
diagnostic_state_space_momentum_candidate_selection=none_exact_sparse
```

Use it to separate three cases that are otherwise easy to conflate:

1. momentum structure is identifiable and wins under exact sparse evidence;
2. candidate-pruned momentum was losing because true paths fell off support;
3. momentum is genuinely hard to distinguish from diffusion under the current
   observation model, time binning, and event durations.

Example:

```bash
hipporeplayimm benchmark data/DataSetFromPfeifferFoster \
  --models sorted-spike-state-space-diffusion,sorted-spike-state-space-momentum-exact-sparse,sorted-spike-state-space-momentum \
  --time-bin-ms 3 \
  --output results/exact_sparse_momentum_smoke
```

For synthetic recovery, the exact sparse row is the default paper-facing
momentum scorer; keep the candidate-pruned row when you want a lower-bound
support audit:

```bash
hipporeplayimm simulate-recovery data/DataSetFromPfeifferFoster \
  --session Rat1/Open1 \
  --true-models momentum,diffusion \
  --models sorted-spike-state-space-diffusion,sorted-spike-state-space-momentum-exact-sparse,sorted-spike-state-space-momentum \
  --output results/simulation_recovery_exact_sparse_momentum
```

The candidate-pruned `sorted-spike-state-space-momentum` row remains useful as a
speed/sensitivity diagnostic, but exact sparse momentum should be preferred for
headline momentum-vs-diffusion evidence comparisons.
