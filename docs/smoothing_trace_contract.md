# First-order smoothing trace contract

The replay decoder now exposes a versioned, auditable forward/backward trace
through `hipporeplayimm.state_space.first_order_smoothing_trace`.

## Transition convention

HippoReplayDynamics uses a column-stochastic transition matrix:

```text
T[destination, source] = P(x_t = destination | x_(t-1) = source)
predicted[t] = T @ filtered[t - 1]
```

Every column must sum to one. Bayesian-ACh uses the equivalent row-stochastic
form `P[source, destination]`, so the exact bridge is `T = P.T`. The public
function rejects a row-stochastic matrix when its columns do not also sum to
one; this prevents a silent transpose error. A doubly stochastic matrix is
mathematically ambiguous, so callers must still record the declared convention.

## Returned quantities

Schema `hipporeplayimm.first-order-smoothing-trace.v1` returns:

- `predicted_probabilities[t]`: conditions only on emissions before time t;
- `filtered_probabilities[t]`: conditions on emissions through time t;
- `smoothed_probabilities[t]`: conditions on the complete supplied interval;
- `backward_messages[t]`: forward-scale-normalized backward messages, not
  categorical probabilities;
- prefix-conditioned and fixed-interval pair marginals, stored sparsely with
  axes `[source, destination]`;
- per-row log-emission offsets and forward scales;
- offset-restored log predictive probabilities, online surprise, and total log
  evidence.

Pair marginals remain sparse because a dense spatial grid would require
quadratic memory. `pair_probability_array` is available only for small
diagnostics.

## Log-emission offsets

For numerical stability, each log-emission row is shifted by its largest finite
value on the active support. The shifts are returned as `emission_offsets`.
All posterior probabilities and pair marginals are invariant to arbitrary
per-row offsets. Absolute quantities are reconstructed as

```text
log_predictive[t] = log(forward_scale[t]) + emission_offset[t]
log_evidence = sum_t log_predictive[t]
```

This is important for a replay artifact: saving only scaled likelihoods without
their offsets destroys absolute evidence even though posterior trajectories
still look correct.

## Causality boundary

Prediction and filtering are prefix-only. Smoothing deliberately uses later
emissions inside the supplied interval. A downstream pre-replay revision field
may use smoothing only for historical snippets whose complete interval ends
before the replay event. It may not include replay emissions or any behavioral
outcome observed after replay.

Posterior replay samples are not new observations. Feeding decoded replay back
into the same smoother would double-count the neural data that created the
posterior.

## Verification

The golden tests cover:

1. equality to the current HippoReplayDynamics first-order smoothed trajectory;
2. exhaustive enumeration of every path and pair in an asymmetric,
   time-varying two-state model;
3. equality to the Bayesian-ACh row-stochastic recursion after transposition;
4. invariance to arbitrary per-row log-emission offsets;
5. prefix non-leakage when a future emission is changed;
6. exact valid-state support and impossible emissions; and
7. explicit rejection of convention and occupancy-mask violations.
