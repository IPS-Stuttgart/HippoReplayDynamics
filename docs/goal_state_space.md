# Goal-conditioned state-space replay model

`sorted-spike-state-space-goal` is a deterministic, exact full-grid alternative
to the stochastic PyRecEst goal-particle models. It augments the replay state
with one fixed candidate goal per event, applies a goal-directed drift-diffusion
transition for each candidate goal, and marginalizes the uniform goal prior.

When session metadata provide well locations, the benchmark uses those wells as
candidate goals. Without well metadata, the model falls back to a deterministic
farthest-point subset of the spatial grid. The legacy alias `state-space-goal`
is also accepted.

Example:

```powershell
hipporeplayimm benchmark D:\Uni-Data\DataSetFromPfeifferFoster `
  --max-events 25 `
  --time-bin-ms 3 `
  --models random,stationary,sorted-spike-state-space-diffusion,sorted-spike-state-space-goal `
  --output results\goal_state_space_smoke
```

The model reports exact-grid evidence via
`diagnostic_goal_state_space_evidence_support=exact_full_grid`, a trajectory
posterior, and terminal goal-posterior diagnostics such as
`diagnostic_goal_state_space_most_likely_goal_probability`.
