# HippoReplayIMM

`hipporeplayimm` is a standalone Python package for benchmarking motion-model and
interacting-multiple-model (IMM) interpretations of hippocampal replay in the
Pfeiffer/Foster open-field dataset.

The raw dataset is expected to remain outside the repository, for example:

```powershell
hipporeplayimm inspect D:\Uni-Data\DataSetFromPfeifferFoster
hipporeplayimm benchmark D:\Uni-Data\DataSetFromPfeifferFoster --max-events 25 --output results
hipporeplayimm decode-event D:\Uni-Data\DataSetFromPfeifferFoster --session Rat1/Open1 --event-id 0 --output results\decode_event
hipporeplayimm validate-position D:\Uni-Data\DataSetFromPfeifferFoster --session Rat1/Open1 --output results\position_validation
hipporeplayimm ground-truth D:\Uni-Data\DataSetFromPfeifferFoster --output results\behavioral_ground_truth.csv
hipporeplayimm compare-ground-truth D:\Uni-Data\DataSetFromPfeifferFoster --scores results\event_scores.csv --ground-truth results\behavioral_ground_truth.csv --output results\ground_truth_comparison.csv
```

The first implementation focuses on the eight `Rat1-4/Open1-2` sessions. It
fits occupancy-normalized Poisson place-field encoders from movement periods,
scores replay events with several motion priors, and reports held-out spike log
predictive density as the primary metric.

## Included Models

- `random`: independent uniform latent position at every time bin.
- `stationary`: one constant latent position throughout the event.
- `diffusion`: first-order Brownian/random-walk dynamics; the benchmark uses
  candidate-pruned scoring for speed, while `DiffusionModel` remains available
  for exact small-grid checks.
- `momentum`: candidate-pruned second-order dynamics with velocity persistence.
- `imm`: candidate-pruned switching model over stationary, diffusion, momentum,
  and jump/fragmented modes.
- `sorted-spike-state-space-stationary`, `sorted-spike-state-space-diffusion`,
  `sorted-spike-state-space-fragmented`, `sorted-spike-state-space-jump`,
  `sorted-spike-state-space-momentum`, and `sorted-spike-state-space-imm`:
  sorted-spike Poisson state-space baselines intended for 1-3 ms replay bins.
  The first-order modes return full forward/backward trajectory posteriors.
  Momentum uses a candidate-pruned second-order recursion and reports that its
  trajectory posterior is candidate-supported. The older `state-space-*` aliases
  are accepted but benchmark output uses the explicit `sorted-spike-*` names.
- `pyrecest-goal-particle`: PyRecEst-backed goal-conditioned particle replay
  filter using well-derived candidate goals when session metadata are available.
  It can optionally rejuvenate position particles from the current decoded grid
  likelihood when PyRecEst provides the proposal API.
- `pyrecest-goal-particle-imm`: PyRecEst-backed goal-conditioned particle IMM
  filter with per-particle switching among stationary, diffusion, momentum,
  goal-directed, and jump dynamics. The same optional position proposal knob is
  available for the particle IMM.

The candidate-pruned models use the same candidate sets for train and joint
likelihoods during held-out scoring, so `log p(train, test) - log p(train)` is
well-defined under the same approximate state support.

The state-space models currently use sorted-unit spike identities and Poisson
place-field emissions. `inspect`, benchmarks, and position-validation outputs
report detected spike-mark features, but the clusterless marked-point-process
likelihood is explicitly marked `not_implemented`.

Replay bin width can be changed with `--time-bin-ms` in `benchmark`,
`decode-event`, and `compare-ground-truth`; the state-space baselines are the
main target for 1-3 ms replay-bin experiments:

```powershell
hipporeplayimm benchmark D:\Uni-Data\DataSetFromPfeifferFoster `
  --max-events 25 `
  --time-bin-ms 3 `
  --bin-size-cm 6.0 `
  --smoothing-sigma-bins 2.0 `
  --min-speed-cm-s 5.0 `
  --models random,stationary,sorted-spike-state-space-diffusion,sorted-spike-state-space-momentum,sorted-spike-state-space-imm `
  --output results\state_space_smoke
```

The model-evidence workflow exposes the same encoder settings. The validated
behavioral-decoding settings are `--decode-bin-s 1.0`, `--bin-size-cm 6.0`,
`--smoothing-sigma-bins 2.0`, and `--min-speed-cm-s 5.0`. These settings passed
the Rat3/Open1 and Rat3/Open2 position-validation matrix with median
posterior-mean errors below 15 cm and median MAP errors below 20 cm.
For full sessions, use the manual `Benchmark replay model evidence
event-sharded` workflow so momentum-dominated state-space runs are split across
event shards and aggregated into the standard model-evidence CSV schema.

`decode-event --output` writes `event_scores.csv` plus posterior `.npz`
artifacts for models that expose `trajectory_log_posterior`. The batch tracking
scripts write the same posterior arrays for downstream plotting.

The PyRecEst goal-conditioned particle model is opt-in because it is stochastic
and more expensive:

```powershell
hipporeplayimm benchmark D:\Uni-Data\DataSetFromPfeifferFoster `
  --max-events 25 `
  --models random,stationary,imm,pyrecest-goal-particle,pyrecest-goal-particle-imm `
  --pyrecest-particles 512 `
  --output results\pyrecest_smoke
```

Small reproducible PyRecEst parameter sweeps can be run with:

```powershell
hipporeplayimm sweep-pyrecest D:\Uni-Data\DataSetFromPfeifferFoster `
  --max-events 5 `
  --random-seeds 1,2,3 `
  --pyrecest-models pyrecest-goal-particle,pyrecest-goal-particle-imm `
  --particles 128,512 `
  --position-proposal-probability 0.0,0.5,1.0 `
  --alpha 0.6,0.8 `
  --position-jump-sigma-cm 10,25 `
  --jump-probability 0.0,0.03 `
  --imm-mode-stickiness 0.9,0.98 `
  --output results\pyrecest_sweep
```

The sweep writes per-seed `sweep_summary.csv`, per-seed `pareto_summary.csv`,
seed-aggregated `aggregate_summary.csv`, seed-aggregated
`pareto_aggregate_summary.csv`, `event_scores.csv`, and, unless
`--skip-ground-truth` is passed, behavioral ground-truth comparison tables. The
CLI prints the aggregate Pareto summary so likelihood, goal accuracy, endpoint
error, and true-well posterior tradeoffs stay visible across stochastic seeds.

## Behavioral Position-Decoding Validation

`hipporeplayimm validate-position` cross-validates sorted-spike Poisson
place-field decoding on running behavior before interpreting replay results. It
splits running-position windows into folds, fits place fields on the remaining
movement frames, decodes held-out position windows, and writes:

- `position_decoding_samples.csv`: one row per held-out behavior window.
- `position_decoding_summary.csv`: median posterior-mean error, median MAP
  error, true-bin posterior probability, spike counts, mark availability, and
  the observation-model label.

Example smoke run:

```powershell
hipporeplayimm validate-position D:\Uni-Data\DataSetFromPfeifferFoster `
  --session Rat1/Open1 `
  --decode-bin-s 1.0 `
  --n-folds 5 `
  --max-windows 1000 `
  --bin-size-cm 6.0 `
  --smoothing-sigma-bins 2.0 `
  --output results\position_validation_rat1_open1
```

## Behavioral Ground-Truth Proxy

The real data do not contain latent replay trajectories as ground truth.
`hipporeplayimm ground-truth` derives a behavioral proxy from open-field well
fills and positions: a well filled at row `i` in `Well_Sequence.mat` is assigned
the median animal position just before row `i + 1`, then each run ripple is
labeled by the first valid post-ripple well visit.

`hipporeplayimm compare-ground-truth` merges this table with benchmark scores
and adds event-by-model fields such as decoded endpoint, decoded well,
`goal_correct`, `endpoint_error_cm`, `true_well_posterior`, and
`true_well_rank`.
