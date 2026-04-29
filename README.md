# HippoReplayIMM

`hipporeplayimm` is a standalone Python package for benchmarking motion-model and
interacting-multiple-model (IMM) interpretations of hippocampal replay in the
Pfeiffer/Foster open-field dataset.

The raw dataset is expected to remain outside the repository, for example:

```powershell
hipporeplayimm inspect D:\Uni-Data\DataSetFromPfeifferFoster
hipporeplayimm benchmark D:\Uni-Data\DataSetFromPfeifferFoster --max-events 25 --output results
hipporeplayimm decode-event D:\Uni-Data\DataSetFromPfeifferFoster --session Rat1/Open1 --event-id 0
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
- `pyrecest-goal-particle`: PyRecEst-backed goal-conditioned particle replay
  filter using well-derived candidate goals when session metadata are available.
- `pyrecest-goal-particle-imm`: PyRecEst-backed goal-conditioned particle IMM
  filter with per-particle switching among stationary, diffusion, momentum,
  goal-directed, and jump dynamics.

The candidate-pruned models use the same candidate sets for train and joint
likelihoods during held-out scoring, so `log p(train, test) - log p(train)` is
well-defined under the same approximate state support.

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
  --pyrecest-models pyrecest-goal-particle,pyrecest-goal-particle-imm `
  --particles 128,512 `
  --alpha 0.6,0.8 `
  --position-jump-sigma-cm 10,25 `
  --jump-probability 0.0,0.03 `
  --imm-mode-stickiness 0.9,0.98 `
  --output results\pyrecest_sweep
```

The sweep writes `sweep_summary.csv`, `event_scores.csv`, and, unless
`--skip-ground-truth` is passed, behavioral ground-truth comparison tables.

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
