# HippoReplayIMM

`hipporeplayimm` is a standalone Python package for benchmarking motion-model and
interacting-multiple-model (IMM) interpretations of hippocampal replay in the
Pfeiffer/Foster open-field dataset.

## Installation

Install the core package and its development test dependencies from a checkout
with:

```bash
python -m pip install -e ".[dev]"
```

PyRecEst-backed particle models are optional. Install the additional extra only
when running `pyrecest-goal-particle`, `pyrecest-goal-particle-imm`, or the
PyRecEst sweep workflows locally:

```bash
python -m pip install -e ".[pyrecest]"
```

For local development that also runs the PyRecEst-focused test subset, use:

```bash
python -m pip install -e ".[dev-pyrecest]"
```

Without the PyRecEst extra, the core benchmarks, state-space models, clusterless
models, and non-PyRecEst tests remain available; PyRecEst-specific tests and
models are skipped or unavailable until the extra is installed.

The raw dataset is expected to remain outside the repository, for example:

```powershell
hipporeplayimm inspect D:\Uni-Data\DataSetFromPfeifferFoster
hipporeplayimm benchmark D:\Uni-Data\DataSetFromPfeifferFoster --max-events 25 --output results
hipporeplayimm decode-event D:\Uni-Data\DataSetFromPfeifferFoster --session Rat1/Open1 --event-id 0 --output results\decode_event
hipporeplayimm validate-position D:\Uni-Data\DataSetFromPfeifferFoster --session Rat1/Open1 --output results\position_validation
hipporeplayimm simulate-recovery D:\Uni-Data\DataSetFromPfeifferFoster --session Rat1/Open1 --output results\simulation_recovery
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
  Use `sorted-spike-state-space-momentum-exact-sparse` for paper-facing
  exact/comparable momentum evidence; the candidate-pruned row is a lower-bound
  support audit unless it keeps the full grid.
- `imm`: candidate-pruned switching model over stationary, diffusion, momentum,
  and jump/fragmented modes.
- `sorted-spike-state-space-stationary`, `sorted-spike-state-space-diffusion`,
  `sorted-spike-state-space-fragmented`, `sorted-spike-state-space-jump`,
  `sorted-spike-state-space-momentum-exact-sparse`,
  `sorted-spike-state-space-momentum`, and `sorted-spike-state-space-imm`:
  sorted-spike Poisson state-space baselines intended for 1-3 ms replay bins.
  The first-order modes return full forward/backward trajectory posteriors.
  Momentum uses a candidate-pruned second-order recursion and reports that its
  trajectory posterior is candidate-supported. The older `state-space-*` aliases
  are accepted but benchmark output uses the explicit `sorted-spike-*` names.
- `clusterless-state-space-stationary`, `clusterless-state-space-diffusion`,
  `clusterless-state-space-fragmented`, `clusterless-state-space-jump`,
  `clusterless-state-space-momentum`, and `clusterless-state-space-imm`:
  state-space baselines using clusterless marked-point-process emissions when
  spike-mark features are present. The default mark model is a spatially local
  KDE fit from run-period spike marks; `--clusterless-mark-likelihood
  diagonal-gaussian` keeps the older diagonal-Gaussian approximation available
  for ablation runs.
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

The sorted-spike state-space models use sorted-unit spike identities and Poisson
place-field emissions. The `clusterless-state-space-*` models instead use
detected spike marks and report the selected `clusterless_mark_likelihood` in
diagnostics. Position-validation still validates the sorted-spike Poisson encoder.

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
  --models random,stationary,sorted-spike-state-space-diffusion,sorted-spike-state-space-momentum-exact-sparse,sorted-spike-state-space-momentum,sorted-spike-state-space-imm `
  --output results\state_space_smoke
```

Synthetic replay-dynamics recovery can be run with the same validated encoder
settings. It generates Poisson spike-count emissions from fitted place fields
under known latent dynamics and then scores those synthetic events with the
state-space model-evidence stack:

```powershell
hipporeplayimm simulate-recovery D:\Uni-Data\DataSetFromPfeifferFoster `
  --session Rat1/Open1 `
  --events run `
  --max-template-events 25 `
  --events-per-model 25 `
  --time-bin-ms 3 `
  --bin-size-cm 6.0 `
  --smoothing-sigma-bins 2.0 `
  --min-speed-cm-s 5.0 `
  --output results\simulation_recovery_rat1_open1
```

The recovery benchmark writes `simulation_recovery_event_scores.csv`,
`simulation_recovery_confusion_matrix.csv`, `simulation_recovery_summary.csv`,
and `simulation_recovery_settings.yml`. The manual `Simulate replay recovery`
workflow runs the same command on GitHub Actions and uploads these files as a
`simulation-recovery-<run_id>` artifact.

The model-evidence workflow exposes the same encoder settings. The validated
behavioral-decoding settings are `--decode-bin-s 1.0`, `--bin-size-cm 6.0`,
`--smoothing-sigma-bins 2.0`, and `--min-speed-cm-s 5.0`. These settings passed
the Rat3/Open1 and Rat3/Open2 position-validation matrix with median
posterior-mean errors below 15 cm and median MAP errors below 20 cm.
For full sessions, use the manual `Benchmark replay model evidence
event-sharded` workflow so momentum-dominated state-space runs are split across
event shards and aggregated into the standard model-evidence CSV schema.
The `diffusion_sigma_cm` and `momentum_sigma_cm` workflow inputs configure only
the older candidate-pruned models. For `sorted-spike-state-space-*` models, use
the explicit `state_space_*` workflow inputs, whose defaults reproduce the
original state-space settings: diffusion and momentum noise `85 cm/sqrt(s)`,
momentum velocity decay `0.95`, and momentum candidate support `128` bins.
Use the manual `State-space replay evidence parameter sweep` workflow for a
small reproducible dynamics sweep over state-space diffusion noise, momentum
noise, initial momentum noise, velocity decay, and momentum candidate support.
The default sweep is capped to 25 `Rat1/Open1` run events and uploads ranked
momentum-vs-diffusion comparison tables.
Use the manual `Simulation recovery parameter sweep` workflow as the matching
synthetic-identifiability check. Its default grid simulates known diffusion and
momentum events from the fitted sorted-spike Poisson encoder and ranks settings
by momentum recovery accuracy before those settings are trusted on real replay.
After both sweeps finish, use the manual `Select state-space replay parameters`
workflow to join the evidence and recovery summary artifacts. It uploads a
decision table, the configurations passing the recovery gate, and one
recommended parameter row, plus a JSON provenance manifest, workflow-input YAML,
and CLI arguments for the selected settings, so real-event evidence is not tuned
without synthetic identifiability.
Use the manual `Compare model-evidence runs` workflow to compare KD-aligned and
state-space model-evidence artifacts by canonical dynamics labels. The default
inputs compare the KD-aligned event-sharded run `25435692734` against the
state-space event-sharded run `25744259285` and upload best-model agreement,
canonical crosstabs, and paired relative-evidence tables.

Clusterless state-space evidence is included in the event-sharded workflow
defaults so local-KDE marked-point-process evidence is scored beside
sorted-spike Poisson evidence. To run only clusterless models, replace the model
list, for example:

```text
clusterless-state-space-stationary clusterless-state-space-diffusion clusterless-state-space-momentum clusterless-state-space-imm
```

The workflows expose `clusterless_mark_likelihood`,
`clusterless_mark_smoothing_sigma_bins`, `clusterless_mark_prior_count`,
`clusterless_mark_variance_floor`, `clusterless_rate_floor_hz`,
`clusterless_mark_kde_bandwidth`, `clusterless_mark_kde_spatial_sigma_bins`, and
`clusterless_mark_kde_max_neighbors` for local-KDE/Gaussian mark modeling and
spike-intensity calibration.
The event-sharded aggregator rejects attempts to combine shards with different
clusterless or encoder settings, including spike-rate scale and clusterless KDE
settings, so aggregate model-evidence tables remain provenance-consistent.

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

Do not interpret a single 512-particle PyRecEst run as a stable model ranking.
For result tables, run multiple particle counts, nonzero position-proposal
probabilities, and several random seeds, then compare only seed-aggregated
Pareto summaries. The manual `PyRecEst particle robustness sweep` workflow runs
this sweep on GitHub Actions and uploads the same CSV artifacts as the CLI.

Small reproducible PyRecEst parameter sweeps can be run with:

```powershell
hipporeplayimm sweep-pyrecest D:\Uni-Data\DataSetFromPfeifferFoster `
  --max-events 5 `
  --random-seeds 1,2,3 `
  --pyrecest-models pyrecest-goal-particle,pyrecest-goal-particle-imm `
  --particles 2048,4096,8192 `
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
The session-scoped `Benchmark Pfeiffer-Foster held-out likelihood` workflow also
exposes all PyRecEst scalar parameters for focused debugging runs, but it should
be treated as a single-seed diagnostic unless the same configuration is repeated
across several `random_seed` values.

A production-sized sweep can be launched manually from Actions with defaults
equivalent to:

```text
random_seeds: 1,2,3
pyrecest_models: pyrecest-goal-particle,pyrecest-goal-particle-imm
particles: 2048,8192
position_proposal_probability: 0.0,0.5
alpha: 0.8
position_jump_sigma_cm: 25.0
jump_probability: 0.03
imm_mode_stickiness: 0.95
```

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
  --min-occupancy-s 0.02 `
  --rate-floor-hz 1e-4 `
  --smoothing-sigma-bins 2.0 `
  --output results\position_validation_rat1_open1
```

## Observation-Model Calibration Sweep

`hipporeplayimm sweep-observation` is the recommended first gate before tuning
replay dynamics. It sweeps the sorted-spike Poisson encoder and replay-bin
calibration settings, writes cross-validated behavior-decoding metrics, and
optionally runs synthetic state-space recovery under the same observation
settings:

```powershell
hipporeplayimm sweep-observation D:\Uni-Data\DataSetFromPfeifferFoster `
  --sessions Rat1/Open1 `
  --bin-size-cm 4,6 `
  --smoothing-sigma-bins 1.5,2.0 `
  --min-occupancy-s 0.01,0.02,0.05 `
  --rate-floor-hz 1e-5,1e-4 `
  --time-bin-ms 2,3,5 `
  --spike-rate-scale 0.5,1.0,2.0 `
  --max-windows 1000 `
  --simulation-events-per-model 10 `
  --output results\observation_sweep_rat1_open1
```

The primary file is `observation_sweep_summary.csv`. Prefer settings with low
behavioral posterior-mean/MAP error and acceptable synthetic recovery accuracy
before interpreting model-evidence differences on real replay events.

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
