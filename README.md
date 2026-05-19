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
- `imm`: candidate-pruned switching model over stationary, diffusion, momentum,
  and jump/fragmented modes.
- `sorted-spike-state-space-stationary`, `sorted-spike-state-space-diffusion`,
  `sorted-spike-state-space-fragmented`, `sorted-spike-state-space-jump`,
  `sorted-spike-state-space-goal`,
  `sorted-spike-state-space-goal-bidirectional`,
  `sorted-spike-state-space-goal-forward-biased`,
  `sorted-spike-state-space-goal-forward-biased-switching`,
  `sorted-spike-state-space-goal-reverse-biased`,
  `sorted-spike-state-space-momentum`, and `sorted-spike-state-space-imm`:
  sorted-spike Poisson state-space baselines intended for 1-3 ms replay bins.
  The first-order modes return full forward/backward trajectory posteriors.
  Momentum uses a candidate-pruned second-order recursion and reports that its
  trajectory posterior is candidate-supported. The goal model is an exact
  first-order mixture over inferred candidate wells. It can optionally use an
  event-specific prior that favors the task well active at the ripple peak and
  an initial-position prior centered on the ripple-peak animal position. A
  direction mode can restrict that start prior to toward or away components,
  which lets bidirectional runs use a forward start anchor without penalizing
  reverse components.
  small reset probability can also be swept to tolerate fragmented position
  jumps while keeping the latent goal fixed. The bidirectional goal model
  marginalizes over forward and reverse sweeps relative to the same goal. A
  terminal goal prior can be swept to reward trajectories that end near their
  candidate well, and an initial goal prior can be swept to reward reverse
  sweep components that start near their candidate well. Reverse sweep
  components can also use a terminal position prior centered on the animal's
  ripple-peak position. A direction prior can be swept to change the
  bidirectional model's forward/reverse mixture weight.
  The forward-biased and reverse-biased goal aliases are bidirectional variants
  with fixed toward-direction prior weights of `0.9` and `0.1`, respectively,
  for direct model-comparison runs. The forward-biased-switching alias also
  fixes the component-switch probability at `0.03`, which was the best
  single-event smoke setting before full-session validation. The goal
  transition can also use a lateral
  sigma scale below `1.0` to make directed sweeps sharper perpendicular to the
  source-goal axis while preserving the along-axis transition noise.
  A diffusion-mixture transition weight can add a zero-drift local diffusion
  component inside each latent goal, which helps test replay events with pauses
  or weakly directed bins without using a full position reset.
  A component-switch probability can be swept to allow the latent goal or
  direction component to change between bins while preserving the predicted
  position distribution.
  The older `state-space-*` aliases are accepted but benchmark output uses the
  explicit `sorted-spike-*` names.
- `clusterless-state-space-stationary`, `clusterless-state-space-diffusion`,
  `clusterless-state-space-fragmented`, `clusterless-state-space-jump`,
  `clusterless-state-space-momentum`, and `clusterless-state-space-imm`:
  state-space baselines using clusterless marked-point-process emissions when
  spike-mark features are present. The first implementation uses a
  position-dependent spike-intensity model and a diagonal-Gaussian mark
  likelihood fit from run-period spike marks.
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
detected spike marks and report `clusterless_mark_likelihood=diagonal-gaussian`
in diagnostics. Position-validation still validates the sorted-spike Poisson
encoder.

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
  --models random,stationary,sorted-spike-state-space-diffusion,sorted-spike-state-space-goal,sorted-spike-state-space-goal-bidirectional,sorted-spike-state-space-goal-forward-biased,sorted-spike-state-space-goal-forward-biased-switching,sorted-spike-state-space-momentum,sorted-spike-state-space-imm `
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
The event-sharded workflow also exposes
`goal_state_space_reset_probability`; values above `0` add a per-bin uniform
position reset to the goal-state-space dynamics while keeping the latent goal
fixed. `goal_state_space_reset_initial_position_prior_weight` can blend reset
destinations toward the event initial-position prior when a ripple-position
start prior is enabled. It exposes
`goal_state_space_terminal_prior_sigma_cm`; values above `0` add a mean-one
terminal likelihood factor around each candidate goal for toward-goal dynamics.
`goal_state_space_terminal_goal_prior_weight` can soften that factor when the
terminal anchor overcommits.
It exposes `goal_state_space_initial_goal_prior_sigma_cm`; values above `0` add
a mean-one initial likelihood factor around each candidate goal for
away-from-goal dynamics, which makes the bidirectional model less dependent on a
uniform reverse-sweep start. `goal_state_space_initial_goal_prior_weight` can
soften the matching reverse-sweep start anchor.
It exposes `goal_state_space_toward_direction_prior_weight`; values above `0.5`
favor toward-goal components in bidirectional goal models, while values below
`0.5` favor away-from-goal components.
It exposes `goal_state_space_lateral_sigma_scale`; values below `1.0` make
goal-conditioned transitions narrower perpendicular to the source-goal axis and
can improve evidence for straight directed sweeps.
It exposes `goal_state_space_diffusion_mixture_weight`; values above `0` mix a
zero-drift diffusion transition into the goal-directed transition so the same
goal component can absorb local pauses or slow bins.
It exposes `goal_state_space_component_switch_probability`; values above `0`
let the exact recursion redraw the latent goal/direction component between
time bins, which can help events whose evidence changes target or sweep
direction mid-event.
It exposes
`goal_state_space_active_goal_prior_weight`; values above `0` assign that prior
probability to the well active at the ripple peak for goal-state-space models,
with the remaining mass spread over other inferred wells.
It also exposes `goal_state_space_ripple_position_prior_sigma_cm`; values above
`0` initialize the goal-state-space model from a Gaussian prior around the
animal's ripple-peak position. The companion
`goal_state_space_ripple_position_prior_weight` blends that Gaussian with the
uniform spatial start prior, so values below `1` keep a weaker behavioral-start
anchor when the full ripple-position prior overcommits.
`goal_state_space_initial_position_prior_direction_mode` controls whether this
start prior applies to `all`, `toward`, or `away` components. The `toward`
setting is useful when bidirectional goal models should compare forward sweeps
against reverse sweeps without forcing reverse components to start at the
animal's ripple position.
For reverse replay hypotheses, `goal_state_space_reverse_terminal_position_prior_sigma_cm`
adds the matching terminal position prior around the ripple-peak animal
position to away-from-goal components, and
`goal_state_space_reverse_terminal_position_prior_weight` softens that terminal
anchor.
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

Clusterless state-space evidence can be run with the same model-evidence
workflows by replacing the model list, for example:

```text
clusterless-state-space-stationary clusterless-state-space-diffusion clusterless-state-space-momentum clusterless-state-space-imm
```

The workflows expose `clusterless_mark_smoothing_sigma_bins`,
`clusterless_mark_prior_count`, `clusterless_mark_variance_floor`, and
`clusterless_rate_floor_hz` for the Gaussian mark and spike-intensity model.
The event-sharded aggregator rejects attempts to combine shards with different
clusterless or encoder settings, including spike-rate scale and clusterless rate
floor, plus goal prior settings, so aggregate model-evidence tables remain
provenance-consistent.

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
