# Result-improvement extensions

This document describes the opt-in patch entry points added for improving and
auditing replay model-evidence results without changing the existing default
benchmarks.

## Improved model-evidence benchmark

Run a smoke benchmark with exact goal-conditioned and bidirectional hypotheses:

```bash
python scripts/benchmark_model_evidence_improved.py \
  --dataset-root data/DataSetFromPfeifferFoster \
  --session Rat1/Open1 \
  --events run:0-25 \
  --time-bin-s 0.003 \
  --models "random stationary sorted-spike-state-space-diffusion sorted-spike-state-space-momentum sorted-spike-state-space-imm sorted-spike-state-space-goal sorted-spike-state-space-goal-bidirectional" \
  --output results/model-evidence-improved
```

The improved benchmark now writes the same standard post-processing columns as
the base and event-sharded evidence paths: comparable-evidence flags,
candidate-support quality labels, event-level evidence-margin categories, and
model-averaged endpoint summaries when endpoint diagnostics are available.

The improved script exposes the existing exact goal-state-space model, reverse
and bidirectional wrappers, adaptive state-space momentum candidate support,
clusterless local-KDE controls, replay-specific emission calibration, spatial
shuffle controls, and model-averaged endpoint summaries.

## Candidate support and IMM switching

State-space second-order models can be run with larger emission supports and
momentum-predicted candidate augmentation:

```bash
--state-space-momentum-candidate-top-k 256 \
--state-space-momentum-predicted-candidate-top-k 16
```

For bin-width-invariant IMM tuning, use a physical-time switching parameter:

```bash
--state-space-imm-switch-tau-s 0.060
```

When this value is positive, the effective per-bin stickiness is
`exp(-time_bin_s / tau)` and is written to the output CSV.

## Replay emission calibration

The default remains the original Poisson observation model.  Opt-in alternatives
are available:

```bash
--replay-gain-mode event-cell \
--replay-gain-prior-count 10 \
--sorted-spike-emission-model negative-binomial \
--negative-binomial-dispersion 50
```

These options are intended for sensitivity analysis.  Final claims should still
be backed by held-out behavior validation and synthetic recovery.

## Spatial shuffle controls

Spatial-bin permutation controls can be added to every event/model pair:

```bash
--null-shuffles 25
```

This adds columns such as `spatial_shuffle_delta_vs_null_median` and
`spatial_shuffle_null_empirical_p_value`.

## Repeated held-out cell splits

Run the existing held-out benchmark over repeated train/test cell splits:

```bash
python scripts/repeated_cell_split_benchmark.py \
  data/DataSetFromPfeifferFoster \
  --random-seeds 1,2,3,4,5 \
  --time-bin-ms 3 \
  --output results/repeated-cell-splits
```

This writes per-seed event scores and summary tables plus an across-seed
aggregate summary.
