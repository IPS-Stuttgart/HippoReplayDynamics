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
  --models "random stationary sorted-spike-state-space-diffusion sorted-spike-state-space-momentum-exact-sparse sorted-spike-state-space-momentum sorted-spike-state-space-imm sorted-spike-state-space-goal sorted-spike-state-space-goal-bidirectional" \
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

Momentum velocity decay can now be expressed in the same physical-time form:

```bash
--state-space-momentum-velocity-decay-tau-s 0.060
```

When this value is positive, each transition uses
`exp(-transition_duration_s / tau)` instead of a fixed per-bin velocity decay.
This keeps momentum settings comparable across 1, 2, 3, and 5 ms replay bins.

The state-space momentum/IMM beam can also use a train-only first-order
diffusion posterior as its support source:

```bash
--state-space-momentum-candidate-source posterior
```

Use this as a diagnostic beside the default emission-ranked support; it can
recover dynamically plausible bins whose instantaneous emission rank is too low
for a fixed top-k beam.

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

## Result-quality audit

After a model-evidence run, write a single audit dashboard and diagnostic CSVs:

```bash
python scripts/audit_model_evidence_results.py \
  --scores results/model-evidence/event_model_evidence.csv \
  --output results/model-evidence-audit
```

The audit adds evidence-margin summaries, model-disagreement events,
candidate-support quality tables, window-sensitivity tables when window variants
are present, session/rat influence summaries, null-control recommendations,
adversarial synthetic case suggestions, and provenance warnings.

If you rerun selected low-margin events with a common candidate support, pass the
second score table to compare native and common-support evidence:

```bash
python scripts/audit_model_evidence_results.py \
  --scores results/model-evidence/event_model_evidence.csv \
  --common-support-scores results/common-support/event_model_evidence.csv \
  --output results/model-evidence-audit
```

Observation-model calibration sweeps can be selected without using real replay
evidence by passing a validation/recovery summary and optional gates:

```bash
python scripts/audit_model_evidence_results.py \
  --scores results/model-evidence/event_model_evidence.csv \
  --observation-sweep-summary results/observation_sweep_summary.csv \
  --max-behavior-error-cm 15 \
  --min-recovery-accuracy 0.60 \
  --output results/model-evidence-audit
```
