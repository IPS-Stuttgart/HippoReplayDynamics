# Exact-sparse momentum viability gate

This gate is the next run after promoting
`sorted-spike-state-space-momentum-exact-sparse` to the paper-facing momentum
row.  It is intentionally smaller than a full evidence sweep: by default it
generates one synthetic diffusion event and one synthetic momentum event per
Pfeiffer/Foster open-field session, scores exact/comparable rows plus
candidate-pruned audit rows, and writes a pass/fail dashboard.

Run from the repository root:

```bash
python scripts/run_exact_sparse_momentum_gate.py data/DataSetFromPfeifferFoster \
  --output results/exact-sparse-gate \
  --continue-on-error
```

The script runs these scoring rows by default:

```text
sorted-spike-state-space-diffusion
sorted-spike-state-space-momentum-exact-sparse
sorted-spike-state-space-fragmented
sorted-spike-state-space-first-order-imm
sorted-spike-state-space-momentum
sorted-spike-state-space-imm
```

The first four are required exact/comparable gate rows.  Candidate-pruned
momentum and IMM are retained as lower-bound audit rows.

## Outputs

The top-level output directory contains:

```text
exact_sparse_momentum_gate.md
exact_sparse_momentum_gate_status.json
exact_sparse_momentum_gate_event_scores.csv
exact_sparse_momentum_gate_event_summary.csv
exact_sparse_momentum_gate_session_summary.csv
exact_sparse_momentum_gate_runtime_summary.csv
```

Each session also gets its native `simulate-recovery` output directory and a
`run.log` file with the exact command line.

## Pass criteria

Default pass criteria are:

```text
true momentum exact-surrogate recovery >= 0.70
true diffusion expected-model recovery >= 0.70
first-order IMM best-model fraction <= 0.80
no failures among the required exact/comparable rows
no missing required exact/comparable rows
```

Use `--aggregate-only` to re-score the pass/fail dashboard from already
completed session outputs:

```bash
python scripts/run_exact_sparse_momentum_gate.py data/DataSetFromPfeifferFoster \
  --output results/exact-sparse-gate \
  --aggregate-only
```

Use `--dry-run` to print the commands without launching the expensive scorer.
