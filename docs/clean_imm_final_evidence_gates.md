# Clean IMM Final Evidence Gates

This note keeps the strict first-order IMM claim separate from broader
trajectory-family evidence. The paper-facing phrase should only be used for the
subset that passes all relevant gates:

```text
time-order-sensitive, posterior-trajectory-active clean IMM events
```

## Gate Ladder

1. IMM beats fragmented by calibrated paired evidence.
   - Primary table: `imm_fragmented_head_to_head_event_table.csv`
   - Claim role: separates clean IMM candidates from fragmented or ambiguous
     trajectory-family positives.

2. IMM advantage depends on temporal order.
   - Primary workflow: `clean-imm-time-order-shuffle-control.yml`
   - Primary tables:
     - `clean_imm_time_order_shuffle_decisions.csv`
     - `clean_imm_time_order_shuffle_gate_summary.csv`
   - Claim role: tests whether the IMM-vs-fragmented advantage collapses when
     whole time bins are shuffled within events.

3. IMM posterior uses nonstationary trajectory content.
   - Primary workflow: `first-order-imm-event-mean-mode-usage-audit.yml`
   - Primary tables:
     - `first_order_imm_mode_usage_event_summary.csv`
     - `first_order_imm_mode_usage_gate_summary.csv`
   - Required diagnostics include mean nonstationary mode probability, fraction
     time MAP-nonstationary, nonstationary bout count, posterior expected path
     length, and posterior net displacement.
   - Older artifacts with terminal-only IMM mode probabilities should fail this
     gate and must not be used for posterior-trajectory-active wording.

4. IMM beats fragmented under held-out-cell scoring.
   - Primary workflow: `cell-split-heldout-control.yml`
   - Primary tables:
     - `cell_split_heldout_imm_vs_fragmented.csv`
     - `cell_split_heldout_imm_vs_fragmented_summary.csv`
     - `rat_cell_split_heldout_imm_vs_fragmented_summary.csv`
     - `cell_split_heldout_imm_vs_fragmented_gate_summary.csv`
   - Claim role: tests whether clean IMM predictive advantage survives held-out
     cell scoring rather than only fitting the cells used for encoding.

## Claim Boundary

If Gates 1 and 2 pass but Gate 3 or Gate 4 fails, keep the wording conservative:

```text
time-order-sensitive first-order IMM model-evidence candidates
```

If Gates 1 through 4 pass on the strict subset, the stronger wording is
defensible:

```text
A strict subset of replay-candidate events contains time-order-sensitive,
posterior-trajectory-active switching dynamics that predict held-out cells
better than fragmented dynamics.
```
