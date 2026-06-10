# First-Order IMM Mode-Usage Audit

`scripts/audit_first_order_imm_mode_usage.py` checks whether the leading
first-order IMM row supports only a trajectory-capable model-class claim or the
stronger claim that the event posterior is actually nonstationary.

Example:

```bash
python scripts/audit_first_order_imm_mode_usage.py \
  --event-model-evidence path/to/all_sessions_event_model_evidence.csv \
  --output results/first-order-imm-mode-usage-audit-27011374643 \
  --margin-threshold 5.5
```

Comparison example:

```bash
python scripts/audit_first_order_imm_mode_usage.py \
  --event-model-evidence path/to/all_sessions_event_model_evidence.csv \
  --promoted-off-swr-event-model-evidence path/to/promoted_off_swr_candidate_exact_core_event_model_evidence.csv \
  --one-per-source-decisions path/to/off_swr_candidate_one_per_source_group_decisions.csv \
  --output results/first-order-imm-mode-usage-audit \
  --margin-threshold 5.5
```

Outputs:

- `first_order_imm_mode_usage_event_table.csv`
- `first_order_imm_mode_usage_event_summary.csv`
- `first_order_imm_mode_usage_summary.csv`
- `first_order_imm_mode_usage_rat_summary.csv`
- `first_order_imm_mode_usage_gate_summary.csv`
- `swr_off_swr_first_order_imm_mode_usage_comparison.csv`
- `rat_first_order_imm_mode_usage_summary.csv`
- `off_swr_one_per_source_group_mode_usage_summary.csv`
- `off_swr_one_per_source_group_posterior_content_gate.csv`

The gate summary intentionally separates:

- `trajectory_capable_model_class_claim_supported`: exact trajectory-capable
  models beat the stationary/static comparator.
- `terminal_nonstationary_majority_among_first_order_imm_best`: terminal IMM
  mode probabilities lean nonstationary.
- `posterior_content_claim_supported`: event-mean IMM mode probabilities are
  present and mostly nonstationary.

Older artifacts may only contain terminal mode probabilities. In that case the
audit can complete, but `posterior_content_claim_supported` remains false. Use
the safer wording "trajectory-capable exact switching models" until an artifact
with event-mean first-order IMM mode masses is available.

When comparing promoted off-SWR candidates, the script groups candidate windows
by `session`, `event_index`, and `null_index` so multiple promoted windows from
the same source event remain separate. The one-per-source comparison uses the
de-duplication decision table to select representative candidates, then applies
the same posterior-content gates.
