# First-Order IMM Event-Mean Mode-Usage Audit

`scripts/audit_first_order_imm_event_mean_mode_usage.py` tests the stronger
posterior-content claim for first-order IMM winners:

```text
The winning first-order IMM row assigns sustained posterior mass to
nonstationary trajectory modes and expresses spatial displacement.
```

Example:

```bash
python scripts/audit_first_order_imm_event_mean_mode_usage.py \
  --event-model-evidence path/to/all_sessions_event_model_evidence.csv \
  --promoted-off-swr-event-model-evidence path/to/promoted_off_swr_candidate_exact_core_event_model_evidence.csv \
  --one-per-source-decisions path/to/off_swr_candidate_one_per_source_group_decisions.csv \
  --output results/first-order-imm-event-mean-mode-usage-audit \
  --path-threshold-cm 10
```

Dataset-backed workflow sequence:

```bash
# 1. Regenerate detected replay/SWR full-core evidence on the branch that
#    contains the first-order IMM event-mean diagnostics.
gh workflow run model-evidence-all-sessions.yml \
  -R IPS-Stuttgart/HippoReplayIMM \
  --ref <branch> \
  -f max_events=20 \
  -f spike_rate_scale=2.0 \
  -f emission_likelihood_temperature=0.3 \
  -f time_bin_s=0.004 \
  -f state_space_diffusion_sigma_cm_sqrt_s=60.0 \
  -f state_space_momentum_sigma_cm_sqrt_s=50.0 \
  -f state_space_momentum_initial_sigma_cm_sqrt_s=45.0 \
  -f state_space_momentum_velocity_decay=0.93 \
  -f state_space_max_step_sigma=3.0

# 2. Regenerate promoted off-SWR exact-core validation on the same branch.
gh workflow run off-swr-promoted-candidate-validation.yml \
  -R IPS-Stuttgart/HippoReplayIMM \
  --ref <branch> \
  -f discovery_run_id=27237703414 \
  -f candidate_filter=promotion-ready

# 3. Combine the regenerated artifacts into the posterior-content audit.
gh workflow run first-order-imm-event-mean-mode-usage-audit.yml \
  -R IPS-Stuttgart/HippoReplayIMM \
  --ref <branch> \
  -f detected_run_id=<new_model_evidence_run_id> \
  -f promoted_run_id=<new_promoted_validation_run_id>
```

Outputs:

- `first_order_imm_mode_usage_event_summary.csv`
- `first_order_imm_mode_usage_gate_summary.csv`
- `rat_first_order_imm_mode_usage_summary.csv`
- `session_first_order_imm_mode_usage_summary.csv`
- `swr_off_swr_first_order_imm_mode_usage_comparison.csv`
- `off_swr_one_per_source_group_mode_usage_summary.csv`

The moderate content gate requires:

```text
(mean_nonstationary_mode_probability >= 0.5
 OR fraction_time_map_nonstationary >= 0.5)
AND
(posterior_expected_path_length_cm >= 10
 OR posterior_net_displacement_cm >= 10)
```

The strong content gate requires:

```text
mean_nonstationary_mode_probability >= 0.5
AND fraction_time_map_nonstationary >= 0.5
AND posterior_expected_path_length_cm >= 10
```

Current scorer changes emit the required diagnostics for future first-order IMM
artifacts:

- event-mean stationary/diffusion/fragmented mode probabilities
- MAP stationary/nonstationary time fractions
- nonstationary bout count and longest bout duration
- posterior expected path length, net displacement, and path speed

Older artifacts that only contain terminal mode probabilities will run through
the audit but fail the posterior-content gates. Use this as a claim boundary,
not as a failed pipeline.

The workflow artifact is named
`first-order-imm-event-mean-mode-usage-audit-${run_id}` and includes the
source-event de-duplication tables used for the one-per-source off-SWR subset.
