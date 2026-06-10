# Off-SWR Candidate Source-Event De-Duplication

`scripts/deduplicate_off_swr_candidates.py` audits whether promoted off-SWR
candidate validation survives after collapsing multiple windows around the same
source event.

Example:

```bash
python scripts/deduplicate_off_swr_candidates.py \
  --validation-decisions path/to/promoted_off_swr_candidate_exact_core_decisions.csv \
  --candidate-table path/to/off_swr_candidate_table.csv \
  --high-specificity-table path/to/off_swr_high_specificity_candidate_table.csv \
  --output results/off-swr-candidate-source-dedup \
  --margin-threshold 5.5
```

Outputs:

- `off_swr_candidate_source_event_group_summary.csv`
- `off_swr_candidate_one_per_source_group_decisions.csv`
- `off_swr_candidate_one_per_source_group_summary.csv`
- `off_swr_candidate_cluster_robustness_gate_summary.csv`

The one-per-source table applies three rules:

- strongest exact trajectory-minus-nontrajectory margin;
- strongest discovery trajectory-family margin;
- earliest promoted window.

The gate summary passes only when the de-duplicated source-event subset remains
nontrivial, trajectory-confident, nontrajectory-negative, positive-margin, and
keeps first-order IMM common under the strongest exact-margin rule. This is a
reviewer-facing guard against duplicate-window inflation in the promoted
off-SWR discovery.
