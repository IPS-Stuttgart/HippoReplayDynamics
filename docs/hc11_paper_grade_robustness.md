# hc-11 Paper-Grade Robustness Report

`scripts/report_hc11_paper_grade_robustness.py` upgrades an existing hc-11
model-evidence smoke into a paper-readiness audit. It is non-rescoring: it reads
an event-model evidence CSV, summarizes spread across animals and sessions, and
optionally incorporates a clean-IMM time-order shuffle artifact.

The intended claim boundary is:

```text
hc-11 remains an external positive smoke until the robustness gates pass.
```

In particular, the paper-grade gate requires a real hc-11 clean-IMM time-order
shuffle result. Without `--time-order-shuffle-decisions`, the report still
writes all tables but fails the paper-grade gate.

## Example

```bash
python scripts/report_hc11_paper_grade_robustness.py \
  --event-model-evidence results/hc11-achilles-1d-postnrem-mua-smoke-all290/event_model_evidence.csv \
  --output-dir results/hc11-paper-grade-robustness
```

With a clean-IMM shuffle artifact:

```bash
python scripts/report_hc11_paper_grade_robustness.py \
  --event-model-evidence results/hc11-achilles-1d-postnrem-mua-smoke-all290/event_model_evidence.csv \
  --time-order-shuffle-decisions results/hc11-clean-imm-time-order/clean_imm_time_order_shuffle_decisions.csv \
  --output-dir results/hc11-paper-grade-robustness
```

## Outputs

- `hc11_event_claim_table.csv`
- `hc11_by_animal_summary.csv`
- `hc11_by_session_summary.csv`
- `hc11_leave_one_animal_out_summary.csv`
- `hc11_animal_cluster_bootstrap.csv`
- `hc11_rat/animal_cluster_bootstrap.csv`
- `hc11_imm_vs_fragmented_audit.csv`
- `hc11_time_order_shuffle_clean_imm.csv`
- `hc11_gate_summary.csv`

## Gate Questions

The report answers:

- Is the trajectory-family signal spread across animals and sessions?
- Does the result survive leave-one-animal-out summaries?
- Do animal-cluster bootstrap lower bounds support the family-level margin?
- Is the first-order IMM advantage distinguishable from fragmented?
- Is the hc-11 clean-IMM advantage time-order dependent?
- Is stationary still present as a meaningful comparator?

Only `hc11_gate_summary.csv` should be used for the final promotion decision.
