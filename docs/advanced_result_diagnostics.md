# Advanced replay result diagnostics

This patch adds opt-in diagnostics that can be run after a model-evidence table
has been generated.  The goal is not to add yet another replay model, but to
make existing claims more robust by exposing hidden failure modes.

## What is included

| Improvement | Implementation |
| --- | --- |
| Multi-map / wrong-environment controls | `scripts/wrong_map_evidence_controls.py`, `wrong_map_delta_summary` |
| Cell stability and place-field quality filters | `scripts/place_field_quality_report.py`, `place_field_quality`, `stable_cell_ids` |
| Event-window sensitivity | `scripts/event_window_sensitivity_plan.py`, `event_window_variants`, `summarize_window_sensitivity` |
| Posterior-predictive spike-train checks | `posterior_predictive_count_checks`, `posterior_predictive_poisson_log_score` |
| Session/rat hierarchical summaries | `hierarchical_summary`, `hierarchical_bootstrap` |
| Influence diagnostics | `leave_one_group_influence`, `drop_one_cell_influence` |
| Calibrated evidence margins | `evidence_margin_table`, `add_evidence_margin_columns` |
| Common-support evidence checks | `common_support_from_emissions`, `common_support_audit` |
| Automated dashboards | `scripts/advanced_result_diagnostics.py`, `write_dashboard` |
| Adversarial synthetic generator catalog | `adversarial_synthetic_case_specs` |
| Clusterless mark-drift diagnostics | `mark_drift_diagnostics` |
| Pre/post behavioral context features | `context_conditioning_table` |
| Model-disagreement mining | `model_disagreement_events` |
| Hyperparameter provenance audit | `ProvenanceRecord`, `provenance_audit` |
| Regression-test scaffolding | `tests/test_advanced_result_diagnostics.py` |

## Typical usage

```bash
python scripts/advanced_result_diagnostics.py \
  --scores results/model-evidence/event_model_evidence.csv \
  --output results/model-evidence/advanced-diagnostics \
  --parameter-source synthetic_selected \
  --selection-run-id 123456789 \
  --selection-metric momentum_recovery_accuracy \
  --selection-passed-recovery-gate true \
  --selection-used-real-evidence false \
  --bootstrap-model sorted-spike-state-space-imm \
  --bootstrap-model sorted-spike-state-space-goal
```

The script writes a Markdown dashboard plus CSVs for evidence margins,
hierarchical summaries, influence diagnostics, disagreement mining, and
provenance warnings.

## Wrong-map controls

For same-rat Open1/Open2 controls:

```bash
python scripts/wrong_map_evidence_controls.py \
  --dataset-root data/DataSetFromPfeifferFoster \
  --event-session Rat1/Open1 \
  --map-session Rat1/Open2 \
  --events run:0-25 \
  --models "random stationary sorted-spike-state-space-diffusion" \
  --output results/wrong-map-rat1-open1-vs-open2
```

Then compare with the ordinary current-map run:

```bash
python scripts/advanced_result_diagnostics.py \
  --scores results/current-map/event_model_evidence.csv \
  --wrong-map-scores results/wrong-map-rat1-open1-vs-open2/event_model_evidence.csv \
  --output results/current-map/advanced-diagnostics
```

## Place-field quality filters

```bash
python scripts/place_field_quality_report.py \
  --dataset-root data/DataSetFromPfeifferFoster \
  --session Rat1/Open1 \
  --output results/place-field-quality-rat1-open1
```

Use the resulting `stable_cell_ids.txt` to define stable-cell-only sensitivity
runs or to inspect whether high evidence is driven by a small number of unstable
or unusually high-rate cells.

## Event-window sensitivity

If you have a CSV with `event_index,start,end`, generate window variants with:

```bash
python scripts/event_window_sensitivity_plan.py \
  --events ripple_windows.csv \
  --paddings-s 0,0.01,0.02 \
  --output results/window_sensitivity_plan.csv
```

The plan can be used by downstream benchmark wrappers to score original,
padded, and center-only windows.  Large model-ranking changes across these
windows indicate segmentation sensitivity.

## Evidence margins

Wins are categorized as:

- `tie`: Δ log evidence ≤ 1
- `weak`: 1 < Δ log evidence ≤ 3
- `strong`: 3 < Δ log evidence ≤ 10
- `decisive`: Δ log evidence > 10

This prevents a low-margin event from being interpreted like a decisive model
preference.

## Notes

These diagnostics are intentionally post-hoc and opt-in.  They should not be
used to tune final real-event claims unless the provenance fields explicitly
record that real evidence was used for selection.
