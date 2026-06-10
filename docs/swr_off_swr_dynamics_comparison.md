# SWR vs off-SWR dynamics comparison

`scripts/compare_swr_off_swr_dynamics.py` compares the exact full-core
trajectory-family signature of detected replay/SWR events with promoted off-SWR
candidate windows.

The intended inputs are:

```bash
python scripts/compare_swr_off_swr_dynamics.py \
  --swr-event-model-evidence path/to/all_sessions_event_model_evidence.csv \
  --off-swr-event-model-evidence path/to/promoted_off_swr_candidate_exact_core_event_model_evidence.csv \
  --off-swr-decisions path/to/promoted_off_swr_candidate_exact_core_decisions.csv \
  --off-swr-high-specificity-candidates path/to/off_swr_high_specificity_candidate_table.csv \
  --output results/swr-off-swr-dynamics-comparison \
  --margin-threshold 5.5
```

The comparison writes:

- `swr_off_swr_dynamics_comparison.csv`
- `swr_off_swr_model_winner_summary.csv`
- `swr_off_swr_family_margin_summary.csv`
- `swr_off_swr_rat_session_summary.csv`
- `swr_off_swr_behavior_summary.csv`
- `swr_off_swr_gate_summary.csv`

The primary gate checks whether both classes have trajectory-family support and
whether first-order IMM is the leading exact trajectory row in both detected
replay/SWR events and promoted off-SWR candidates. The behavior summary also
reports whether promoted off-SWR candidates are immobile and distant from known
SWR detections, and, when a high-specificity candidate table is supplied, whether
rejected high-specificity candidates are movement-skewed.

Paper-safe interpretation:

```text
Strict off-SWR replay-like candidates share the trajectory-family dynamical
signature of detected replay/SWR events when both classes are trajectory-family
confident and first-order IMM is the dominant exact trajectory row in both. This
supports a shared latent trajectory-family dynamics interpretation, while
remaining careful that detected SWR absence is not the same as ripple-power
negativity unless LFP evidence is added.
```
