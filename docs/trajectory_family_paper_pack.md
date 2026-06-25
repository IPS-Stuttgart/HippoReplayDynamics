# Trajectory-family paper pack

`scripts/build_trajectory_family_paper_pack.py` builds a compact paper-facing result pack from completed benchmark artifacts. It does not rerun model evidence. It recomputes the headline full-core trajectory-family summaries from the event-level evidence table and then copies the relevant control summaries into a single provenance-backed directory.

## Inputs

The full-core all-session artifact is required. The control artifacts are optional by default so a partial local audit can still be generated, but paper-ready runs should pass `--require-controls`.

Expected inputs:

- `model-evidence-all-sessions-27011374643`
- wrong-map control artifact
- event-window artifact `26884355750`
- cell-split artifact `26965909403`
- K=10 matched-null artifact `26886723196`
- K=50 lightweight matched-null artifact `27060148887`

The full-core artifact must contain `all_sessions_event_model_evidence.csv` or `event_model_evidence.csv`. Control artifacts are scanned recursively for their existing gate and summary CSVs.

## Usage

```bash
python scripts/build_trajectory_family_paper_pack.py \
  --full-core-artifact path/to/model-evidence-all-sessions-27011374643 \
  --wrong-map-artifact path/to/wrong-map-control-artifact \
  --event-window-artifact path/to/event-window-26884355750 \
  --cell-split-artifact path/to/cell-split-26965909403 \
  --matched-null-k10-artifact path/to/matched-null-26886723196 \
  --matched-null-k50-artifact path/to/matched-null-27060148887 \
  --output results/trajectory-family-paper-pack-27011374643 \
  --confidence-threshold 5.5 \
  --require-controls
```

For a local smoke test with only the full-core artifact:

```bash
python scripts/build_trajectory_family_paper_pack.py \
  --full-core-artifact ../model-evidence-all-sessions-27011374643 \
  --output results/trajectory-family-paper-pack-27011374643-local-smoke
```

Missing control inputs are recorded in `control_stack_summary.csv` and the per-control output table.

## Outputs

- `paper_claim_manifest.json`
- `main_trajectory_family_summary.csv`
- `rat_trajectory_family_summary.csv`
- `leave_one_rat_out_trajectory_family_summary.csv`
- `bootstrap_trajectory_family_summary.csv`
- `exact_core_model_winner_summary.csv`
- `paired_momentum_diffusion_summary.csv`
- `control_stack_summary.csv`
- `matched_null_summary.csv`
- `cell_split_summary.csv`
- `event_window_summary.csv`
- `wrong_map_summary.csv`
- `figure_source_manifest.csv`
- `trajectory_family_paper_claim_summary.md`

## Manifest

`paper_claim_manifest.json` records:

- `code_commit`
- `artifact_run_ids`
- `artifact_names`
- `artifact_digests`
- `model_list`
- `calibrated_row_parameters`
- `confidence_threshold`
- `event_count`
- `rat_session_coverage`
- `primary_claim`
- `explicit_caveats`

Directory artifact digests are SHA-256 hashes over sorted file names and file hashes. This makes the pack reproducible without embedding the source CSVs in the manifest.

## Interpretation

Use `main_trajectory_family_summary.csv` for the headline exact trajectory-family-vs-nontrajectory result. Use the rat, leave-one-rat-out, and bootstrap tables to verify cross-rat robustness.

Use `exact_core_model_winner_summary.csv` to keep the first-order IMM full-core winner statement separate from the family-level claim. Use `paired_momentum_diffusion_summary.csv` to report the recovered exact-sparse momentum-vs-diffusion axis.

`control_stack_summary.csv` is the first audit table to inspect. A paper-ready pack should have all supplied control artifacts marked `ok`; any omitted artifact is explicitly marked as missing. The per-control summary files keep source-table provenance through `artifact_label`, `artifact_run_id`, `artifact_path`, and `source_table` columns.
