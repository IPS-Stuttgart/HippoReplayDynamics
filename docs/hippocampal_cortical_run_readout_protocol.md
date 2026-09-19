# Independent RUN spatial-readout feasibility

This is a prerequisite check, not replay scoring, a two-context classifier, or
evidence for memory transfer. Exact maze labels remain unresolved. Do not turn a
weak spatial readout into a claim that context coding is absent: these differ.

## Input and source convention

Use all 15 checksum-verified navigation files from published DANDI 001695,
version 0.260319.2023. No POST outcome has been examined. Separate position streams
at timestamp gaps greater than one second. Call these recorded segments, not
verified mazes. The previous inventory gives 18 segments (360 folds below).

The authors' pinned example `pCCA_example_CA1-CA3-RSC.ipynb`, commit
ff579297beb12a3bbe86dde12b7dc39857315d54, selects CA1/CA3 with the literal
Pyramidal Cell label but RSC with cell_type != Narrow Interneuron. In the export,
the latter includes Wide Interneuron labels and one Unknown unit. Record that
distinction, do not silently relabel cell types. The author-rule RSC counts sum
to their published 708 cells in 12 no-gap files and 119 in the three gap-candidate
files; literal export-pyramidal counts are 207 and 24. Treat this as source-rule
reconciliation, not new cell-type classification or decoder validation.

Freeze four cohorts: CA1_pyramidal, CA3_pyramidal, RSC_author_non_narrow, and
RSC_export_pyramidal. The latter is a sensitivity check, not an alternative from
which to select a favorable result. The source uses different cell inclusion
for its replay analysis; this procedure does not reproduce that replay analysis.

## Fixed readout

- Only observed position epochs; actual timestamps, no 25-Hz repeat-and-trim
  alignment or interpolation across gaps. Use native speed >2.5 cm/s, consistent
  with the authors' movement filter. Require at least five finite frames in a
  complete, non-overlapping 250-ms bin. Include zero-spike test bins.
- Project 2D behavior onto its first principal axis within each segment, with
  >=90% explained variance. Normalize by the behavior-only 1st-99th percentile
  extent. This is a dimensionless label, NOT a centimetre conversion. Do not
  clip error targets to the normalized range or infer physical speed.
- Five contiguous elapsed-time folds per segment, one-second guard around each
  held-out block. Encoding-unit inclusion (>=20 spikes), firing maps, smoothing
  and rate estimates use training bins only. Require >=200 training bins,
  >=40 testing bins and >=2 training-active units; record failures explicitly.
- 25 position bins, Gaussian smoothing sigma one spatial bin, 0.25-s shrinkage
  toward the training mean firing rate. Pool movement directions for this
  feasibility check. No HMM, IMM, momentum, or temporal decoding prior.
- Multinomial population-composition likelihood conditional on total spike
  count, with a uniform position prior. Total population rate alone cannot
  identify position under an uninformative composition map.
- Report posterior-mean/MAP errors as fractions of the observed extent,
  training-median-position baseline error, unit/spike coverage, and error gain
  over 99 deterministic circular shifts of held-out RUN population-bin vectors.
  Shifts are 10-90% of eligible test-bin count; this is a compressed-RUN null,
  not a physical-time or state-matched POST null. Keep population vectors intact.
  Independent per-bin decoding permits shifting their already-computed outputs.

## Inference boundary and outputs

Save every fold, including failures, and per-fold encoding/prediction arrays on
the server. Aggregate folds within file, then equal file summaries within animal;
report successful fold/file denominators. A nominal shift p-value is a spatial
readout diagnostic, not a biological group-level test. All-complete technical
status remains separate from context readiness, which stays false.

Use a clean committed producer and detached server job with PID, log and terminal
status. Check input hashes before and after processing. Do not edit a live
producer, alter parameters based on these results, or use these RUN checks to
select a favorable POST event set. If feasible, the next requirement is the
author-confirmed context/epoch assignment, followed by blocked RUN context
validation and a separately frozen POST test.
