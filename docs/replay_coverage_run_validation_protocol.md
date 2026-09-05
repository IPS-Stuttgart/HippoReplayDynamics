# Training-Only RUN Decoder Validation

Frozen before inspecting real held-out results. This is an encoding prerequisite
for the recording-coverage study, not a test of biological replay uniformity.

## Inputs and Separation

Use all 33 cached PF/Tanni sessions. Load raw position, sorted spikes, native
arena bounds where known, and tracking-supported RUN intervals. The PF source
excitatory-cell annotation may be used; all-RUN firing-rate, stability, or other
unit-selection metrics may not. Do not load full-RUN maps or support masks.

Split each session's RUN clock span into five chronological blocks. For each
held-out block, remove its positions and spikes plus a one-second guard from
training. Re-estimate the spatial grid, occupancy, rate maps, and RUN unit QC
using only the remaining training data. Tracking gaps are never interpolated
into training occupancy or test windows. Unit thresholds and 8 cm grid / 1.5-bin
smoothing remain the previously frozen input defaults. No relaxation based on
the test results is permitted; failed folds remain in the output denominator.

The canonical split-half stability filter is recomputed within each fold's
training data. Known physical arena dimensions are external metadata, not a
function of test positions. PF tracking extrema are not labeled as arena walls.

## Test Windows

Choose centers at a 250 ms stride, from behavior only, with center running speed
at least 10 cm/s. The entire 250 ms window must lie inside the held-out block
and a tracking-supported RUN interval. Uniformly sample at most 250 eligible
centers per fold with seed 20260906 and stable session/fold keys. At least 20
eligible test centers are required per fold. Retain zero-spike windows.

At each center decode both a 250 ms and a 20 ms window, independently with a
uniform spatial prior. Use Poisson likelihood as primary and count-conditioned
multinomial as a labeled sensitivity. No temporal motion model is used.
Repeat with five seeded cell-identity permutations of training rate-map rows
as descriptive wrong-map controls. They are not calibrated hypothesis tests.

## Outputs and Interpretation

Retain per-window posterior-mean and MAP error to tracked center position,
error to the exact piecewise-linear window-mean position, posterior RMS,
entropy, cell/spike support, and nominal 50/80/95% highest-density-set coverage.
Include probability ties. True positions outside training support count as
coverage misses; also report a support-conditional sensitivity. Error thresholds
of 8/16/20/40 cm are descriptive sensitivity columns, not a selected paper gate.

Report all behavior-selected windows as primary, with >=2 cells / >=3 spikes
as a secondary conditional subset. Session and animal summaries must distinguish
these populations. Five folds and thousands of windows are not independent
animals. The figure's calibration panel is descriptive session-mean coverage.

Write predictions, folds, training-unit QC, session/animal summaries, technical
gates, hashed input/code manifest, training fold models, figure, and markdown.
The technical gates require nonempty complete fold/window/likelihood coverage,
finite error outputs, and unchanged code/input hashes; they do not assert good
decoding or calibrated posterior uncertainty.

Very short RUN windows may represent theta sequences rather than the animal's
instantaneous physical position, and spike support differs from replay. Poor
20 ms RUN coverage is not by itself a decoder bug or a quantitative estimate of
replay error. Good RUN decoding likewise cannot validate replay speed without
ground-truth simulation and recovery tests. Rare extreme tracking speeds are
flagged, not silently filtered; later sensitivity must be separately labeled.
