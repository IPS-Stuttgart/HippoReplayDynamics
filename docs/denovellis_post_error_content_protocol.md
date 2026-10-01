# Does post-error replay content predict behavioral correction?

Frozen protocol v1.0, seed 20261001. This is a new protocol for previously
inspected Denovellis data, not an independent replication of the earlier
exploratory rate-versus-correction analysis. Numeric choices are recorded in
`denovellis_post_error_protocol.json` and must not be changed to obtain support.

## Prior art and observable question

Denovellis et al., eLife 64505, Methods/Animals and data collection, specifies
outer-to-center and center-to-less-recently-visited-outer rules:
https://elifesciences.org/articles/64505 . Source release:
https://github.com/Eden-Kramer-Lab/replay_trajectory_paper . Data:
https://doi.org/10.7272/Q61N7ZC3 . The independently inspected author position
loader uses stored position and speed directly, including smoothed columns when
present; `cmperpixel` must not be applied again.

Shin et al. (2019), https://doi.org/10.1016/j.neuron.2019.09.012 , analyzes
past/future path prediction, learning stages, incorrect upcoming outbound choices,
and coordinated CA1/PFC replay (Figs. 3, 5, S6 and replay-prediction methods).
Gillespie et al. (2021), https://doi.org/10.1016/j.neuron.2021.07.029 , compares
replay content with future choices and correct/error repeat-phase performance.
Neither inspected analysis establishes the exact conditional contrast **after
a preceding outbound error**, comparing correct-alternative and mistaken-route
content jointly against a history/quality baseline. This bounded literature audit
does not establish exhaustive novelty. If further source audit identifies the
same established contrast, stop the novelty-driven branch and record it.

## Behavioral definitions

Reconstruct all visits from raw 2D position and documented well coordinates,
then compare with released linear-distance/exit-enter annotations. Consecutive
bouts at the same well are one visit unless tracking continuity is broken.
Missing position or gaps >250 ms reset task history; do not bridge them.
The first bout, not a later return to the same well, defines the analysis pause.
The frozen eligibility rule requires at least 95% agreement of raw-position and
released linear annotations at jointly near-well frames, aligned clocks, RUN
metadata, native-event table epoch coverage, and marks/tetrode metadata. Every
task-recorded RUN epoch is inventoried, including excluded epochs. An epoch
absent from the native table is not silently treated as having zero ripples.
Score each completed move using history that existed before its destination was
visited. The next required outer arm is not inferred from the next actual choice.

For each incorrect center-to-outer move, retain the next center pause and next
outbound outcome, or an explicit unclassifiable reason. Use the first 10 seconds
of that pause and >=0.5 s valid immobile exposure. Native SWRs must be wholly
contained in one valid immobile interval. The statistical unit is a behavioral
transition, including eligible zero-replay trials, not a ripple or time bin.
Missing DIO is unknown reward, never an omission. DIO channel identities and
clock scales must be independently checked before calling any pulse a reward.

## Neural and statistical validation

Raw clusterless marks, author hippocampal tetrode annotations and W-track graph
are necessary. Fit encoding on preceding RUN only. Independent flat-prior
decoding uses non-overlapping 20-ms bins and a 3-cm graph grid. Validate maximum
absolute posterior-weighted time/route-distance correlation against 1,000
whole-bin permutations, maximizing over both routes again in every null draw.
Require five populated bins, two active tetrodes and p<=.05. Shared-stem mass
is not assigned to either alternative. Unresolved events are not zero evidence.

Chronological held-out RUN balanced accuracy must reach .80 and each arm's
recall .75. Separate development from frozen validation. Use 1,000 replicates
per specified generator and require the upper 95% false-positive bound <=.075
for stationary/unordered sequence controls and history-only/generic-quality
statistical nulls. Report power over the fixed effect-size grid, not a selected
passing alternative. No threshold relaxation or favorable-animal replacement.

The primary L2 logistic model has lambda=1, animal-balanced weights, animal/day
intercepts, both content rates and all specified history/quality covariates.
Evaluate held-out complete recording days; standardize on training observations
only. Bootstrap animals 2,000 times and report leave-one-animal-out results.
Full support requires positive correct-route and negative mistaken-route
coefficients and positive held-out log-score improvement, with all three 95%
intervals excluding zero. The bootstrap for predictive score must state whether
models are refitted or intervals are conditional on fixed out-of-fold fits.

## Stop rules, stages and archive

The staged runner supports feasibility, calibration, analysis and report.
Feasibility inventories every represented epoch, visit, transition and candidate
assignment. Cohort and chronology gates precede neural calibration. Calibration
requires actual preceding-RUN readout artifacts, not placeholders or simulated
performance labelled as real. Analysis requires the passed calibration manifest.
Absent or failed prerequisites produce an inconclusive stop artifact, not an
empty successful analysis. Report mode does not decode or refit anything.

Execute on gpuserver6000 (OS hostname workstation2), detached from SSH, from a
committed isolated worktree. Record logs, checkpoints, exit status, protocol/input
hashes, environment and commit. Independently verify accounting and calculations.
Archive compact evidence in 2026-09-HippoReplayDynamics, without raw recordings
or manuscript claim changes. Secondary windows/post-correct/all-ripple analyses
cannot replace the primary. Wide intervals are not evidence of absence.
