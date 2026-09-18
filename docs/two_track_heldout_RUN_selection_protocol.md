# Behavior-anchored held-out RUN selection calibration

## Purpose and boundary

The known-label simulation distortion survives a count-conditioned likelihood,
but model-generated spikes still limit the result. Test sampling/selection on
real RUN spikes with context identity supplied by behavior. This is a measurement
positive control, not newly discovered replay or a SleepPOST bias confirmation.
Physical track occupancy labels the behavioral context; it does not label every
spike's represented content. Do not transfer an effect size to sleep replay.

Freeze both existing strict-primary sessions, RAT3_SESS2 and RAT5_SESS2. Their
earlier RUN-quality-based inclusion is not an independent replication cohort.
Do not expand or choose the cohort after these results.

## Training and observations

Use the original regular position clock and real spikes. Partition the recording
into 10-second blocks from the first position timestamp; block modulo five is
the outer fold. Fit maps and apply cell QC only to the other folds, excluding
one-second neighbours of all held-out blocks. Do not reuse full-RUN cell QC or
full-RUN maps to choose the test populations. Retain the established 10-cm map,
occupancy and RUN-speed (5 < v < 50 cm/s) rules.

Behavior-only candidate windows are nonoverlapping one-second intervals on that
clock grid, entirely inside one test block, entirely within a single track and
the RUN speed mask. Do not require neural activity, an estimated path, successful
decoding or a minimum observed displacement to choose them. Record their actual
displacement/path span and behavior-bin positions. Within each fold select the
same number per track, capped at 20, by a fixed identity hash. Any shortfall is
reported. No replacement based on spikes or decoder outcomes. The intended
maximum is 200 windows per session, exactly balanced between contexts.

Each window contains ten nonoverlapping 100-ms real-count bins. To reuse the
20-ms sequence code, multiply floored RUN rates by five. This is mathematically
equivalent to independent Poisson decoding at the actual 100-ms exposure:
duration * rate is unchanged and the extra log(5) count term is position-constant.
It is an exposure conversion, not evidence that behavior was replayed five
times faster, and adds no new spikes or smoothing. Retain actual one-second
duration and actual clock timestamps throughout metadata and interpretation.

Within each fold, use five seeded disjoint inference/evaluation partitions of
training-QC-passing units; reserve the standard detector fraction without using
it to select RUN windows. Keep evaluation cells fixed while comparing full
inference with five fixed random half subsets. No full-RUN cell identities from
the replay bank enter this split design. Record all identities and training masks.

Use the original five-active-cell/five-nonempty-bin opportunity rule, 499
whole-bin and 499 per-cell field-shift nulls, and p < .025 for both per track.
Primary decoder: Poisson with exposure conversion. Sensitivity: conditional
population-identity likelihood. Whole-bin-shuffled copies preserve counts and
population snapshots; they are a temporal negative control. No observation is
generated from an encoding model. No real replay event is rescored.

## Independent readout and aggregation

Compute fixed evaluation-cell context support using the count-conditional
likelihood and 199 within-RUN-rate-group cell-identity permutations. Orient it
to the behavior-defined track, never the inferred track. Readouts stay unchanged
across coverage subsets, sequence likelihoods and time shuffles. Report finite
support and context classification accuracy for all windows and for lost windows.

Average repeats/splits within each original RUN window before aggregation.
Keep groups full, half, retained, lost, gained and all, separating losses from
failed activity support versus failed sequence tests. Known context proportions
come from behavioral labels, never posterior averages. Report decoded labels
and their errors separately.

Primary comparison: half-minus-full known track-2 fraction among selected RUN
windows, with two-sided uncertainty. Also report full/half acceptance by context,
loss given full detection and independent true-context support among losses.
Use 2,000 paired block bootstraps stratified by track and fold. Resample original
10-second test blocks; normalize weights to each stratum's original window count
to preserve the balanced context design. Keep every split/repeat/likelihood of
a window together. Require at least two sampled blocks per stratum and >=95%
finite draws for intervals. These are within-session measurement intervals,
not animal-population confidence bounds.

## Interpretation

Loss of externally anchored context after cell thinning would strengthen the
measurement result using real spikes rather than model draws. A selective
context shift would show bias in this designed behavioral calibration. Neither
establishes selective loss of biological sleep replay. A null remains a null;
do not change duration, balancing, cell QC or sequence thresholds after reading
the outputs. Preserve all previous negative/inconclusive results.
