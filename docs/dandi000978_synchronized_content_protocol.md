# Synchronized RUN content calibration

Frozen before outcome computation, 2026-09-23. This is a follow-up to the joint
calibration, not a confirmatory sleep analysis. No new replay events or thresholds.

## Question

Does using CA1/PFC spikes from the same RUN moment improve the existing joint
route-content readout relative to independently sampled same-route donors?
The previous control could mismatch local position despite matching route.

## Fixed Design

Reuse the preceding 14 held-out RUN epochs, training-only unit masks and rate
maps, four directed route labels, and fixed CA1 donor windows from the archived
sparse calibration. Do not refit or choose windows by decoding performance.

Compare PFC donors:
1. synchronous: exactly the same 250 ms window as CA1;
2. position-matched: nearest observed 2D position on the same directed route
   in a different held-out trial (deterministic first-index tie-break);
3. independent: original independently sampled PFC donor on the same route.

Nearest position is only an approximate phase match; report its distance, with
no post-hoc cutoff. Different-trial matching cannot prove absence of shared
state, arousal or position-dependent coding. Same-window enhancement is not
evidence of communication or replay.

Readout remains PFC per-spike log score at the CA1 argmax route, centered over
four routes. Also retain the known true-route oracle. No soft-posterior tuning.
Zero-spike CA1 retains the historical argmax convention; report zero fractions.
Zero-spike PFC contributes zero centered score.

Use original native counts and exact sleep-count thinning without replacement.
Preserve the old CA1/PFC independent thinnings. New donor thinning uses seed
20260925, keyed by file, epoch and donor condition. Never add spikes, clip target
counts, search for higher-count donors or silently replace unavailable targets.

Report native all-anchor scores separately. Paired native and sleep-count
comparisons require every route and both regions to be supported in all three
donor conditions; thus they use identical blocks. Report condition-specific
and common coverage, including unsupported primary count profiles.

Summaries average routes/repeats within count target, then targets within epoch,
then epochs within animal. Two ZT2 files remain one animal. Raw count targets,
reused donors and thinning repeats do not create independent biological units.

## Conditional Sensitivity

Reuse the exact joint-calibration virtual-study algorithm: seed 20260924,
500 studies, 199 randomized PFC-route draws, alpha=.05. Primary target list is
the frozen 3 JS14/11 ZT2 list. If any profile lacks common support, label that
design unsupported. Additional sizes 10/25/50 draw unique supported targets.
Use identical draws across donor conditions/references for paired comparisons.
Only sleep-matched counts enter these virtual studies; native all-anchor scores
are a separate descriptive check. Do not turn simulated precision into animal
confidence intervals or label these rates actual sleep-study power.

No outcome is a sleep-content validation. If synchronization offers little gain,
close donor timing mismatch as an adequate explanation under this readout. If
it helps, retain the effect as a measurement result and require independent
RUN-to-NREM validation before any new biological test. Even improvement does
not justify loosening the frozen replay tier or dropping an animal.
