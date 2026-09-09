# Matched 1-ms / 20-ms Clock Recovery

Freeze before observing scores. Simulation only, still a known-path oracle.
Re-use the 1,056 geometric paths, original maps, clocks and count-profile totals
from the independently verified `literal-clock-recovery-all33-20260909` run.
New observations, not a rescore of real replay or an independent replication of
the source animals. The previous 20-ms screen achieved 61.56% PF and 56.91% Tanni
classification; its 80% practical event-level target failed in both.

For each path/generator (physical/neural)/emission condition (exact/gain drift),
draw ten new observations under a new `clock_timing_v1|20260909` RNG namespace.
Subdivide each original 20-ms bin into twenty 1-ms subbins. Integrate generating
rates in each subbin with 16-point Gauss-Legendre quadrature. For each parent
bin with fixed native total N, draw N labels jointly over (subbin, cell), with
probability proportional to integrated generating rate. This is a discretized
marked inhomogeneous Poisson observation conditional on the parent total; not
exact continuous spike timing. Gain drift uses the frozen per-cell factors;
the scorer is not told them. True clocks remain the original RUN-defined clocks.

Use EXACTLY the same sampled counts for four paired scores:

1. `coarse_identity`: sum subbins first; likelihood of cell identities at 20 ms.
2. `fine_identity`: likelihood of cell identities conditional on each 1-ms total.
3. `fine_timing`: likelihood of subbin totals conditional on the 20-ms total.
4. `fine_joint`: joint likelihood of (subbin, cell) labels.

Fine-joint score equals fine-identity plus fine-timing, up to common observation
coefficients that cancel in every clock contrast. For exact emissions, expected
fine-joint true-vs-wrong KL must be no smaller than coarse-identity KL (data
processing). Fine-identity alone does not have that inequality because it
conditions on the finer totals and removes timing-envelope information.

Score both known clocks without estimating the true path from data. The same
oracle privilege applies to all arms. No 1-ms point-position decoding, temporal
smoothing, event selection or biological inference. The experiment isolates
timing information, not whether 1-ms point estimates make a cleaner trajectory.

Report accuracy per path after averaging observation repeats and both true
clocks, then recordings within animal and animals within dataset. Primary paired
contrast: fine-identity accuracy minus coarse-identity accuracy. Separately
report fine-joint and timing-only. Practical availability target remains >=.80
accuracy in each dataset and >.50 in every source animal, applied separately to
each arm/condition. A five-percentage-point paired improvement is a descriptive
timing-benefit screen, not statistical significance or a new biological gate.
Do not infer population-level impossibility from event-level failure.

Save every sampled fine count array and generating fine-rate table. Verify fresh
RNG draws, conservation of native parent totals, conditional/marginal probability
normalization, score decomposition, independent scipy integration at 32 nodes,
all raw scores and hierarchical summaries. Numerical readiness requires max
probability discrepancy <=1e-4 and nonnegative exact expected data-processing
excess within 1e-9. Freeze code and retain manifests/hashes.

Possible conclusions: timing discarded useful information; timing mainly adds
population-envelope information; or the tested clocks remain difficult to
separate even at finer timing. None proves speed uniformity, a neural-sheet
mechanism, or the effectiveness of fuzzy continuity processing. This is a
measurement test in the continuing publication search, not the publication goal.
