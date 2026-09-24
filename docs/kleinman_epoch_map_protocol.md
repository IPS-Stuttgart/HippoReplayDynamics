# Kleinman reward-epoch RUN map transfer

Frozen before examining these outcomes, 2026-09-24. This is a prerequisite
measurement audit, not a replay, surprise, dopamine or reward-effect result.

## Question and cohort

Can a RUN encoding map trained during one reward epoch decode real behavior
during another? Use all 127 previously RUN-qualified sessions, preserve all 135
source sessions in the denominator, and flag the previously frozen 14-session
extent-eligible subset without changing its membership. Do not select by replay.
The baseline / increased-reward / return-to-baseline epochs follow the verified
native epoch_change alignment. Transitions and incomplete traversals are excluded.

## Paired design

Within each epoch, group consecutive complete opposite-end traversals in pairs.
Discard an unmatched final traversal. For every ordered source/target epoch pair
and five fixed splits, sample an equal number of complete two-traversal groups
from each epoch for training: min(floor(n_source/2), floor(n_target/2)). Require
at least two groups per training map. Seed is 20260924 and session identities
are SHA256-derived; no retry or favorable seed selection. All remaining target
groups supply held-out test windows. No test spikes determine maps or cells.

Use the existing 2-cm bins, 4-cm rate smoothing, RUN speed/clock filters, 250-ms
disjoint test windows and interior exclusion (20 cm from visit thresholds).
Compare same-epoch and cross-epoch maps on exactly the same target windows,
common training-qualified cells and common supported spatial/direction states.
Require five common cells and six spikes per test window. Record separately:
training groups/cells, original/common support, truth outside common support,
tracking/behavior/spike exclusions and all failed/empty comparisons. Common
support is a conditional estimand, not evidence that lost states are harmless.

Decode with both the existing Poisson likelihood and a count-conditioned
cell-composition likelihood. Both have uniform spatial/direction priors and no
HMM or temporal smoothing. The conditional arm removes global-rate mismatch,
not cell-specific remapping. Primary paired metric: cross minus within absolute
posterior-mean position error. Also direction accuracy, posterior entropy, MAP
error and true-state log score above uniform when supported. Epoch-map population
Hellinger distances are descriptive, not a remapping significance test.

## Aggregation and boundaries

Retain every held-out window and split. Compute mean errors within each split,
median across repeated splits per ordered epoch pair, then median across the
six ordered pairs per session. Animal summaries give sessions equal weight.
Do not pool windows or repeated splits as independent animals. Report all
sessions and the frozen extent subset separately. Preserve the earlier RUN
35-cm / 0.60 direction screens as descriptive transfer-QC flags, not a newly
calibrated stability/equivalence test. No absence-of-significance equals stability.

Reward epoch is confounded with time, training occupancy and behavior; matching
lap counts cannot establish reward-induced remapping. Large transfer degradation
would invalidate assuming a single session map is stable for reward-content
comparisons. Good transfer alone does not establish adequate short replay
decoding or validate monotone templates. No real replay event is fitted here.

Technical checks: all source sessions accounted; all 127 RUN-pass attempted;
30 ordered-pair/split records per attempted session; no unexpected exceptions;
source hashes frozen; train/test groups disjoint; matched training group counts;
both likelihood arms compare identical observations/support/cells. Technical
completion and scientific map adequacy must remain distinct.
