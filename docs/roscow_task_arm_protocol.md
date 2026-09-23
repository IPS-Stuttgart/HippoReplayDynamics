# Roscow held-out task-arm feasibility protocol

Frozen before inspecting neural readout results, 2026-09-23.
This is a prerequisite for a biological content question, not a replay result.

## Endpoint and input

Predict the actual chosen arm (1/2/3) from CA1 spikes in [-2.0, -0.25) seconds
relative to the published reward-arrival timestamp. This is a categorical
pre-arrival readout, not a continuous-position decoder. The fixed interval avoids
reward consumption/outcome activity but does not guarantee fixed physical position
or remove expectation, speed and movement covariates.

Use only metadata-pass sessions and legitimate choices from the pinned Roscow
preflight. Require at least five CA1 units and three trials per arm. Keep an
explicit exclusion row for every session. Do not recover flagged timestamps or
pool cells across sessions. Neural cohort: Quirinius, Severus, Trevor.
The trial-support screen retains at most 11/4/2 sessions respectively after the
cell-count criterion. This screen was based on labels/counts, not neural accuracy.

## Fixed estimator

- Leave one entire trial out; training rates, including the shrinkage prior,
  use only the remaining trials. All CA1 units are retained, no activity-based
  test-cell selection.
- Per-arm rate estimates use one second of shrinkage toward the training-only
  pooled rate and a numerical floor of 1e-10 Hz. Flat arm prior.
- Use the existing Poisson, composition-only, and total-count-only posterior
  helper from validate_dandi000978_run_readouts.py.
- Composition-only conditions on total population count; it is the mandatory
  check against classifying the arm merely from firing intensity.
- Report balanced accuracy and mean true-arm log probability minus log(1/3).
  Accuracy ties receive fractional credit across equally likely maxima.
- Score each trial once per readout, not overlapping pseudo-independent bins.

## Controls and aggregation

Refit the complete leave-one-trial-out pipeline for every circular shift of the
chronological arm-label sequence (including zero, which is the real alignment).
This preserves label counts/order and spike drift. Repetitive action sequences
can make this conservative: a failed shift control does not prove no arm signal.
Keep these limitations visible rather than replacing a failed null post hoc.

Use equal session weights within each animal. Build 10,000 aggregate null draws
by independently sampling one of all shifts in each session (identity included).
Seed 20260923, deterministic per animal/readout. Do not use the number of neurons
or trials as replication. Three animals remain only three neural subjects.

Feasibility continuation screen: at least two eligible sessions in each animal,
and observed mean session-balanced accuracy exceeds both 1/3 and the animal's
95th shift-null percentile for Poisson AND composition-only. Count-only is a
diagnostic, not a model that can substitute for composition. The screen is not a
biological significance claim, and passing does not authorize a rest-replay claim.

No replay, PRE/POST effect, reward-outcome effect or high/low-probability neural
contrast is scored. Trial-cycle/sequence anchors and a biological protocol still
need validation even if this screen passes.

## Prior-art boundary

Roscow et al. already report prediction-related reactivation:
https://doi.org/10.1038/s41467-025-65354-2 . Merely finding reward-biased activity
would duplicate that result. Mattar and Daw already predict prioritized reverse
backups and asymmetric reward/omission responses:
https://doi.org/10.1038/s41593-018-0232-z . Generic surprise-gated backward replay
is not an established novelty claim for our project. A potential empirical
extension must distinguish ordered represented content from coactivity and gain,
with independent task anchors; its novelty remains unproven.
