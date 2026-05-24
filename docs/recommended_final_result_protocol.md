# Recommended final result protocol

This protocol turns the result-improvement utilities into a conservative final
analysis path.  It is intentionally stricter than the smoke benchmarks: final
claims should survive observation-model validation, candidate-support checks,
null controls, repeated splits/seeds, and evidence-margin reporting.

## 1. Validate the observation model first

Use behavioral position decoding and synthetic recovery before selecting replay
dynamics.  Do not tune replay dynamics on real-event evidence until the sorted
spike or clusterless observation model has passed an independent validation
gate.

Recommended sorted-spike sweep axes:

```text
bin_size_cm: 4, 6
smoothing_sigma_bins: 1.5, 2.0
min_occupancy_s: 0.01, 0.02, 0.05
rate_floor_hz: 1e-5, 1e-4
time_bin_ms: 2, 3, 5
spike_rate_scale: 0.5, 1.0, 2.0
emission_likelihood_temperature: 0.5, 1.0, 2.0
negative_binomial_overdispersion or dispersion: compact Poisson/NB grid
```

Keep settings with acceptable behavioral posterior-mean/MAP errors and synthetic
recovery.  Treat replay-only gains or overdispersion as sensitivity analyses
unless they are selected without looking at real-event model wins.

## 2. Use exact and truncated evidences in separate tables

Exact full-grid models can be normalized into event-level model probabilities.
Candidate-pruned momentum/IMM rows are lower bounds; they should be reported in
the lower-bound audit tables unless their candidate-support quality is known to
be good and the interpretation is explicitly lower-bound based.

The benchmark writer now emits:

```text
event_model_evidence_with_margins.csv
result_quality_gate_summary.csv
result_quality_model_summary.csv
result_quality_event_summary.csv
exact_model_evidence_summary.csv
truncated_lower_bound_summary.csv
evidence_support_counts.csv
```

Use the exact summary for posterior model probabilities and the truncated
summary for candidate-support lower-bound diagnostics.

## 3. Run candidate-support sensitivity before interpreting IMM failures

Second-order momentum/IMM evidence depends on candidate support.  A compact
support audit should include:

```text
state_space_momentum_candidate_top_k: 128, 256, 512
state_space_momentum_predicted_candidate_top_k: 0, 8, 16, 32
state_space_momentum_candidate_mass_threshold: empty, 0.99, 0.999, 0.9999
```

Headline tables should flag or omit rows whose support quality is poor or
unknown.  Inspect `candidate_min_log_mass`, `candidate_support_quality`, and the
result-quality gate summary.

## 4. Parameterize IMM switching in physical time

Prefer `--state-space-imm-switch-tau-s` over a per-bin stickiness when comparing
different replay bin widths.  A practical initial grid is:

```text
state_space_imm_switch_tau_s: 0.020, 0.040, 0.060, 0.100, 0.200
```

For 3 ms bins, `tau_s = 0.060` gives an effective per-bin stickiness near 0.951,
close to the legacy default while remaining bin-width invariant.

## 5. Include directional and goal-conditioned hypotheses

Use exact goal-conditioned models and forward/reverse/bidirectional wrappers
when endpoint or route interpretation matters.  A conservative default model
set is:

```text
sorted-spike-state-space-stationary
sorted-spike-state-space-diffusion
sorted-spike-state-space-fragmented
sorted-spike-state-space-first-order-imm
sorted-spike-state-space-momentum-exact-sparse
sorted-spike-state-space-momentum
sorted-spike-state-space-velocity-momentum
sorted-spike-state-space-momentum-bidirectional
sorted-spike-state-space-imm
sorted-spike-state-space-goal
sorted-spike-state-space-goal-bidirectional
clusterless-state-space-stationary
clusterless-state-space-diffusion
clusterless-state-space-momentum
clusterless-state-space-velocity-momentum
clusterless-state-space-fragmented
clusterless-state-space-imm
```

Report endpoint accuracy, true-well posterior/rank, and evidence margins; do not
reduce interpretation to winner counts alone.

## 6. Run null controls and event-window sensitivity

For the improved benchmark, add spatial-bin permutation controls:

```bash
--null-shuffles 25
```

For final claims, pair this with wrong-map controls, event-window padding
sensitivity, stable-cell/place-field-quality filters, circular time shifts, and
posterior-predictive spike-count checks.

## 7. Repeat cell splits and summarize hierarchically

Held-out cell analyses should be repeated over multiple random seeds and
stratified splits.  Use session- and rat-level summaries plus hierarchical
bootstrap intervals; avoid treating pooled ripple events as independent final
replicates.

Recommended minimum:

```text
random_seeds: 1..10
cell_split_strategy: random, mean-rate, peak-rate
uncertainty: session/rat hierarchical bootstrap
```

## 8. Run the momentum-recovery ladder before selecting second-order parameters

Candidate-pruned pairwise momentum/IMM rows are useful lower-bound diagnostics,
but a failed strict recovery table is not enough to tell whether momentum is
unidentifiable, unsupported by the candidate beam, or excluded by the exact-
evidence gate.  Before tuning real replay evidence, run the ladder:

```bash
python scripts/run_momentum_recovery_ladder.py \
  --dataset-root data/DataSetFromPfeifferFoster \
  --session Rat1/Open1 \
  --events run:0-25 \
  --events-per-model 25 \
  --output results/momentum-recovery-ladder
```

Interpret the four tiers as a diagnostic progression: full-grid pairwise
momentum, exact finite-velocity/displacement momentum, oracle candidate support,
and native candidate support.  A paper-level second-order claim should use exact
finite-velocity/displacement evidence for the headline model comparison and keep
candidate-pruned pairwise momentum/IMM as lower-bound support diagnostics.

## 9. Treat clusterless and PyRecEst results as separate validation tracks

Clusterless local-KDE models should pass mark-drift and mark-likelihood
sensitivity checks before being mixed into the sorted-spike narrative.  PyRecEst
particle results should be reported as seed/particle-count/proposal aggregated
Pareto summaries rather than single-seed rankings.

## 10. Diagnose synthetic recovery before selecting parameters

Use exact sparse momentum for the headline exact/comparable momentum row.  For
candidate-pruned momentum/IMM rows, strict recovery is exact-comparable only and
can exclude the expected model from winning.  Always inspect the diagnostic view
before interpreting a candidate-pruned momentum-recovery failure:

```bash
python scripts/diagnose_simulation_recovery.py \
  --scores results/simulation-recovery-sweep-summary/simulation_recovery_sweep_event_scores.csv \
  --output results/simulation-recovery-diagnostics
```

Use `simulation_recovery_diagnostic_summary.csv` to separate strict-gate
exclusion, certified lower-bound recovery, candidate-support loss, and genuinely
nondecisive lower bounds.

## 11. Build a single paper-pack artifact

After the final evidence, recovery, null-control, and parameter-selection runs,
collect them into one auditable directory rather than copying tables by hand:

```bash
python scripts/build_paper_pack.py \
  --scores results/model-evidence-all-sessions/event_scores.csv \
  --simulation-recovery-scores results/simulation-recovery-sweep-summary \
  --primary-model sorted-spike-state-space-momentum-exact-sparse \
  --baseline-model sorted-spike-state-space-diffusion \
  --output results/paper-pack
```
