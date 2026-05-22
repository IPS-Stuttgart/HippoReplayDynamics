# Paperworthy benchmark patch protocol

This protocol makes the replay-dynamics benchmark more defensible without
turning PyRecEst or candidate-pruned momentum into unvalidated defaults.

## 1. Use certified-vs-exact recovery for candidate-pruned momentum/IMM

Candidate-pruned second-order models report lower-bound evidences unless their
support is the full valid grid.  The parameter selector therefore blocks strict
exact-comparable recovery gates for candidate-pruned momentum/IMM by default.
Use strict recovery only as an exploratory legacy diagnostic:

```bash
python scripts/select_state_space_parameters.py \
  --evidence results/state-space-evidence-sweep-summary \
  --recovery results/simulation-recovery-sweep-summary \
  --output results/state-space-parameter-selection \
  --recovery-gate-metric certified-vs-exact
```

The override is intentionally explicit:

```bash
python scripts/select_state_space_parameters.py ... --force-strict-recovery-gate
```

## 2. Triage failed momentum recovery before changing models

Run the triage script on every recovery sweep summary:

```bash
python scripts/triage_momentum_recovery.py \
  --scores results/simulation-recovery-sweep-summary/simulation_recovery_sweep_event_scores.csv \
  --output results/momentum-recovery-triage
```

The output separates strict exact recovery, lower-bound-certified recovery,
candidate-support loss, oracle-support recovery, exact non-recovery, and
nondecisive lower bounds.

## 3. Split synthetic true dynamics from scoring parameters

Simulation recovery can now hold the synthetic world fixed while sweeping
scoring parameters:

```bash
hipporeplayimm simulate-recovery data/DataSetFromPfeifferFoster \
  --session Rat1/Open1 \
  --true-models "diffusion momentum" \
  --models "sorted-spike-state-space-diffusion sorted-spike-state-space-momentum" \
  --true-state-space-diffusion-sigma-cm-sqrt-s 85 \
  --true-state-space-momentum-sigma-cm-sqrt-s 85 \
  --true-state-space-momentum-initial-sigma-cm-sqrt-s 85 \
  --true-state-space-momentum-velocity-decay-tau-s 0.060 \
  --state-space-diffusion-sigma-cm-sqrt-s 60 \
  --state-space-momentum-sigma-cm-sqrt-s 110 \
  --output results/simulation-recovery-fixed-world
```

Every recovery row records both `true_state_space_*` and
`scoring_state_space_*` columns.

## 4. Treat incomplete grids as exploratory

The simulation-recovery sweep aggregate now joins completed artifacts to the
planned matrix and writes planned/completed/missing/failed config tables.  By
default, missing planned configurations fail the aggregate ranking.  Set
`allow_incomplete_grid=true` only for exploratory runs.

## 5. Final-paper framing

Use the triage output and support-aware selector with the existing paper-claim
tables.  The defensible paper claim is a reproducible, lower-bound-aware replay
benchmark with trajectory-vs-static evidence, momentum-vs-diffusion paired
effects, and session/rat heterogeneity.  PyRecEst remains a supplemental
multi-seed validation track unless it wins under robust aggregation.
