# Accuracy-upgrade building blocks

This patch adds opt-in utilities for improving the accuracy and calibration of
HippoReplayIMM analyses without changing the default benchmark path.

## What is included

- **Arena-valid latent states**: derive valid-state masks from occupancy and run
  exact diffusion on the restricted state space.
- **Topology-aware transitions**: build graph transitions over adjacent valid
  grid bins, reducing leakage into invalid/low-occupancy bins.
- **Continuous-time replay emissions**: build point-process emissions from
  inter-spike/no-spike intervals instead of fixed-width replay bins.
- **Replay gain and overdispersion calibration**: estimate per-cell replay gains
  and score negative-binomial or Gamma-Poisson predictive emissions.
- **Posterior calibration diagnostics**: summarize behavioral position-decoding
  calibration and model-probability confidence.
- **Nested validation helpers**: construct leave-one-rat and leave-one-session
  split definitions.
- **Empirical transition priors**: fit exact first-order transition matrices from
  run behavior and score them as replay models.
- **Forward/reverse/bidirectional hypotheses**: wrap existing replay scorers to
  test reverse and equal-prior bidirectional model mixtures.
- **Behavioral context labels**: add pre-well, post-well, and well-route context
  labels around replay events.
- **Position-quality flags**: robust median filtering plus high-speed jump flags
  for tracking diagnostics.
- **Tetrode-aware clusterless scaffolding**: infer mark-feature partitions by
  tetrode/channel naming conventions.
- **Exact small-grid second-order evidence**: compute exact momentum evidence on
  small grids for candidate-pruning gap audits.
- **Adaptive event windows**: generate padded replay-window candidates for
  window-sensitivity analysis.
- **Observation-model ensembles**: combine aligned emission tensors with a
  weighted product-of-experts likelihood.

## Quick diagnostic run

```bash
python scripts/accuracy_upgrade_report.py \
  --dataset-root data/DataSetFromPfeifferFoster \
  --session Rat1/Open1 \
  --events run \
  --max-events 25 \
  --output results/accuracy-upgrade-report
```

This writes compact CSV diagnostics for valid-state masks, empirical transition
priors, continuous-time emissions, behavioral context labels, position tracking
quality, and tetrode/mark partitions.

## Recommended validation order

1. Validate position tracking quality and behavioral position-decoding
   calibration.
2. Compare fixed-bin and continuous-time emissions on held-out behavior.
3. Tune emission gains/overdispersion using training sessions only.
4. Select dynamics hyperparameters with leave-rat-out or leave-session-out
   validation.
5. Use exact small-grid second-order checks to estimate candidate-pruning gaps.
6. Interpret final replay model probabilities with calibration diagnostics and
   evidence-margin summaries, not only winner counts.

The utilities are intentionally modular.  They should be treated as accuracy
infrastructure and validated before being used for scientific claims.
