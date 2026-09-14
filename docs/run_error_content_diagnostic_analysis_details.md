# RUN-risk report operational details

Written before inspecting the external scientific outcomes of the frozen
RUN-error diagnostic. This supplements, and does not change, its frozen protocol.

- The risk/error correlation gate is Pearson correlation of predicted risk with
  physical B position error (cm), computed within animal with equal-session
  weights. Correlation with log-normalized error is also reported, not substituted
  after inspection. Undefined correlations do not count as positive.
- Relative improvement gates use differences between equal-animal mean outcomes
  divided by the equal-animal baseline mean. Bootstrap intervals resample the
  five animals, not events, using seed 20260914 and 2,000 resamples. Session and
  simulation-draw weights are equal within each animal/source/split.
- The location-matched reference is computed separately per original session,
  split, source and draw, using all available errors within each known true tile,
  reweighted to the selected observations' true-tile proportions. It is a
  descriptive validation control, not a fitted predictor or replay truth label.
- All input metrics, source/draw groups and three frozen splits must be present.
  Missing errors fail input validation rather than being dropped from summaries.
- Split0 and the full risk predictor remain primary. Other policies and splits
  are sensitivity descriptions, never replacements for a failed primary gate.
- The independent audit reconstructs all source posterior metrics and known
  errors, including the exact simulated states and conditional count generator;
  it also checks actual raw RUN and endpoint spike counts.
