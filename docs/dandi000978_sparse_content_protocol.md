# DANDI000978 sparse-count RUN calibration

Frozen before running this calibration; seed 20260923. The preceding sleep
coverage/content pilot is unchanged and did not pass its two-control endpoint.

## Question

Can independent PFC route composition be read out at the spike counts of the
1,354 frozen sleep candidates when the target is known held-out RUN behavior?
CA1 is a separate regional positive control. This is a measurement calibration,
not a new replay test, not sleep ground truth, and not evidence of absence.

## Design

- Use the three source-supported files (JS14, two ZT2 files; two animals).
- For each of the 14 RUN epochs preceding the frozen candidates, hold out that
  entire epoch. Train on other RUN epochs in the same file, reusing the existing
  movement bins, training-only unit inclusion and route-rate estimation.
- Calibration can use later RUN epochs, unlike the sleep pilot. It is not a
  prospective replay decoder or an exact replication of its single-RUN maps.
- Four directed center/side routes; each needs three training trials. Include
  units with at least 20 training RUN spikes. Minimum 20 CA1 or five PFC units.
- Recount each frozen sleep window using these training-selected units. Use
  only the total spike count, never its inferred route or cortical score.
- For every sleep count, sample one held-out RUN trial per true route, uniformly,
  then one eligible 250 ms movement window uniformly within that trial. Repeat
  ten times. Also retain the entire selected trial as an optimistic control.
- For each anchor, sample exactly the sleep count without replacement from the
  actual spikes (multivariate hypergeometric). No added or duplicated spikes.
  Insufficient anchors remain unmatched; do not search for a more favorable
  window or clip the target count. Zero counts remain uninformative.
- Decode composition with a uniform four-route prior. Record accuracy with
  fractional credit for ties, and per-spike true-route score centered over the
  four route scores, identical to the sleep readout's metric.
- Show original scores for the same matched anchors, so thinning has a paired
  native-count baseline. Whole-trial results are not short-window performance.
- Null: 199 permutations of behavioral route labels among held-out trials,
  applied consistently to all uses of each donor trial. No independent shuffle
  of repeated windows or thinning draws. No naive repeated-draw standard errors.
- Summarize within each RUN holdout, then equally across RUN epochs in each
  animal. ZT2 remains one animal. Match coverage is reported separately.
- PFC short-window calibration is detectable only if both animals exceed their
  trial-label null p95 for centered content and accuracy, with at least 80%
  matched anchors and all expected regional folds scored. This does not validate
  RUN-to-sleep transfer or reverse the sleep pilot's result.

## Limits And Outputs

Matched anchors condition on sufficient RUN spikes, so sparse-count results
apply only to that reported subset. Route identity can reflect position and
direction, not abstract cortical content. Independent per-region count matching
does not test CA1/PFC coordination. RUN labels are known; sleep labels are not.

Write fold metrics, animal summaries, count distributions, matching coverage,
trial-label null summaries, gates, manifest, compact figure and markdown report.
Keep anchor identities/count arrays privately for independent verification.
Do not modify frozen sleep events, continuity rules, or replay claims.
