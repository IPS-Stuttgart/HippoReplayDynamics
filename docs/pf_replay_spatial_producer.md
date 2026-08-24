# Causal PF replay spatial producer

This producer writes the predictor side of
`bayesian-ach.replay-spatial.v2`. It contains raw replay log emissions and
strictly pre-replay behavioral candidate fields. It never reads a later
behavioral outcome.

## Event schedule

The prior 160-event table is not reused because it was assembled downstream of
full-session decoder evidence. The claim-bearing cohort is reselected from the
source session:

1. enumerate ripple events whose peak is inside a RUN epoch;
2. require the predeclared minimum elapsed training time and number of completed
   historical routes;
3. rank by the event's raw LFP ripple power only, with event index as the
   deterministic tie breaker; and
4. retain the fixed top N per session.

No decoded trajectory, decoder evidence, next route, later well, or candidate
field participates in selection. The schedule and its complete parameter digest
are frozen in the manifest. Because top-N ranking is performed over the full
session, this is an offline, decoded-content-independent sampling rule. It
supports a conditional replay-content analysis; it does not support a causal
claim about online event incidence.

## Causal decoder and resolution

For an event beginning at `s`, the place-field encoder is refitted from
position samples and spikes at times no later than `nextafter(s, -inf)`.
Cells with no spike in that prefix are not introduced from the future.
The exported event audit records the requested cutoff and the largest position
and spike timestamps actually used.

Raw Poisson/NB spatial log likelihoods are evaluated during the replay interval.
Each row is max shifted and the removed offset is retained. Spatial coordinates
and point-spread values use centimeters.

Recovery resolution is empirical. A second prefix encoder is trained through
`s - holdout_window`, then decoded on moving RUN bins in the strictly
pre-event holdout interval. Ripple bins are removed and the 68th percentile
position error is frozen as `decoder_point_spread_cm`. Recovery later stresses
0.5, 1, and 2 times that value.

## Behavioral smoothing field

The latent state is compact destination-well identity, never the PF grid. For
each completed historical traversal:

- the prior is a pseudocount-regularized destination transition distribution
  learned only from still-earlier completed traversals;
- state-conditioned route templates are learned only from still-earlier RUN
  paths;
- filtering uses the predeclared spatial prefix of that traversal;
- the exact Hippo first-order trace uses identity well-state dynamics and
  smooths that prefix with the remaining path plus terminal well observation;
- the signed state difference is weighted by
  `KL(smoothed || filtered)`, age weighted, and projected through the
  pre-event route kernels.

The same event-prefix history builds online-surprise, posterior-content,
current-location, recency, prospective, and finite tabular TD-error fields.
All hyperparameters and per-event candidate-availability flags are frozen.
A negligible revision KL or constant candidate field makes that candidate
unavailable; it is not silently replaced by a zero-effect biological result.

## Files

- `replay_spatial_predictors.npz`: padded raw shifted emissions, offsets,
  masks, centimeter grid coordinates, empirical point spread, shared nuisance
  base, seven candidate fields, availability/cutoff arrays, posterior-derived
  well masses, and identifiers.
- `replay_spatial_manifest.json`: schema, exact producer commit from a clean
  worktree, verified canonical dataset-tree digest plus the dataset-manifest
  file SHA-256, route-segment and route-point input SHA-256s, ordered cohort and
  event-audit SHA-256s, trace/transition conventions, event-selection and
  hyperparameter digests, and predictor SHA-256.
- `replay_spatial_event_audit.csv`: selection rank/power, all causal cutoffs,
  training maxima, calibration support, field availability, and revision
  identifiability per event.
- `replay_spatial_export_summary.md`: compact run status.

Example:

```bash
python scripts/export_pf_replay_spatial_contract.py \
  --dataset-root /mnt/lexar4tb/datasets/pfeiffer-foster \
  --dataset-manifest /mnt/lexar4tb/datasets/pfeiffer-foster/dataset_manifest.json \
  --route-segments results/replay-behavior-route-primitives/replay_behavior_route_segments.csv \
  --route-points results/replay-behavior-route-primitives/replay_behavior_route_segment_points.csv \
  --dataset-sha256 LOCKED_DATASET_DIGEST \
  --output-dir results/pf-replay-spatial-contract
```

`--dataset-sha256` is checked against the canonical digest embedded in the
dataset manifest; supplying a label without the matching manifest is rejected.
The exporter also refuses a dirty or unavailable producer commit before writing
any claim-bearing artifact.

The output is predictor evidence, not a biological conclusion. Bayesian-ACh
must hash-check it, run LOAO/LOSO recovery, apply simultaneous
smoothing-versus-every-alternative rat contrasts, and abstain if any gate fails.
