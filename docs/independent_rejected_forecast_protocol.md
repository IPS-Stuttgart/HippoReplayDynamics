# Independent Detection and Rejected-Event Forecasting

Protocol date: 2026-09-17. This is a prospective analysis specification for
already explored recordings, not independent biological confirmation.
Question: do trajectory-rejected events retain predictive temporal population
organization? Do not infer true replay, accurate physical speed, or IMM uniqueness.

## Frozen Cohort and Independence

Use the 33 cached full recordings from coverage inputs of 2026-09-05:
8 PF sessions / 4 animals and 25 Tanni sessions / 5 animals. Verify all cache
hashes; reuse only RUN-derived cell QC/maps, full spikes, position and supported
RUN intervals. Ignore all previously detected candidate intervals and scores.

Assign a fixed 20% (rounded, minimum 2) of RUN-QC cells in each session to
detection with deterministic seed 20260917 and session identity. These cells
never enter the encoding/prediction evaluation population. Of the remaining
cells use five reproducible 70/30 inference/evaluation partitions; a seeded
nested half of inference cells supplies the thinning sensitivity.
Evaluation cells may contribute to encoding from RUN and model training on
OTHER calibration events, but never to test-event detection, continuity
classification, latent inference, or forecast construction. Candidate windows
are identical across the five partitions.

## Detection

Detector-only population spike counts at 1 ms; Gaussian sigma 10 ms, truncated
at 4 SD. Valid immobility uses speed below 5 cm/s, Gaussian speed smoothing at
100 ms separately within tracking-supported RUN segments; no interpolation
over gaps longer than 100 ms. Standardize over valid immobile bins of the
session. Also require the unsmoothed RUN-encoding speed to stay below 5 cm/s
throughout each detector bin, preventing overlap with the >=10-cm/s RUN fit.
Boundaries are zero-z crossings restricted to valid immobile time;
peak must exceed z=3. Require duration 50-2000 ms and at least max(2, ceil(10%
of detector cells)) active detector cells. No evaluation-cell spike support
gate. Detected windows are disjoint. This is a transferred high-MUA detector
on a reserved subset, not an exact replication of the original PF release.

## Geometric Classification

Use the existing training_continuity implementation: independent flat-prior
Poisson MAP in overlapping 20-ms windows advanced every 5 ms; longest run
with jumps <20 cm, >=10 frames, >=40 cm endpoint displacement; supported
edges need >=2 inference spikes. RUN rate maps remain fixed, 8-cm grid,
including silence term. No new spatial masking, smoothing or parameter sweep.
The primary rejected-with-opportunity group has >=10 supported frames but
fails geometry. Secondary lost-with-thinning passes full inference geometry
and fails nested-half geometry. Primary group membership is split-specific.
This is the necessary geometric stage of the conventional criterion; rejected
events cannot pass its conjunction. Do not label geometry-pass events as
shuffle-validated replay: the two original 5000-shuffle tests are not run.
Record short/unsupported separately, never merge them into primary rejection.

## Cross-Event Fitting and Future Prediction

Five chronological event folds, excluding calibration events within 1 s of
any test event. Fit the existing regularized K=50 learned multinomial HMM,
2 restarts / 500 maximum iterations, on calibration-event non-detector cells.
Nonconvergence is a technical failure, not permission to change K or drops.
Full 20-ms bins only; discard and record the partial final bin.
Primary horizon is two bins: target center 40 ms after origin center, with
a full 20-ms unobserved interval between windows. Sensitivities 20 and 80 ms.
No target or intervening inference spikes enter a given forecast.
Held-out cell identities are scored conditional on held-out bin spike totals.
No posterior update from evaluation spikes. These are sums of marginal
predictive scores, not joint event evidence.

Primary model: learned HMM. Spatial diffusion with the pre-existing 60
cm/sqrt(s) kernel is a descriptive second representation, not an alternative
chosen after seeing the HMM outcome. No replay-rate gains fitted from test data.
Controls:
- stationary-occupancy and self-transition matched maximum-entropy kernel,
  sharing the original causal origin filter (legacy propagation contrast);
- that matched kernel using its OWN causal filter (stronger primary null);
- frozen origin posterior;
- no-history propagation from the same starting prior;
- global cell composition learned from other calibration events.
All have the same target cells and windows. Report independent-position
same-time decoding only as classification machinery, not future prediction.

## Endpoints, Denominators and Stopping Rules

Primary: learned-HMM minus independently filtered matched-null score per
held-out target spike for rejected-with-opportunity events at 40 ms.
Require positive contrasts versus global composition and no-history as
additional adequacy controls. Frozen-origin/shared-origin contrasts are
reported, not silently substituted for a failed primary.

Compute contrasts within event/split first. Median across qualifying splits,
then equal-event session means, equal-session animal means, equal-animal
dataset means. Also report unnormalized nats and per-target-bin sensitivity.
Zero-spike targets have zero identity score; event/splits with zero target
spikes have undefined per-spike scores and remain visible in denominators.
No qualifying events in a session/animal is insufficient support, not zero
effect or automatic pass. Record all 33 session statuses and every failure.
Exact animal bootstrap intervals are descriptive with only 4/5 animals;
report all per-animal effects, leave-one-animal-out means, and exact
two-sided sign tests. Conditional uncertainty does not account for the
broader exploratory program. Splits/bins/events are not independent animals.

Call dataset-level support only if the primary interval is positive, all
animals are represented and positive, and global/no-history comparisons pass.
Otherwise report mixed, inconclusive, or nonpositive as supported by data.
Full two-dataset replication requires both datasets; PF does not rescue Tanni.
The lost-with-thinning analysis separately scores full and reduced inference
populations against the same evaluation cells. Do not claim smaller-population
prediction from larger-population scores alone.
No thresholds, neurons, animals, models or lags will be chosen to obtain
a positive outcome. No physical-speed, memory-function or novelty claim.

## Verification

Freeze code/protocol in Git before real scoring. Save input/output hashes,
cell IDs, detections, count arrays, folds, model fits, geometric paths,
scores and technical statuses. Synthetic tests must include:
disjoint deterministic partitions; detection invariant to evaluation spikes;
no events across tracking gaps/movement; held-out/future-spike forecast
invariance; directional synthetic positive and IID negative; matched-null
constraints; correct split-before-event aggregation; zero/missing failures.
An independent verifier must reconstruct detections, sample geometries and
forecast scores, and all aggregate primary estimates from saved arrays.
Run detached on gpuserver6000 with logs and terminal status so SSH loss
does not interrupt it. Preserve failed runs, never overwrite evidence.
