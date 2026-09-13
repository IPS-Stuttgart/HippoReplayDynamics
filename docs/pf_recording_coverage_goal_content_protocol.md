# PF Recording Coverage and Represented Destinations

Frozen 2026-09-13, after metadata QC and before inspecting decoded content effects.

## Question and Scope

Does removing recorded cells change represented destinations for unchanged events,
or change the population of events that enters a content summary? This is a
recording-intervention diagnostic, not a replication or refutation of the PF goal
planning result. The full-cell decode is a reference, not neural ground truth.

Use all eight PF sessions/four rats and the source-high-MUA detected cores in the
completed 33-session two-shuffle benchmark. Reuse its exact three half-cell
subsets, full-cell population, masks, maps, and original-order decisions (edge-only,
10 frames, both shuffle p-values < .02). Do not detect or select new events.

## Metadata

The release documents well-fill times but not coordinates or explicit Home IDs.
Infer Home as the unique ID present in every unequal consecutive well pair.
Estimate coordinates from median position during the second before the NEXT fill
for visits whose current fill ID is Home. Require >=5 visits and p75 radial
deviation <=15 cm; retain all visit estimates and outliers in the audit.
Home is an inferred task landmark, not verified author ground-truth annotation.
All eight sessions satisfy these gates (p75 3.6-5.6 cm). Rat2/Open1 has one large
outlier; use the median, do not selectively discard visits.

## Decoding and Endpoints

Independent flat-prior Poisson decoding, 20 ms windows/5 ms step, frozen common
8 cm maps. No HMM, IMM, momentum, temporal smoothing, or replay-derived prior.
For the all-candidate readout, fix the endpoint to the last full-cell frame with
>=2 spikes; do not move it when cells are removed. Report empty support explicitly.
For trajectory-selected analyses, fix the anchor to the last frame of the
full-cell earliest-longest continuous segment, even if that segment fails the
duration/displacement test. The same timestamp is used by all populations.

Readouts: endpoint posterior mean shift (cm), MAP shift (cm), posterior total
variation, posterior mass within 20 cm of Home, and binary MAP-in-Home label.
Home mass above the uniform-support area fraction is descriptive, NOT a
behavior/distance/occupancy-matched goal preference test. A Home-vs-other result
does not measure prospective planning. Report active-fill Home intervals only
where the entire event is bracketed by consecutive recorded fills; no last-fill
extrapolation. Primary neighborhood is 20 cm; 10 and 30 cm are fixed sensitivities.

## Selection Decomposition

For each half subset, using 20 cm Home mass and the frozen two-shuffle labels:

1. Decoding: half minus full on the SAME full-accepted events and full anchors.
2. Selection composition: half decode on half-accepted minus full-accepted events,
   keeping full anchors in both groups.
3. Endpoint timing: half own-segment endpoint minus full-segment endpoint, on the
   SAME half-accepted events.

Their sum equals the change from the full accepted/own-anchor summary to the half
accepted/own-anchor summary. Do not interpret the reselected contrast as a
same-event decoding effect. Report all denominators, missing groups, and label
agreement. Selection on the full population is not independent validation.

Aggregate replicates within session, sessions within rat, then equal rats.
Use 5,000 resamples of four rats for descriptive cluster-bootstrap intervals;
four independent animals remain a strong precision limitation. No event-pooled
significance tests or biological pass gates.

## Known-Content Simulation

For each session simulate 200 straight, 200 ms, 400 cm/s paths, half ending within
10 cm of Home and half outside 30 cm. Use the real session maps as Poisson rates
on nearest supported spatial bins, with 5 ms base counts and 20/5 ms decoding.
Require path samples within 10 cm of supported states. Freeze seed 20260913.
Use identical spike realizations before/after each recorded-cell removal. Report
endpoint error, Home sensitivity/specificity, and predicted-vs-true Home fraction
on all simulations (no continuity selection). This is a matched-model recovery
control, not proof that real events follow the simulator. It isolates whether
known content can become less recoverable without any content change. It does
not simulate variable goal prevalence, population correlations, or real replay.

## Stop Rule

Finish this bounded PF diagnostic and report direction, instability, and null
results honestly. No new dataset, new replay-model campaign, or goal-planning
claim follows automatically. Preserve source hashes, exact IDs, code commit,
protocol, per-session checkpoints, and independent reconstruction checks.
