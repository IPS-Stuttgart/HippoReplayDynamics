# Frozen Detector/Window Coverage Decoding

This follows the frozen all33 event-definition preparation and precedes its
decoding. Source definitions and thresholds are not retuned to replay outcomes.

## Cohort and Comparison

- Input: `replay-coverage-event-definitions-all33-20260905-v2`, successful
  preparation/reconstruction, including two explicitly unavailable Tanni ripple
  definitions. Preserve all33 sessions and unavailable/zero denominators.
- Decode every eligible core and peak-centered 200 ms window, once each. Do not
  select on spikes, posterior, continuity or model evidence at event level.
- Analyze MUA and native PF/new Tanni ripple definitions separately. Primary
  new Tanni peak threshold z3; z4/z5 use the same scored rows and frozen peaks.
- Report all eligible windows, both-detector overlaps, detector-only windows,
  and unknown ripple status separately. Never count overlap edges as new events.
  Overlap labels refer to the primary z3 definition; z4/z5 restrict ripple
  peaks without redefining which MUA windows overlapped primary ripple events.
- Main recording contrast: all RUN-QC cells versus half, three nested subsets,
  seed20260905, exact existing session-key convention. Reuse the original
  all33 population permutations and fixed maps/state support. Do not refit maps
  or redetect candidates after removing cells.

## Decoder and Measurements

Independent uniform-prior Poisson and count-conditioned multinomial decoding;
20 ms windows, 5 ms stride, existing common 8 cm RUN maps. MAP and posterior
mean. Compare no support filter to >=2 active cells AND >=3 spikes per window.
No HMM, IMM, momentum, path smoothing or temporal prior.

Continuity: earliest longest supported sequence with every jump <20 cm,
at least 10 frames and at least 40 cm net displacement. This is geometric
screening only, not shuffle-significant replay. Keep all failures in denominators.

Speed: adjacent non-overlapping 20 ms decoded windows, requiring all intervening
5 ms frames to be supported; never bridge missing frames. Report whole-event
and selected-core medians, with measurable-event/step counts. Selected-core
speed is conditional on the geometric selection and must not be presented as
an unbiased population estimate.

For the within-event full/half speed contrast, additionally require support in
BOTH populations on the same steps. Report median stepwise (half minus full)
speed differences and common measurable-step counts. Do not compare medians
from different surviving step sets as though they were paired speed errors.
Real replay truth remains unknown; the contrast is readout sensitivity, not bias.

Average recording replicates within session, sessions within animal, then
animals equally. Report animal-bootstrap CIs and every animal's effect. Detector
and window interactions pair session-level cell-removal effects before animal
aggregation. These compare different candidate sets, not randomized causal
effects of detector identity. Nine animals, not windows/splits, are the highest
biological replication level. No biological positivity threshold is a technical
gate. Keep the two datasets' detectors distinguished in every report.

## Trace Inspection

Before interpreting the LFP assay, for each Tanni animal and each available
eligible-core class (both, MUA-only, ripple-only), choose the event at the median
QC spike-count rank with stable session/source-ID tie breaking. Select using
these pre-decoding fields only, not trajectory results. Show fixed native
channel (lowest usable channel ID), locally bandpassed LFP, full-recording
aggregate envelope z, spike raster and both detector intervals. A single fixed
channel need not express a distributed ripple. Never shift LFP to align peaks.

These examples can expose gross problems but cannot prove artifact freedom or
hardware synchronization. PF raw LFP is unavailable here; do not fabricate PF
waveforms from its native event table.

## Verification and Remaining Scope

Freeze script/module/input hashes, commit and actual population IDs. Save
compact decoded paths, counts and window offsets for reconstruction. Independently
count every scoring window directly from sorted spikes and check sampled
posteriors with an analytic likelihood; reconstruct all summary metrics.

This does not replace the separate established replay-detection comparison:
PF/Foster edge trimming plus cell-ID/place-field-position shuffle significance,
with the actual shuffle budget and empirical null behavior reported. Calibration
transfer to new biological populations and the integrated paper pack remain
separate completion requirements. No speed-uniformity claim follows here.
