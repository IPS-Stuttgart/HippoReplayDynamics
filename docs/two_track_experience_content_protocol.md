# Recording coverage and experience-specific content

## Question and scope

Does thinning the neurons available for replay classification selectively hide
independently supported track-specific content and change the selected content
distribution? Loss of a geometric label alone is not the desired result.
This analysis must allow a null or inconclusive answer.

Dataset: Tirole et al., Dryad DOI 10.5061/dryad.ksn02v76h, pinned version 238435.
Author code: bendor-lab/replay_detection_cross_validation, commit
754a9efe01eae551a8e4766f1137c38b1365b2aa. These are independent public sources;
Daniel Bush is the collaborator, not an author of these sources.

Tirole et al. (2022), eLife 79031, already established complementary rate and
place representations. Takigawa et al. (2024), eLife 85635, already compared
sequence fidelity with sequenceless track discrimination. Neither concept is
claimed as new here. The intervention is neuronal subsampling with an independent,
fixed evaluation population and a measured selection-induced content shift.

## Preflight frozen before RUN calibration

- Verify the public SHA256 and size of every input. Keep both names in duplicate
  groups but exclude every ambiguous identity from the primary animal analysis.
  Never select one duplicate arbitrarily. Metadata inspection identified matching
  spike AND position files for RAT1_SESS2 and RAT4_SESS1.
- RAT2_SESS1 contains four track entries. Do not silently choose or pool tracks.
  Require an external track/epoch mapping or quarantine it in the primary run.
- Read linear coordinates as centimetres and track lengths as metres; use the
  regular position.t grid, not the different raw-camera position.clean grid.
- Author-aligned maps: 10 cm bins, 5 < RUN speed < 50 cm/s, occupancy-normalized
  unit spike counts; forward/backward [0.5,0.5] rate smoothing. Eligible units:
  smoothed peak >=0.5 Hz, raw peak >=1 Hz, track-epoch mean <=5 Hz, spatial
  information >0 on BOTH tracks. Record union counts as a sensitivity denominator.
- No waveform or per-unit region table is supplied in this subset. Do not infer
  anatomical labels from tetrode numbers. State this limitation explicitly.
- Require >=20 common units and >=80% of bins with >=0.2 s RUN occupancy.
- Five blocked RUN folds: consecutive 10 s blocks assigned cyclically; hold out
  a fold plus 1 s guard. Refit maps AND unit QC only in training samples.
  Independent 250 ms Poisson test windows, equal track priors and uniform prior
  over valid bins within track. Each track/fold needs >=20 windows, >=0.8 context
  accuracy and <=35 cm median position error conditional on the true track.
- PRE/POST require >=300 s immobile (<=5 cm/s) remote-rest samples each. These are
  not sleep-stage labels. No replay/content statistic informs inclusion.
- Audit supplied ripple data and its alignment before the primary published-
  criterion analysis. MUA-only remains a separately labelled sensitivity stratum.

## Scientific experiment to freeze after preflight, before replay outcomes

1. Reserve a detector-only population using RUN-only eligibility and a fixed seed.
   Detect immobile population bursts once. Keep windows, evaluation cells, map
   estimates and all evaluation spikes fixed as inference cells are thinned.
2. Within repeated disjoint inference/evaluation splits, use nested fractions
   1, 0.75, 0.5 and 0.25. Count an event once through repeat/split summaries, not
   as independent observations each time it is rescored.
3. Primary sequence criterion: weighted position-time correlation with both
   whole-time-bin permutation and per-cell place-field circular-shift nulls,
   following the established two-track framework. Use joint track/position
   normalization. Freeze two-track multiplicity, null count, rate floor and ripple
   criterion explicitly before scientific execution. Report null false-positive
   rates rather than assuming nominal alpha is correct.
4. Infer the track from inference neurons; evaluate track content using a disjoint
   evaluation population with RUN maps, also restricting to common-track units.
   No evaluation replay spikes may select windows, select a track, tune thresholds
   or choose inference-neuron subsets. Cellwise track-ID nulls preserve activity
   but break cross-population experience assignment. Test global firing-rate
   confounds using a conditional-count identity score in addition to Poisson.
5. Distinguish two claims: newly rejected events still carry independent track
   content; selection changes WHAT track/experience is inferred at the population
   level. The second requires a fixed evaluation readout and explicit composition
   decomposition, not only a loss of accepted events or wider uncertainty.
6. PRE/POST is a secondary interaction with immobility/state, event strength and
   per-unit drift controls. PRE is not a no-replay ground truth. Unequal track
   exposure times and first/second track identities must remain visible.
7. Static context reactivation is not ordered replay. If independent track content
   survives, test future held-out-neuron prediction against an occupancy/stickiness
   matched temporal null using the existing leakage-safe forecasting harness.
8. Negative controls: cellwise track-label permutation, evaluation cell-identity
   permutation matched for RUN rate, time-bin order permutation for sequence, and
   independently randomized candidate data. Calibration uses known-track RUN and
   synthetic paths; neither establishes latent replay ground truth.
9. Aggregate paired event-level changes within session then equally within animal;
   report all animal effects and animal-cluster uncertainty. Five animals at most:
   do not hide the small-cluster limitation behind narrow event-bootstrap CIs.

## Advancement and stop rules

The source-aligned criterion and independent content test must pass synthetic,
null and RUN calibration first. A pilot is technical only. Freeze executable
configuration and code hash before the scientific run. The held-out prediction
must be truly future prediction, not same-time smoothing or evaluation-spike
reinference. Existing negative content-certification attempts remain negative.

A positive answer needs independently supported content among newly rejected
events AND a reproducible selection-induced experience-content bias. If only
the first holds, report hidden content, not a changed biological conclusion.
If neither holds, preserve the negative result. Do not turn this into proof of
replay truth, memory consolidation, planning, uniform speed, or IMM novelty.

## Existing evidence carried forward

- PF/Tanni coverage dose intervention already shows acceptance loss; PF has
  future held-out predictive information among newly rejected events. Tanni's
  corresponding information result remains inconclusive.
- An earlier, different cohort included both 5,000-shuffle PF-style checks and
  still showed acceptance loss. Do not merge that cohort with the new predictive
  cohort or claim the joint criterion/content conclusion has already been shown.
- Prior PF Home-content, independent-population and predictive-context attempts
  had poor calibration or failed advancement gates. No known replay destination
  or goal truth is inferred from those attempts.
