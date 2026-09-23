# Frozen DANDI000978 coverage / cortical-content pilot

Version 1, 2026-09-23. Freeze this protocol and producer before evaluating PFC
candidate-window composition. This is exploratory in two source-supported animals,
not a confirmatory population-level result or a claim of independent ground truth.

## Question and novelty boundary

Does reducing recorded CA1 coverage cause otherwise identical candidate windows
to lose a geometric trajectory label even when independently recorded PFC
composition contains concordant RUN-route information?

Shin and Jadhav already report cross-area reactivation and altered hippocampal
sequentiality during coordinated ripples. Reproducing these alone is not novel:
https://doi.org/10.1016/j.cub.2024.05.018
The proposed addition is controlled measurement loss within fixed windows, with
independent cortical-content controls. Neither positive content nor a continuity
label proves replay of a particular experienced trajectory or causal transmission.

## Inputs and anatomical scope

Use pinned DANDI000978 0.240511.0307 and the hash-verified external source archive.
JS14 uses source-clarified tetrode IDs; ZT2 uses explicit per-file crosswalks.
Other animals stay excluded for unresolved anatomy, never decoding performance.
ZT2 is one animal. Source cluster matching does not prove sorting stability.
Private attachments and per-unit crosswalks are not redistributed.

## Freeze candidate selection before content scoring

1. Source SWR intervals, fully contained in source NREM and a single NWB rest
   epoch (no task trials). No PFC spike counts, rates, route scores or ripple
   coincidence enter event selection. These are source-defined SWR candidates;
   this experiment does not independently redetect or validate their LFP origins.
2. Duration 65-500 ms; all source boundaries preserved. Reject zero/invalid
   intervals and out-of-epoch events explicitly. Source NREM intervals can
   overhang NWB epoch boundaries by sub-ms rounding, but an accepted SWR must
   still fit the NWB epoch strictly. Do not shift or enlarge events.
3. Use the immediately preceding RUN epoch in the SAME file for encoding.
   No future RUN or cross-file unit matching is needed. Training-only inclusion:
   at least 20 RUN spikes/unit, at least 20 CA1 units and 5 PFC units, and at
   least three training trials of each canonical directed center/side route.
   The same 250 ms RUN preparation and 8 cm spatial map settings as the verified
   RUN validator are reused. Record map/epoch exclusions before event outcomes.
4. Require at least 10 full-CA1 spikes and five active encoding units in a source
   candidate. Do not impose PFC event activity thresholds. Resolve any overlapping
   eligible candidate intervals by keeping the earliest interval, documenting
   exclusions. Select at most 100 candidates per rest epoch with fixed seeded
   random sampling BEFORE continuity classification. Keep all if fewer qualify.
5. Write selection, training maps, eligibility tables and hashes. Scoring checks
   those hashes and requires the exact same clean producer commit.

## CA1 coverage and geometric label

Decode independently with flat spatial priors, Poisson observations and the fixed
training RUN maps: 20 ms windows advanced by 5 ms. Reuse the existing
`training_continuity` implementation: trim edges to two-spike windows, find the
longest consecutive MAP sequence with adjacent jumps strictly below 20 cm,
require at least 10 decoded positions and 40 cm endpoint displacement. Primary
uses edge support only; a 3-spike / 2-active-unit per-window version is sensitivity.
This is the Foster-style GEOMETRIC criterion, not the original complete shuffle-
validated replay definition. No ordered-replay claim follows from it alone.
Euclidean distances are used in the occupied W-track grid; graph topology and
direction-specific spatial maps remain limitations, so this is not a W-track
replay benchmark.

Keep event windows and spatial support fixed. Nested recorded-unit fractions:
100%, 75%, 50%, 25%, with 20 seeded repetitions. Subsets are chosen per training
RUN epoch and repeat, then reused for all its candidate events, never optimized
per event. The 100% label is computed once. PFC is never thinned or used to label.

Primary group: full-CA1 geometric-pass events that fail at 50% CA1 coverage in at
least half the repetitions. Report label-loss probability per event and count
each event once. Report all-event, full-pass, retained and never-pass groups too.
Distinguish lost label from insufficient support. Do not replace this group with
whichever coverage fraction, support rule or animal yields a positive result.

## Independent content endpoint and controls

Training rates for four directed routes use native start/end wells, as in RUN QC.
They can encode location/direction/behavior; they are not proof of abstract route
identity. Incorrect trials are not removed; noncanonical routes are excluded
from four-route templates but remain in spatial map fitting.

Full-CA1 population composition determines the reference route by maximum route
posterior in the fixed candidate window. PFC content is the per-spike conditional
log likelihood for that reference route minus the mean across four routes.
Condition on PFC total spike count; zero-spike PFC events contribute zero evidence
and remain in the population denominator. Use all trained PFC units, with no
eventwise confidence/activity filtering. This endpoint is sequenceless.

Two predeclared controls, each 199 draws:

- Matched-event pairing: permute PFC event score vectors among candidates in the
  same rest epoch, duration half (median split), and PFC spike-count stratum
  (zero / below-or-at nonzero median / above nonzero median). Require four events
  per stratum and derange pairings. Log all unmatched events. This preserves
  coarse duration/count/state and route marginal preferences, not exact firing
  patterns or fine sleep microstate. Primary analysis requires matched controls.
- PFC map-content control: independently permute the four training route labels
  within each PFC unit's rate vector, preserving each unit's rate values. Reuse
  each permutation across events in its epoch. Compare real concordance to that
  null with the CA1 reference route unchanged. Do not use an arbitrary mismatched
  animal's cells as a wrong map.

Report event-level real-minus-null-mean excess for both controls, nats per PFC
spike, zero-spike fractions, the fraction positive, full null distributions and
route/epoch composition. Summaries weight events once, then show epoch and animal
breakdowns. Repetitions are not independent events. With two rats, do not present
a rat bootstrap confidence interval as a robust population-level inference.

## Decision rule

Technical gates: frozen nonempty selection, all expected coverage rows, matched
anatomy, disjoint populations, no PFC selection input, no scoring failures and
source/producer hashes valid. Outcome gates remain separate.

Minimum pilot evidence: at least 10 primary lost-label events per animal, spanning
at least two rest epochs per animal; at least 80% of that group has matched-event
controls. A promising result requires positive mean excess versus BOTH controls
in BOTH animals, and observed animal-aggregated score above the control p95 for
both. A failed or underpowered group is not a negative biological conclusion.
No scaling, threshold relaxation or substitute subgroup is automatic.

Even a positive pilot does not establish publication readiness: more verified
animals, RUN-to-rest calibration and nuisance/sequence specificity remain needed.
The next action must follow the frozen outcome, including stopping this lead.
