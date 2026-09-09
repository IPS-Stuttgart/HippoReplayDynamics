# Continuity Rejection Does Not Imply Absence Of Predictive Structure

Completed on gpuserver6000, 2026-09-10. Retrospective methods result, not a new
biological replay classification or evidence for uniform physical speed.

## Question And Design

This follow-up asks whether events rejected by a recording-sensitive geometric
screen nevertheless contain independently predictive population structure.
It connects the earlier recording-coverage benchmark to real-data validation.
The [protocol](training_continuity_prediction_protocol.md) was committed before
the new classifications and stratified results were computed. The underlying
datasets and whole-cohort predictive results had already been inspected.

All 9,225 frozen immobile high-MUA candidates were retained: 4,001 PF candidates
in eight sessions/four animals and 5,224 Tanni candidates in 25 sessions/five
animals. Five existing 70% training/30% held-out cell partitions were reused.
Candidate detection had used all cells; claims therefore condition on this
previous ascertainment. No new event selection was based on held-out scores.

Each training-cell population was decoded independently in overlapping 20 ms
windows at 5 ms strides, with a flat spatial prior and untempered Poisson
likelihood, including the silence term. The screen retained the longest MAP
run with adjacent jumps below 20 cm, at least ten frames and at least 40 cm
endpoint displacement. Unsupported edges were trimmed. This is the geometric
stage of a transferred PF-style criterion, not an exact authors' pipeline:
the two 5,000-shuffle significance tests were not repeated.

An independently seeded half of the training population was also screened.
Thus the smaller population contains approximately 35% of all included cells;
the held-out 30% never enters either screen. RUN-derived maps were fixed.

Two prespecified groups were tested:

- Rejected with opportunity: the full training population supplies enough
  supported frames to attempt the test, but fails geometry.
- Lost with thinning: passes geometry with 70% training cells and fails with
  the nested half. Predictive scores still use the larger training population.

Existing predictive scores were reused, not rescored or fitted to the new
labels. They predict held-out cell identities conditional on the held-out
spike total in each bin. These are sums of marginal predictive log scores,
not joint event evidence or a future-time forecast. Held-out spikes do not
update the inferred latent posterior.

## Primary Results

Values are event medians over qualifying splits, then session means,
equal-session animal means and equal-animal dataset means. Brackets are
95% animal/session/event hierarchical bootstrap intervals, conditional on
the frozen maps, partitions, candidates and 20 order shuffles.

| Group | Predictive contrast, nats | PF | Tanni |
| --- | --- | --- | --- |
| Rejected with opportunity | IMM - independent positions | +1.541 [1.335, 1.761] | +0.913 [0.671, 1.190] |
| Rejected with opportunity | IMM - static location | +5.419 [3.169, 7.891] | +2.935 [1.641, 4.614] |
| Rejected with opportunity | IMM - nonspatial composition | +11.672 [7.500, 16.041] | +1.773 [-0.245, 3.865] |
| Rejected with opportunity | Original - shuffled, real map | +1.008 [0.772, 1.209] | +0.591 [0.433, 0.765] |
| Rejected with opportunity | Order-by-map interaction | +0.441 [0.305, 0.563] | +0.184 [0.133, 0.238] |
| Lost with thinning | IMM - independent positions | +1.603 [1.417, 1.791] | +0.801 [0.476, 1.178] |
| Lost with thinning | IMM - static location | +9.887 [5.967, 14.334] | +4.410 [1.965, 7.479] |
| Lost with thinning | IMM - nonspatial composition | +14.838 [9.271, 20.662] | +2.109 [-0.410, 4.526] |
| Lost with thinning | Original - shuffled, real map | +1.427 [1.161, 1.664] | +0.769 [0.466, 1.068] |
| Lost with thinning | Order-by-map interaction | +0.661 [0.493, 0.814] | +0.265 [0.169, 0.359] |

The nonspatial comparator is a global cell-composition model calibrated on
other events, not a flat distribution over neurons. The factorial interaction
asks whether original-minus-shuffled prediction is greater with the real map
than with the fixed shared population-code permutation.

Every primary PF contrast has a positive interval and positive means in all
four animals. Both PF groups pass the prespecified joint predictive-support
rule. Tanni fails that joint rule in both groups: the composition contrast
crosses zero and only three of five animals have a positive mean. Positive
shuffle effects do not override this failed comparator.

There are 3,542 PF and 5,055 Tanni events in the rejected group, and 1,477 PF
and 752 Tanni events in the lost-with-thinning group. These count events
qualifying in at least one split. They are not counts of independently
significant events, and the two groups can overlap across different splits.
They must not be added together. Four PF and five Tanni animals, not thousands
of events, remain the biological replication units.

## Sensitivities And Mechanistic Limits

The equal-animal geometric acceptance fraction falls from 20.75% to 7.20% in
PF and from 5.63% to 1.42% in Tanni when moving from the 70% training population
to its nested half. These are geometric-only rates, not the earlier all-cell
two-shuffle acceptance rates.

Changing ten frames to eleven, adding internal spike/cell support requirements,
and applying the declared Tanni boundary-mask sensitivity preserves the sign
of all five PF contrasts in all four animals. For rejected PF events the
order advantage ranges from +1.008 to +1.117 and the order-by-map interaction
from +0.441 to +0.492 nats. These are descriptive variants, not newly selected
primary tests. Split zero alone also preserves the PF signs in all animals.

Tanni's composition limitation persists across these variants. Its per-spike
composition intervals are positive at the primary setting, but per-spike
normalization was a different, descriptive estimand; it cannot replace the
failed primary raw predictive-score endpoint.

PF arena bounds were explicitly unavailable in the parent caches. Its clipped
rows are tagged no-op aliases and do not establish clipping robustness. Tanni
has finite bounds; occupied bins can straddle the boundary, so clipping their
centres is not assumed to be the correct biological mask. See the preserved
[missing-bounds addendum](training_continuity_missing_bounds_addendum.md).

This is not uniquely an IMM effect. In rejected PF events, the prespecified
descriptive diffusion contrasts also have positive animal means in all four
animals: +1.415 versus independent positions, +5.320 versus static location,
+11.571 versus composition, +2.656 original-minus-shuffled and +2.352 for the
order-by-map interaction. They have not been given additional primary CIs.
Tanni diffusion does not uniformly beat the independent-position comparator.

The larger-training-population prediction for events lost under thinning does
not demonstrate prediction from the smaller population. Nor does a predictive
group mean certify every rejected event, establish latent continuous replay,
recover accurate physical speed, or estimate the full heuristic's false-negative
rate. Structured but noncontinuous activity remains a possible explanation.
No fuzzy continuity classifier was validated by this experiment.

## What This Adds To A Paper

The useful statement is:

> In Pfeiffer-Foster, failure of a transferred geometric continuity screen does
> not imply absence of independently predictive neural structure. This includes
> events whose geometric acceptance is lost when training neurons are removed.

It strengthens a methods paper about the distinction between recording-dependent
trajectory detectability and underlying population information. It does not
establish a new replay mechanism, prove that rejected events are true replay,
or establish that physical replay speed is spatially uniform. The Tanni failure
against a nonspatial comparator must accompany, not disappear behind, the PF
positive result. Whole-cohort replication decisions are unchanged.

Independent validation without replay ground truth is already a central aim of
[Takigawa et al. (2024)](https://elifesciences.org/articles/85635), who use track
discriminability to assess sequence-based detection. Comparing sequence metrics
is also not itself new: [Huh et al. (2026)](https://doi.org/10.1038/s41467-026-74822-2)
introduce a firing-order likelihood method. The possible incremental contribution
here is the combination of same-event cell thinning, a training-only geometric
screen, and separate-neuron prediction with composition and order-by-map controls.
This limited literature check does not establish priority for that combination.

Recommended placement: a real-data validation panel in the existing recording-
coverage methods manuscript, with the simulation-based sensitivity/calibration
results. Do not promote it to a standalone biological discovery or alter
thresholds to make Tanni pass. Collaborator assessment of this incremental
contribution is more useful than another unconstrained threshold sweep.

## Reproducibility

Server worktree:
`/home/florianpfaff/HippoReplayDynamics-training-only-continuity-prediction`.
Branch: `test-training-only-continuity-prediction`. No push performed.

Production commit: `02667dfe9a9eaeddde84533eb7909419a79094b8`.
Reporter commit: `0ff1634dfddd99929885577fbb7f36bca73f744a`.
Final independent report-audit commit:
`c55d13c26931401eaf8687b3bea6f895accb3c10`.

Artifact prefix:
`/mnt/seagate10tb/florianpfaff/training-continuity-prediction-all9225-20260910-v2`.

- Production directory: prefix itself, all 33 sessions, 369,000 classification
  rows and 46,125 event/split predictive-contrast rows. Production runtime 17.64 s.
- `-audit/training_continuity_audit.json`: pass; 184,500 independently reconstructed
  MAP paths, 738,000 geometric metric checks and 46,125 predictive-contrast
  reconstructions. No differing/tied MAP indices. This audit does not rerun
  the already audited held-out posterior scorer.
- `-report/`: primary event contrasts, session/animal/sensitivity tables,
  primary intervals, decisions, figure, markdown report and provenance manifest.
- `-report-audit-final/training_continuity_report_audit.json`: independent
  reconstruction of all-setting group/session/animal estimates, primary point
  estimates and all 20 primary hierarchical intervals. The descriptive split-zero,
  acceptance-rate and clipping-disagreement tables are not independently
  reaggregated by that audit.
- `-validation-final/validation.json`: 135 tests passed, 5.33 s reported by
  pytest; Ruff, compilation and whitespace checks passed. Logs and code hashes
  are preserved. All scientific computation ran on gpuserver6000.

Production manifest SHA256:
`4dc851e6d498b820ea7ec9d370167a72b196ee26d4232d20c69fd07228553b76`.
Classification audit SHA256:
`ab1f20ba9194b6192638ee7121545f51d8c0b464c2baf4c98fb4567e5f0b7d32`.
Report manifest SHA256:
`62ca8187bc0c2bf768cb578577dda3be8933ae49990e04ecd64ff7cedfcc0ee0`.

The original failed run and the first report audit are retained separately;
the corrected production did not overwrite them. The correction concerned
missing PF arena-bound metadata, not classifier thresholds or predictive scores.
