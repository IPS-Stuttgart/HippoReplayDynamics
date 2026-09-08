# Common-pipeline predictive order by map results

Completed and independently audited on 2026-09-08, on gpuserver6000. This is a
new proper held-out-cell predictive experiment, not the historical all-cell
IMM-versus-fragmented shuffle or temperature-scaled held-out score.

## Fixed Experiment

- 4,001 PF events, eight sessions/four animals; 5,224 Tanni events, 25
  sessions/five animals. All common-eligible high-MUA candidates retained.
- Same RUN-only maps, source counts, five 70/30 cell splits and physical
  parameters as the parent conditional-prediction experiment.
- Twenty whole-bin permutations per event, shared across cells, maps and
  splits. Bin durations move with their population count vectors.
- Infer using training cells only; score held-out identities conditioned on
  their per-bin totals using the frozen posterior. No likelihood temperature.
- Four conditions: real/permuted spatial adjacency x original/shuffled order.
  One population-code-column permutation per session; not alternate-context
  encoding or a distribution of wrong maps.
- Mean over the 20 shuffled scores within split; paired contrasts first,
  then five-split event medians, session means, equal-session animal means,
  and equal-animal dataset means. The 5,000-draw hierarchical bootstrap keeps
  maps, partitions and shuffled draws fixed. Per-spike results are sensitivity
  endpoints, not replacements for raw-score decisions.

## Primary IMM Readout

| Dataset | Contrast | Mean predictive nats | 95% hierarchical CI | Positive animals |
|---|---|---:|---:|---:|
| PF | Original - shuffled, real map | +0.9142 | [+0.6984, +1.1165] | 4/4 |
| PF | Original - shuffled, permuted map | +0.5162 | [+0.4215, +0.6089] | 4/4 |
| PF | Order x map interaction | +0.4009 | [+0.2712, +0.5295] | 4/4 |
| Tanni | Original - shuffled, real map | +0.5646 | [+0.4172, +0.7246] | 5/5 |
| Tanni | Original - shuffled, permuted map | +0.3895 | [+0.2871, +0.4996] | 5/5 |
| Tanni | Order x map interaction | +0.1721 | [+0.1266, +0.2222] | 5/5 |

The interaction is `(real_original - real_shuffled) -
(wrong_original - wrong_shuffled)`, paired before aggregation. Separately
aggregated medians need not add, so do not reconstruct it from rounded table
entries or calculate a fraction of "spatial" versus "nonspatial" information.

Both datasets pass the frozen order-and-adjacency rule. The bounded result is:
temporal order contributes to held-out cell prediction, and correct adjacency
strengthens that contribution relative to the specified permutation. It does
not imply that all candidates are replay, that every event passes, or that an
IMM circuit generates these spikes.

The order effect does not eliminate all predictive benefit. IMM minus
independent-position prediction remains +0.5783 [+0.3857, +0.7262] in PF and
+0.3863 [+0.1171, +0.6830] in Tanni after time shuffling with the real map.
Persistent event content or other dependencies can therefore contribute.

## Diffusion Sensitivity

Diffusion also has a positive order effect and positive interaction in every
animal: PF +2.8549 [+1.5885, +4.3073] and +2.5577 [+1.3743, +3.9353]; Tanni
+1.0922 [+0.6868, +1.5596] and +0.9098 [+0.5543, +1.3157]. A larger shuffle
penalty is not automatically a better model. Tanni's original diffusion-minus-
independent score remains +0.0490 [-0.8451, +0.7655], positive in 3/5 animals.
Thus a positive order control can coexist with a weak unshuffled comparator.

## Limits Retained

The parent Tanni IMM-minus-other-event-composition result remains +1.7407
[-0.2368, +3.8124], positive in only 3/5 animals. The full parent replication
gate failed and is not rescued here. Composition calibration uses other
candidate events whereas spatial maps use RUN, so failure can reflect encoding
transfer rather than an absence of spatial content.

Candidate detection used all recorded cells before the predictive split;
prediction is conditional on frozen ascertainment, not training-only detection.
The endpoint sums marginal log predictions across bins, not joint event
evidence or a future-time forecast. Unknown latent causes can influence both
training and held-out neurons. Four/five animals remain the biological sample
size despite millions of score rows. Simulations and this observational
factorial do not identify a unique biological mechanism.

## Verification and Provenance

- Producer: `4ed213f3ce12a4ed881a7e4995b10b7badf85757`.
- Independent auditor/report version: `41650b1cf66c19f261a1706b033ce8792d393ab9`.
- Run directory:
  `/mnt/seagate10tb/florianpfaff/2d-predictive-order-map-all9225-k20-20260908`.
- Manifest SHA256:
  `40f4b123142ddd47b22e7db1311602b849a4d996d943076d26c07e0485c524ae`.
- Passing independent audit and non-rescoring report in sibling `-audit` and
  `-report` directories. Clean auditor working tree recorded.
- All 1,845,000 shuffled score rows, 184,500 permutations/count conservation,
  3,690,000 invariant scores, 830,250 split contrasts, 166,050 event contrasts,
  and 36 hierarchical intervals checked.
- Separate dynamic solver: 1,584 predictions across every session, both maps,
  two splits and two shuffles of three fixed events/session; maximum absolute
  discrepancy `7.391776080112322e-11`.
- Parent raw-count/map audit hash-pinned and reused; RUN maps not refitted.
- 67 focused tests pass; Ruff clean.

## Publication Interpretation

This adds a two-dataset conditional-prediction validation result. Time-order
controls, latent temporal models and co-firing-versus-sequence distinctions
already have direct precedent in [Maboudi et al. (2018)](https://elifesciences.org/articles/34467).
It is not a discovery that hippocampal spikes have sequential organization.
Neither constant physical speed nor surprise-driven Bayesian smoothing follows.
The strongest current paper remains the recording/decoder/selection/kinematic-
recovery measurement study; this factorial supplies a separate predictive
validation layer, not a high-importance mechanism claim.
