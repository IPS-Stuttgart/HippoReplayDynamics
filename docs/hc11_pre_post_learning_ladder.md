# hc-11 PRE/POST learning ladder

## Biological question

Does novel maze experience transform pre-existing NREM ripple activity into
map-specific, trajectory-active dynamics during POST sleep, particularly in the
slow-firing population associated with learned sequence plasticity?

The primary comparison uses matched PRE and POST ripple events selected without
model evidence. It reports all encoding cells, slow-firing cells, and
fast-firing cells. Slow/fast labels are rate-based proxies, not the original
paper's full plastic/rigid sequence classification. PRE-NREM firing rate is the
leakage-resistant primary definition; overall-session firing rate is the
paper-matched sensitivity. Combined PRE+POST NREM is retained only as a legacy
sensitivity because POST activity can change group membership.

The primary event definition follows the original paper's Bayesian replay
cohort: CA1 pyramidal spikes are binned at 1 ms, smoothed with a 15 ms Gaussian,
and population events must exceed the combined PRE+POST NREM mean by 3 SD. Event
boundaries occur when the smoothed population rate returns to its mean. Events
must last 100--500 ms, include at least one LFP ripple peak, and recruit at least
five and at least 10% of the eligible encoding cells. The replay models use 20 ms
bins. Bare LFP ripple envelopes are a sensitivity analysis, not the primary
event class.

## Event coverage

Five public sessions contain published ripple tables. Their published all-state
ripple catalogues are preferred and intersected with the current NREM labels;
this avoids a known truncated PRE interval in the
`Achilles_10252013.ripplesNREM` derivative while retaining published ripple
peaks. Buddy and two Gatsby
sessions require LFP detection from raw EEG. Generated events are never copied
silently into the processed dataset. They enter scoring through
`hc11_ripple_event_manifest.csv` only when:

- the same detector/channel strategy passes interval-overlap validation against
  a published ripple table;
- the generated event table and detector QC row agree;
- event intervals are finite, positive-duration, and restricted to NREM;
- both PRE and POST contain enough events for the frozen matched cohort.

Raw channels are never averaged before filtering. Ripple channels are filtered
separately and their squared ripple-band power is combined, avoiding cancellation
from phase reversals. Sessions without `channelTags.ripchans` use one documented
CA1 channel per anatomical shank, chosen from CA1 unit waveform channels while
excluding declared bad channels.

## Model taxonomy

The strict family split is:

```text
ordered:    diffusion, first_order_imm
nonordered: stationary, fragmented
```

Fragmented is spatial reactivation without an ordered path. It is never counted
as ordered replay.

## Validation ladder

Every candidate validated trajectory must pass:

1. ordered evidence exceeds nonordered evidence by at least 5.5;
2. first-order IMM posterior content is nonstationary and displaced;
3. content exceeds a cell-to-field map-permutation p95;
4. ordering advantage exceeds a whole-time-bin shuffle p95;
5. a training-cell posterior predicts held-out-cell spikes better than the
   nonordered comparator, without re-inferring the path from held-out spikes.

The final PRE/POST decision weights rats equally. It requires four rats, a
rat-bootstrap lower bound above zero, and positive leave-one-rat-out estimates
for validated trajectory fraction, map-specific mode mass, displacement,
time-order advantage, and held-out prediction. Slow-cell selectivity requires
the same robustness for the slow-minus-fast interaction.

## Current interim boundary

The five native-table sessions cover only Achilles and Cicero. In an earlier
balanced high-information 20-pair LFP-envelope run with 50 ms event padding,
POST improved the pooled
IMM-minus-fragmented margin, but the ordered margin remained median-negative and
the apparent positive events were localized to Achilles. A strict one-session
map/order/held-out audit yielded zero validated trajectories because POST mode
content became less map-specific. PRE-only rate grouping did not rescue the
slow-cell hypothesis.

These are interim two-rat results, not a biological verdict. The campaign remains
open until generated-event QC permits all eight sessions and four rats to enter
the same frozen ladder.

## Primary commands

```bash
python3 scripts/detect_hc11_lfp_ripples.py \
  --session-dir <processed-session> \
  --eeg-path <raw-eeg> \
  --output-mat <generated-ripple-table> \
  --output-dir <detector-session-output>

python3 scripts/prepare_hc11_ripple_event_manifest.py \
  --dataset-root <processed-root> \
  --detector-output-root <detector-output-root> \
  --validation-qc <native-validation-qc.csv> \
  --output-dir <manifest-output>

python3 scripts/score_hc11_pre_post_learning_evidence.py \
  --dataset-root <processed-root> \
  --ripple-event-manifest <hc11_ripple_event_manifest.csv> \
  --event-definition paper_population_synchrony \
  --selection-strategy random \
  --rate-group-scope pre_nrem \
  --time-bin-s 0.02 \
  --event-padding-s 0 \
  --output-dir <original-evidence-output>

python3 scripts/audit_hc11_pre_post_learning_controls.py \
  --dataset-root <processed-root> \
  --selection-csv <hc11_pre_post_event_selection.csv> \
  --rate-group-scope pre_nrem \
  --output-dir <strict-control-output>

# For session-parallel control runs, merge raw shard outputs and recompute all gates:
python3 scripts/merge_hc11_pre_post_learning_control_shards.py \
  --shard-dir <session-1-output> \
  --shard-dir <session-2-output> \
  --output-dir <merged-strict-control-output>
```

The scorer also writes `hc11_pre_post_event_detection_qc.csv`, which records
the eligible and selected event counts for every PRE/POST session before any
model result is inspected. Paper-matched overall-session rate groups and
high-information event selection are predeclared sensitivity runs and must not
replace the primary cohort after results are known.
