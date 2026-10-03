# Bilateral ripple spatial-content feasibility

This delivery assesses sources, independent RUN decoders and supported coincident
windows. It does not compare replay content, infer shared episodes, or change
existing manuscript claims. The earlier compute pause was explicitly lifted for
this study on 2026-10-03; it was not lifted for other experiments.

## Frozen entry point

`scripts/audit_hc11_bilateral_content_feasibility.py` exposes only `inventory`,
`run-qc`, `opportunities`, `verify` and non-rescoring `report`. All require
`--dataset-root`, `--protocol` and `--output-dir`. The protocol and seed are pinned.

Run detached from a clean committed checkout:

```bash
python scripts/run_hc11_bilateral_feasibility.py --detach \
  --dataset-root /mnt/lexar4tb/datasets/hc11-grosmark-buzsaki \
  --protocol docs/hc11_bilateral_content_feasibility_protocol.json \
  --output-dir /home/florianpfaff/results/hc11-bilateral-feasibility-20261003
```

The supervisor writes per-stage logs, progress, exit status, commands, commit,
environment and hash-checked checkpoints. No downloader or association stage is
provided. An `inconclusive_feasibility` report can be a successful execution.

## Required original-source sign-off

Converted unit IDs and region labels alone cannot establish original hemisphere
identity. Original `_sessInfo.mat` spike trains are matched to converted trains
without fitted offsets, with 25 microsecond tolerance (half a 20 kHz sample).
The documented original `SpikeID // 100` and `SpikeID % 100` recover spike group
and within-group cluster, not hemisphere. Missing or duplicate matches remain
unresolved. Group 16 is retained; it is never silently dropped.

Each session needs `bilateral_source_verification.json` next to its converted
spikes before neural stages can qualify. This is an independently reviewed source
crosswalk, not a generated assumption. Required fields:

- `status: verified`, `reviewer`, documented `animal`;
- `evidence` lists for `channel_order`, `spike_group_anatomy`,
  `coordinates_topology`, `traversals_directions`, `awake_rest`, `common_clock`,
  `pyramidal_layer`, each with original relative `path` and `sha256`;
- `spike_groups`, keyed by original group, with `shank`, `hemisphere`, `region`;
- `n_channels`, ordered `eeg_channel_ids`, and evidence-backed
  `channel_count_discrepancy_resolution` when XML disagrees;
- `coordinate_to_cm`, `coordinate_origin_cm`, `track_length_cm`, `topology`;
- chronological `traversals` (`id`, `start_s`, `end_s`, `direction`) and
  `running_directions`, independently checked against source tracking;
- `awake_rest_intervals_s`, independently verified as wakeful rest, not derived
  from low speed or relabelled Drowsy;
- `pyramidal_layer_channel_ids`, hemisphere then shank then eligible channel IDs.

Missing sign-off blocks qualification. Gatsby's published 134-channel override
and the omitted Buddy/Gatsby group-16 mapping are review issues, not defaults.
The original channel-order and recording-summary documents have not yet been
located in the inventoried root; no automatic acquisition is authorized.

## Numerical conventions

RUN maps use actual training spikes, 4 cm bins, distance-based 6 cm smoothing,
0.5 s support and 0.25 s global-rate pseudocount. They never use future RUN/replay
spikes, occupancy priors, a guessed coordinate scale or refitting. Validation is
chronological, traversal-balanced, by direction and hemisphere. Zero-spike bins
remain observations; coverage includes unsupported targets in its denominator.

PRE-NREM alone determines baseline normalization and one eligible channel per
verified shank, ranked by mean detected peak z score with channel-identity ties.
Detection never filters across disjoint state segments. The operational detector
is not advertised as an exact reproduction of the 2025 implementation.
Overlapping channel detections retain parent IDs and compound flags. Compound
events are inventoried but do not qualify as clean opportunities.

Ripple matching maximizes cardinality before minimizing total peak separation.
The sorted identity order breaks equal-cost assignment ties deterministically.
A common window requires the same five 20 ms bins to satisfy bilateral support;
five disjoint supported bins on opposite sides do not qualify.

## Scope of verification and limits

Original timing crosswalks are reconstructed independently. RUN posteriors are
recomputed from saved counts/maps with an independent softmax calculation;
opportunity totals and one-to-one identities are independently recounted.
Untested real-data neural paths remain unavailable if source prerequisites fail.
No representative replay panel is fabricated when there are no qualified windows.
The source-availability panel is an inventory figure only. Fixed earliest examples
require a later source-qualified execution; their absence is reported explicitly.

Novelty is unresolved pending supplement/code and subsequent-study review.
The prior-art ledger identifies close precedents rather than claiming a first.
Readiness requires a closed novelty review, source verification and at least
three animals with 30 qualified windows in each state. These are screening floors,
not a power certificate. Unavailable measurements are null, never zero.
