# Odor-place parser repair and conditional neural feasibility

Protocol: `odor_place_source_v3_neural_feasibility_20261002_v1`, seed `20261001`.
Figshare 19620783 version 3; DANDI 001539 version `0.250815.1203`.
This is feasibility only. No replay sequences, content associations or biological
effects are fitted. Earlier stopgates and manuscript claims remain unchanged.

## Narrow parser amendment

The strict source parser rejects repeated timestamps. CS34 day 1 epoch 2 has
three adjacent nose-poke channel-5 on/off pairs at identical timestamps:
1618.414, 1618.528 and 1618.542 seconds. These are zero-duration samples, not
backward clock motion. The amended protocol inventories them as invalid samples
and allows separately verified surrounding trials to be reconstructed.

The exception applies uniformly across eight animals, exclusively to adjacent
channel-5 `1 -> 0` pairs. Other duplicate timestamps, repeated states, negative
time differences, nonfinite values and annotation conflicts remain invalid.
Zero-duration and aborted samples interrupt next-trial transitions; none are
skipped. Stable original-trigger identities support the immutable before/after
comparison even when the reconstructed row index changes.

Every verified source trial is additionally checked against raw nose-poke,
odor-solenoid and first-well digital edges without reusing the interval parser.
Source/NWB matches retain the 5 ms tolerance; no offsets are fitted. Every
screening and behavior threshold is identical to the original source protocol.

## Conditional neural stages

`acquire-neural` and `run-qc` require a passed source screen, independent raw
verification and a passed bounded exact-contrast novelty review. The existing
prior-art review remains provisional and is not silently promoted. Failure
produces not-run tables and an explicit limiting reason without reading neural
recordings or downloading NWB assets. A new frozen amendment would be needed to
change the novelty status, accompanied by its actual supporting review.

For a qualifying future source delivery, the adapter checks original tetrode,
cluster, anatomy and spike timing against NWB units. It considers row-versus-ID
electrode ambiguity explicitly; verified DynamicTableRegion references provide
row semantics. Multiunit records are not treated as sorted CA1 units. Spike-train
comparison uses a 1e-8 second serialization tolerance, not a fitted clock shift.

Chronological 70/30 separation applies to complete traversals globally, not
interleaved per-arm splits; each split must still satisfy the fixed per-arm
counts. Encoding uses only preceding observations from the same file. Training
unit inclusion, occupancy and flat spatial support cannot use validation spikes,
later replay spikes or future outcomes. Unsupported validation locations remain
in the coverage denominator; zero-spike windows remain decoded observations.
Accuracy is averaged within traversals, then within arms. Shared-stem positions
do not count as unique-arm accuracy observations. Tied arm posterior masses
receive half credit rather than preferentially favoring one arm.

After a decoder pass, maps are refitted from all preceding RUN observations.
The source-defined three-edge graph, fixed likelihood settings and ripple kernel
follow the frozen protocol. Ripple candidates must be wholly contained in valid
immobile pauses; adjacent valid tracking intervals may form a contiguous pause,
but separate ripple detections merge only on actual overlap. Zero-event
qualified trials remain observations. Supported candidates are sequence-testing
opportunities, never validated replay.

Saved likelihood/count arrays allow independent recomputation of RUN accuracy
and coverage. Commit/input-bound checkpoints and output hashes prevent silent
resumption with changed code, protocol or recordings. Conditional neural code is
synthetically tested; a source-blocked delivery is not a real neural validation.

## Durable execution

Run from a clean committed isolated checkout on gpuserver6000. The staged
supervisor survives SSH loss and records progress, logs and terminal exit status.

```bash
python scripts/launch_odor_place_feasibility.py \
  --job-dir /home/florianpfaff/odor-place-jobs/20261002-source-repair-v1 \
  --stages inventory verify acquire-neural run-qc verify report -- \
  audit_odor_place_post_error_feasibility.py \
  --dataset-root /home/florianpfaff/datasets/dandi001539-odor-place/original-v3 \
  --source-archive /home/florianpfaff/datasets/dandi001539-odor-place/original-v3/source/Figure1-6.zip \
  --reference-inventory /home/florianpfaff/odor-place-results/20261001-feasibility-v1 \
  --previous-audit /home/florianpfaff/odor-place-results/20261001-source-v3-audit \
  --protocol docs/odor_place_source_v3_neural_feasibility_protocol.json \
  --output-dir /home/florianpfaff/odor-place-results/20261002-source-repair-v1
```

The home filesystem, not Lexar, holds conditional acquisitions. Space checks
include all remaining selected download bytes plus the unchanged 30 GB reserve.
No data are deleted. A successful process exit means the feasibility calculation
completed, not that the scientific gates passed. Consult the decision and gate
tables first. `ready_for_calibration` requires the decoder-qualified cohort to
retain every unchanged screening floor; it is not a power certificate.
