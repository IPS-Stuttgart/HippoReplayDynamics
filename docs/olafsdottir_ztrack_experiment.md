# Olafsdottir Z-Track Replay Experiment

This experiment applies the HippoReplayIMM model-evidence stack to the
Olafsdottir, Carpenter, and Barry 2016 Z-track dataset from Zenodo record
5566548.

The goal is not to assume that the Pfeiffer/Foster trajectory-family result is
specific to 2D open fields. The goal is to compare the existing 2D open-field
result with a constrained 1D/linearized-track replay setting.

## Bridge Adapter

PR 4 adds a provisional Pfeiffer/Foster-style MAT adapter:

```bash
PYTHONPATH=src python scripts/prepare_olafsdottir_ztrack_sessions.py \
  --extracted-root /home/github-runner/.cache/datasets/olafsdottir2016/extracted \
  --output-root /home/github-runner/.cache/datasets/olafsdottir2016/derived_pfeiffer \
  --sessions R2142/ZTrack20140806 \
  --tetrode-mode hippocampus \
  --lfp-detector-mode mean-envelope \
  --min-event-spikes 5 \
  --min-event-active-cells 3
```

This is a bridge adapter, not a final native Olafsdottir data model. It writes
the MAT files consumed by the existing benchmark scripts:

```text
Position_Data.mat
Spike_Data.mat
Ripple_Events.mat
Epochs.mat
Experiment_Information.mat
```

## Design

Use each Track1/SleepPOST day pair as one derived replay session.

- `track1`: behavior and place-field encoding session
- `sleepPOST`: post-track rest replay/SWR scoring session
- `Training`: open-field foraging, not primary for this 1D comparison

Track1 position and spikes remain at native time. SleepPOST spikes and detected
ripple events are shifted after the Track1 epoch. `Epochs.Run_Times` covers
Track1 only, which forces place-field fitting to use Track1 while replay scoring
uses SleepPOST events.

Do not hide the time-shift trick: it is practical glue for the current
Pfeiffer/Foster-style loader, not a biological statement. The derived session
records the offset in `sleep_time_offset_s`.

## Adapter Details

The adapter currently:

- uses Track1 for encoding through `Run_Times`;
- shifts SleepPOST spikes and ripple events after Track1 for replay scoring;
- uses `.cut` spike sorting and ignores `.clu`;
- handles the R2142 hippocampal tetrode reversal;
- linearizes Track1 position by occupied-bin geodesic distance;
- detects ripple candidates from per-channel LFP envelopes before combining
  channels;
- supports optional spike-support filters with `--min-event-spikes` and
  `--min-event-active-cells`.

Derived sessions include these metadata fields:

```text
source_dataset
source_animal
source_track_session
source_sleep_session
adapter_schema_version
linearization_method
event_detector
event_filter_parameters
sleep_time_offset_s
track_duration_s
```

The same fields are also written into `conversion_summary.json` where practical.

## Primary Scoring

After preparation, run the exact-core model stack on a derived session:

```bash
PYTHONPATH=src python scripts/benchmark_model_evidence.py \
  --dataset-root /home/github-runner/.cache/datasets/olafsdottir2016/derived_pfeiffer \
  --session R2142/ZTrack20140806 \
  --events all \
  --models sorted-spike-state-space-stationary,sorted-spike-state-space-diffusion,sorted-spike-state-space-fragmented,sorted-spike-state-space-first-order-imm,sorted-spike-state-space-momentum-exact-sparse \
  --bin-size-cm 5 \
  --min-speed-cm-s 4 \
  --time-bin-s 0.02 \
  --output results/olafsdottir-ztrack-r2142-exact-core \
  --continue-on-error
```

## Current R2142 Smoke

The current R2142 bridge smoke result is:

```text
stationary 9/21
diffusion 5/21
fragmented 4/21
exact-sparse momentum 3/21
first-order IMM 0/21
strong exact margin fraction 0.0
```

Interpret this as a clean ingestion/scoring smoke only. It shows that the
derived session can be built and scored by the existing evidence path. It does
not yet support a trajectory-family, first-order IMM, or momentum biological
claim.

## Main Comparisons

Within the 1D Z-track artifact:

- trajectory-family vs stationary/static
- first-order IMM vs exact-sparse momentum
- exact-sparse momentum vs diffusion

Against the existing 2D Pfeiffer/Foster result:

- Does trajectory-family still dominate static/nontrajectory?
- Is first-order IMM still the leading exact core row?
- Does exact-sparse momentum still beat diffusion but fail to be the full-core
  winner?

Possible outcomes:

- IMM/trajectory-family weakens in 1D, suggesting 2D geometry exposes richer
  replay dynamics.
- IMM/trajectory-family remains strong in 1D, suggesting a general replay
  dynamics feature.
- The result is mixed and should be analyzed by rat, session, track geometry,
  replay direction, and reward/proximity variables if available.

## Interpretation Boundary

Do not make a trajectory-family, IMM, or momentum claim from a run where ripple
events are spike-poor or exact evidence margins are all ties. In that case, the
correct conclusion is only that the bridge adapter and benchmark path execute.
