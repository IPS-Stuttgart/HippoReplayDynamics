# Olafsdottir 2016 1D Z-Track Dataset

This dataset path supports applying HippoReplayIMM to the Olafsdottir,
Carpenter, and Barry 2016 Nature Neuroscience dataset:

<https://zenodo.org/records/5566548>

The first adapter PR is intentionally limited to dataset discovery and minimal
Axona parsing. Later PRs can add Z-track linearization, sleep replay-event
detection, 1D scoring, and 1D-vs-2D comparison.

## Dataset Layout

The Zenodo record distributes `Olafsdottir2016.zip` under CC-BY 4.0.

Expected archive:

```text
URL: https://zenodo.org/records/5566548/files/Olafsdottir2016.zip
MD5: 93a9288449bd28d74ce4ff7ed7c6f878
```

Axona-style files include:

- `.cut`: spike-sorted cluster assignments; use these over `.clu`
- `.pos`: position sampled at 50 Hz
- `.egf`: LFP sampled at 4.8 kHz
- `.set`: session headers
- `.<number>`: raw tetrode spike waveform/timestamp files
- `.clu`: KlustaKwik output; ignored for this pipeline because `.cut`
  supersedes it

Session names identify behavioral state:

- `track1`: Z-track encoding session
- `sleepPOST`: post-track rest / candidate replay session
- `Training`: open-field foraging, not the primary 1D comparison
- `screening`: screening sessions

## Tetrode Arrangement

The Zenodo description notes a tetrode-region caveat:

- For most recordings, tetrodes 1-8 are MEC and tetrodes 9-16 are hippocampus.
- For R2142, this arrangement is reversed.

The manifest records both `hippocampal_tetrodes` and `mec_tetrodes` explicitly
for every animal/session row.

## Manifest Phase

Build a manifest from an already extracted dataset:

```bash
PYTHONPATH=src python scripts/prepare_olafsdottir2016_dataset.py \
  --dataset-root data/Olafsdottir2016
```

This writes:

```text
data/Olafsdottir2016/olafsdottir2016_manifest.csv
```

Manifest columns:

```text
animal
date
session_type
session_name
session_path
has_pos
has_set
n_cut_files
n_egf_files
n_tetrode_files
hippocampal_tetrodes
mec_tetrodes
notes
```

The script can also download and/or extract the archive, but those actions are
opt-in so CI and unit tests never fetch the 9.9 GB file:

```bash
PYTHONPATH=src python scripts/prepare_olafsdottir2016_dataset.py \
  --dataset-root data/Olafsdottir2016 \
  --zenodo-url https://zenodo.org/records/5566548/files/Olafsdottir2016.zip \
  --expected-md5 93a9288449bd28d74ce4ff7ed7c6f878 \
  --download \
  --extract
```

The manifest command also supports:

```text
--archive-path
--manifest-output
--force-download
```

On the GPU servers, prefer the shared cache path:

```text
/home/github-runner/.cache/datasets/olafsdottir2016
```

## Minimal Axona Readers

This PR adds a small in-repo Axona subset parser:

- `read_axona_set(path)`: reads `.set` metadata into a string dictionary.
- `read_axona_pos(path)`: reads `.pos` position time series and LED-derived
  centimetre coordinates.
- `read_axona_cut(path, tetrode_path=None)`: reads `.cut` cluster labels and,
  when given the matching raw tetrode file, attaches spike times.
- `read_axona_egf(path)`: reads `.egf` int16 LFP samples and timestamps.
- `read_axona_tetrode_spike_times(path)`: reads raw Axona tetrode spike
  timestamps from `.<number>` files.

The readers intentionally cover only the fields needed for the first
paper-facing pass: Track1 position, sorted spike labels/times, SleepPOST LFP,
and Axona header metadata. Tests use tiny synthetic files under
`tests/fixtures/axona_minimal/` and do not require downloading the full Zenodo
archive.

## Z-Track Linearization

PR 2 adds a conservative Track1 position linearization helper:

```bash
PYTHONPATH=src python scripts/linearize_olafsdottir_ztrack.py \
  --pos path/to/track1.pos \
  --output results/olafsdottir-linearized-track1
```

The script can infer a simple occupied-bin diameter centerline, or consume a
hand-specified centerline:

```bash
PYTHONPATH=src python scripts/linearize_olafsdottir_ztrack.py \
  --pos path/to/track1.pos \
  --centerline-json path/to/track_geometry_seed.json \
  --output results/olafsdottir-linearized-track1
```

Required outputs:

```text
linearized_position.csv
track_geometry.json
linearization_diagnostics.csv
```

`linearized_position.csv` columns:

```text
time_s
x_cm
y_cm
linear_position_cm
speed_cm_s
valid_position
```

`linearization_diagnostics.csv` reports scalar diagnostics and occupancy rows:

```text
fraction_valid_position
median_projection_error_cm
max_projection_error_cm
track_length_cm
occupancy_by_linear_bin
position_start_time_s
position_end_time_s
session_duration_s
```

Acceptance criteria should remain diagnostic rather than absolute. For a
realistic track-running session, expect more than 90% valid position samples,
linear position that covers the expected Z-track extent, non-collapsed occupancy
across the track, and projection/geodesic diagnostics that are visible for
inspection. If a session falls below 90% valid position, document it as a
session/data-quality limitation rather than silently excluding it.

The R2142 Track1 smoke check preserves the Axona binary parser fix where
`data_start` can be followed immediately by binary data rather than a newline.
With the current cached R2142 Track1 copy, position time spans 0.0 to 2275.72 s.
The same smoke check produced a valid-position fraction of about 0.71, an
inferred track length of about 396 cm, valid linearized positions spanning the
full inferred track, and nonzero occupancy across all 80 diagnostic bins. This
is useful as an end-to-end file-format and diagnostic-output check, but it is
not by itself a paper-quality linearization claim. Do not overclaim
linearization quality from this single-animal smoke check; use the diagnostics
to decide whether a session is ready for scoring.

## SleepPOST Candidate Event Detection

PR 3 adds an initial conservative SleepPOST replay/SWR candidate detector:

```bash
PYTHONPATH=src python scripts/detect_olafsdottir_sleep_replay_events.py \
  --dataset-root data/Olafsdottir2016 \
  --output results/olafsdottir-sleeppost-events \
  --min-event-spikes 5 \
  --min-event-active-cells 3
```

The detector pairs `track1` and `sleepPOST` rows from
`olafsdottir2016_manifest.csv`, reads hippocampal `.egf` channels, computes a
ripple-band envelope, combines channels, detects threshold crossings, and then
applies duration, spike-count, and active-cell gates.

The default channel combination is `--combine-method max`, which computes the
envelope/z-score separately per available hippocampal EGF channel and then uses
the channel-wise maximum. Use `--combine-method mean` for a stricter
across-channel average.

Supported gate controls:

```text
--min-event-spikes
--min-event-active-cells
--min-duration-s
--max-duration-s
```

Required outputs:

```text
sleep_replay_events.csv
ripple_detection_summary.csv
```

`sleep_replay_events.csv` columns:

```text
event_index
start_time_s
end_time_s
duration_s
peak_time_s
peak_ripple_z
n_spikes
n_active_cells
animal
date
track_session
sleep_session
event_detector
detector_parameters
```

The R2142 pilot run with `--min-event-spikes 5` and
`--min-event-active-cells 3` produced 21 SleepPOST candidate events, with median
event spikes 8 and maximum event spikes 55. Keep this as a pilot value for
pipeline orientation, not as final SWR/replay prevalence.

With this PR's default detector settings on the cached R2142 copy
(`--combine-method max`, `--ripple-z-threshold 3.2`, `--min-duration-s 0.005`),
the same spike-support gates produced 20 candidate events, with median event
spikes 8 and maximum event spikes 34. Treat both values as detector-parameter
sensitivity notes until LFP/ripple diagnostics are reviewed.

Important caveat: this is an initial candidate detector. Do not treat these
events as final SWR detections until LFP/ripple diagnostics are reviewed.

## 1D Evidence Workflow

PR 6 adds a workflow smoke that chains the bridge adapter into exact-core 1D
model-evidence scoring:

```bash
PYTHONPATH=src python scripts/benchmark_olafsdottir_1d_replay_evidence.py \
  --extracted-root /home/github-runner/.cache/datasets/olafsdottir2016/extracted \
  --derived-root results/olafsdottir-1d-derived \
  --output results/olafsdottir-1d-evidence \
  --session R2142/ZTrack20140806 \
  --max-events 5
```

The script writes:

```text
olafsdottir_1d_event_model_evidence.csv
olafsdottir_1d_family_margin_decisions.csv
olafsdottir_1d_family_margin_summary.csv
olafsdottir_1d_exact_core_model_claim_summary.csv
olafsdottir_1d_paired_momentum_diffusion_summary.csv
olafsdottir_1d_session_summary.csv
olafsdottir_1d_control_gate_summary.csv
```

The first milestone is end-to-end ingestion, linearization, candidate-event
detection, exact-core evidence scoring, and summary-table generation. It is not
a positive biological result.

## 1D-vs-2D Comparison Layer

PR 7 adds a comparison script for completed event-level evidence artifacts:

```bash
PYTHONPATH=src python scripts/compare_olafsdottir_1d_2d_trajectory_family.py \
  --olafsdottir-1d-evidence results/olafsdottir-1d-evidence \
  --pfeiffer-foster-2d-evidence path/to/all_sessions_event_model_evidence.csv \
  --output results/olafsdottir-1d-2d-comparison \
  --margin-threshold 5.5
```

Required outputs:

```text
compare_1d_2d_trajectory_family_summary.csv
compare_1d_2d_interpretation_summary.csv
```

The primary comparison columns are:

```text
dataset
environment_type
events
trajectory_confident_claim_fraction
nontrajectory_confident_claim_fraction
mean_family_margin
median_family_margin
first_order_imm_raw_best_fraction
momentum_raw_best_fraction
momentum_vs_diffusion_median
mean_family_margin_per_spike
median_family_margin_per_spike
mean_family_margin_per_time_bin
median_family_margin_per_time_bin
mean_spikes_per_event
median_spikes_per_event
mean_time_bins_per_event
median_time_bins_per_event
```

The normalized margin columns are mandatory because raw 1D and 2D log-evidence
margins are not directly comparable across different spike counts and event
lengths.

The interpretation table can label the comparison as:

```text
weaker_1d_signal
similarly_strong_1d_signal
strong_trajectory_family_but_weaker_imm_dominance
mixed_1d_result
sparse_or_data_limited_feasibility_result
```

The hard caveat is part of the output: do not claim IMM is only apparent in 2D
without a robust weak or negative 1D result.

The current R2142 pilot suggests that 1D Z-track replay, under the provisional
adapter and detector, does not show the strong trajectory-family/IMM signature
seen in the 2D Pfeiffer/Foster open-field result. Keep this as
hypothesis-generating smoke, not a biological conclusion, because it uses one
animal/day, a small candidate-event set, preliminary event detection,
diagnostic-only linearization, and an adapter whose Track1/SleepPOST cell
identity alignment still needs scaled verification. The strong exact-margin
fraction in the pilot is 0.0, so model separation is low-confidence.

Using the 21-event R2142 pilot table against the 160-event Pfeiffer/Foster
full-core artifact, the comparison layer reports 0/21 1D trajectory-confident
claims, 0/21 nontrajectory-confident claims, median 1D family margin about
+0.011, 0/21 first-order IMM raw best, and 3/21 exact-sparse momentum raw best.
The same comparison labels the result
`sparse_or_data_limited_feasibility_result` with directional pattern
`weaker_1d_signal`. Treat that as a scaling target, not a paper claim.
