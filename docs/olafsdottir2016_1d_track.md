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
