# Olafsdottir 2016 1D Z-Track Dataset

This dataset path supports applying HippoReplayIMM to the Olafsdottir,
Carpenter, and Barry 2016 Nature Neuroscience dataset:

<https://zenodo.org/records/5566548>

The scientific goal is to compare the existing 2D Pfeiffer/Foster open-field
replay result with a constrained 1D/linearized Z-track setting. The target
question is whether the trajectory-family / first-order IMM hierarchy weakens,
persists, or becomes mixed in 1D track replay.

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

On the GPU servers, prefer the shared cache path:

```text
/home/github-runner/.cache/datasets/olafsdottir2016
```

## Minimal Axona Readers

Phase 2 adds a small in-repo Axona subset parser:

- `read_axona_set(path)`: reads `.set` metadata into a string dictionary.
- `read_axona_pos(path)`: reads `.pos` position time series and LED-derived
  centimetre coordinates.
- `read_axona_cut(path, tetrode_path=None)`: reads `.cut` cluster labels and,
  when given the matching raw tetrode file, attaches spike times.
- `read_axona_egf(path)`: reads `.egf` int16 LFP samples and timestamps.

The readers intentionally cover only the fields needed for the first
paper-facing pass: Track1 position, sorted spike labels/times, SleepPOST LFP,
and Axona header metadata. Tests use tiny synthetic files under
`tests/fixtures/axona_minimal/` and do not require downloading the full Zenodo
archive.

## Downstream Design

The intended analysis after Phase 1 is:

1. Use `track1` as the 1D place-field encoding session.
2. Use the immediately following `sleepPOST` as the replay/SWR scoring session.
3. Fit linearized 1D place fields on Track1.
4. Detect replay/SWR candidates in SleepPOST.
5. Score candidates under 1D versions of stationary, diffusion, fragmented,
   first-order IMM, exact-sparse momentum, and lower-bound audit rows where
   feasible.
6. Compare the resulting 1D hierarchy to the existing 2D Pfeiffer/Foster
   hierarchy.

Interpretation should remain open:

- If IMM / trajectory-family weakens in 1D, that supports the idea that 2D
  geometry exposes richer replay dynamics.
- If IMM / trajectory-family remains strong in 1D, that supports a more general
  replay-dynamics interpretation.
- If mixed, analyze by animal, session, track geometry, replay direction, and
  reward/proximity variables if available.

## 1D-vs-2D Trajectory-Family Comparison

After the Olafsdottir 1D event-model-evidence table is generated, compare it
against the existing 2D Pfeiffer/Foster all-session evidence with:

`ash
PYTHONPATH=. python scripts/compare_olafsdottir_1d_2d_trajectory_family.py \
  --evidence-1d path/to/olafsdottir_1d_event_model_evidence.csv \
  --evidence-2d path/to/all_sessions_event_model_evidence.csv \
  --output results/olafsdottir-1d-vs-pfeiffer-foster-2d \
  --margin-threshold 5.5
`

Primary output:

`	ext
compare_1d_2d_trajectory_family_summary.csv
`

The headline table reports the same paper-path metrics for both datasets:
trajectory-family confident claim fraction, nontrajectory claim fraction,
mean and median family margin, first-order IMM raw-best fraction,
exact-sparse momentum raw-best fraction, and median exact-sparse
momentum-minus-diffusion margin.

Companion tables provide event-level rows, per-session summaries, per-animal
summaries, exact-core model hierarchy, paired momentum-vs-diffusion summaries,
animal-bootstrap intervals, and a gate summary. Interpret the contrast without
assuming the answer in advance: weaker 1D trajectory-family or IMM advantage
would support a 2D-geometry amplification account, while strong 1D evidence
would support a more general replay-dynamics account.

## Interpretation Rules

Use the comparison output as a guarded interpretation table, not as an automatic
claim generator. The script writes:

`	ext
compare_1d_2d_interpretation_summary.csv
`

Paper-safe rules:

- If 1D shows a weaker IMM or trajectory-family signal, treat this as support
  for the hypothesis that 2D open-field replay exposes richer trajectory
  dynamics that are less apparent in constrained 1D settings.
- If 1D shows a similarly strong trajectory-family signal, treat this as
  evidence that the trajectory-family signature generalizes beyond 2D
  open-field replay and may be a broader replay-dynamics feature.
- If 1D has strong trajectory-family evidence but less first-order IMM
  dominance, say that trajectory replay may generalize while the specific need
  for mode-flexible IMM may depend on environment geometry.
- If 1D fails because events or data are sparse, report a feasibility or data
  limitation, not negative evidence.

Avoid claiming that IMM is only apparent in 2D until the 1D pipeline produces a
robust weak or negative result. The default interpretation thresholds are
conservative and configurable with:

`ash
--min-interpretable-1d-events
--weak-fraction-gap
--similar-fraction-tolerance
`
