# Real Recording Inputs for Coverage Calibration

Frozen before the all-session cache run on 2026-09-05. This step does not
estimate biological replay speed or choose a favorable subset of candidates.

## Cohort and Denominators

- Pfeiffer/Foster: all 4,069 immobile RUN high-MUA candidates in the supplied
  eight-session, four-rat candidate table.
- Tanni: all 8,072 pooled-spike-density candidates in the supplied 25-session,
  five-animal table. Each animal has recordings from several arena sizes, and
  the small arena is repeated. The five large-arena sessions contain the
  original 3,554 candidates; the remainder are not additional large-arena data.
- Use native mean-return core boundaries, not fixed 200 ms peak windows.
- No continuity, posterior, model evidence, or eventual decoded-speed filter.
- Missing sessions, timestamps, or encoding failures stay visible in the
  session/event audit. A failed session never silently disappears.
- Arena order and repeated small-arena sessions remain session covariates;
  these are not randomized arena-size manipulations or independent animals.

## Native Readout

PF uses the existing MAT session reader and supplied cell identities. Tanni uses
native NWB processed position columns (time, x, y), manual_1 clusters, and spike
timestamps. The idx_keep mask is honored whether labels cover all detections or
only retained detections. Unassigned/noise clusters 0 and 1 are excluded.
Tanni units are not labeled excitatory based on waveform data by this adapter.

Keep the absolute recording clock. Do not offset, rescale, or sort position
samples to conceal a clock reset. Samples that fail a cumulative monotonic-clock
check are counted explicitly. Invalid coordinates and gaps over 100 ms break
encoding support. Each contiguous block's edge frames are excluded from RUN
fitting to avoid inflated occupancy and interpolation across missing data.

Tanni wall coordinates come from native arena-size metadata with origin zero.
PF wall coordinates are intentionally unresolved at this stage. Tracking extrema
are saved as tracking extent, NOT substituted for the walls. This distinction
must be resolved before a PF wall-distance or bounded-trajectory experiment.

## Common Encoding Baseline

Use the repository's occupancy-normalized place-field fitter for both datasets:
8 cm grid, spatial smoothing SD 1.5 bins, speed at least 10 cm/s, minimum
occupancy 0.05 s, rate floor 1e-4 Hz, no arena padding. Speed is the canonical
unsmoothed centered finite-difference estimate; speed quantiles and tracking
support are exposed for subsequent QC. High-speed RUN defines the encoder;
replay spikes never estimate rates. No HMM or momentum dynamics enter it.

Store maps for ALL sorted cells, and separately record the pre-evidence mask:
at least 30 RUN spikes, mean RUN rate at most 4 Hz, peak rate at least 2 Hz,
split-half rate-map correlation at least 0.25. PF additionally requires the
provided excitatory-cell designation. Tanni lacks that equivalent designation;
its rate/stability-filtered units must not be described as waveform-verified
pyramidal cells. Counts and maps for failed units remain available for explicitly
labeled sensitivity analyses.

Store first/second chronological RUN-half maps on the same grid. These are
building blocks, not proof of held-out decoder quality. A future RUN decoding
test must make its unit inclusion and map support training-only; using the
all-RUN stability mask would otherwise leak test behavior through selection.

## Cache Contract

Each session NPZ contains full and half-map rates/occupancy, cell IDs, QC mask,
grid centers/edges, spatial-support mask, supported RUN intervals, position,
sorted spikes, and ragged candidate counts indexed by offsets. Candidate bins
are half-open, based at the original start time, with 5 ms nominal duration.
Partial final bins are retained with explicit duration and may not be treated as
full bins in a later window decoder. Event IDs stay stable. Zero-spike windows
and continuity failures remain in the denominator.

Outputs: coverage_input_candidates.csv, coverage_input_units.csv,
coverage_input_sessions.csv, coverage_input_gate_summary.csv,
coverage_input_manifest.json, coverage_input_report.md, and per-session NPZ/JSON.

Raw PF provenance hashes relevant entire MAT files. Tanni provenance hashes
every consumed HDF5 dataset's dtype, shape, and values; this is explicitly NOT a
whole-NWB hash of unused LFP and waveform payloads. Source file size/mtime and
before/after checks guard concurrent changes. Candidate CSVs, source code,
software versions, Git commit, and output cache hashes are recorded.

## Use and Claim Boundary

Input-readiness gates test preservation and parsing, not biological quality.
The next experiments must use paired subsets of these same candidate spikes
and one simulated draw per trial, with no redraw conditioned on decoded success.
Control cell count and actual field coverage; separate stationary/discontinuous
controls, true geometric eligibility, support failure, and decoder failure.
Validate constant speed AND both signs of a meaningful spatial speed gradient
on independent draws before applying any equivalence claim to real events.

The common baseline is not a bitwise reproduction of the old PF/Tanni decoders.
Its shared unit QC and conservative position-gap handling are intentional and
explicit. Any comparison with old retention counts must be labeled accordingly.
