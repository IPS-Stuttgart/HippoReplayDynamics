# Hippocampal-cortical experience-content preflight

Scope: public-data integrity and metadata only. No replay selection, POST content
decoding, fitted experience classifier, or biological claim is authorized by a
successful inventory. This protocol was written before inspecting the other
14 navigation payloads. The authors' example M02/2024-03-13 file was inspected to
establish the schema; this inspection revealed no explicit maze/trial table.

## Frozen inputs

- DANDI 001695, published version 0.260319.2023, DOI
  10.48324/dandi.001695/0.260319.2023, CC-BY-4.0.
- Published catalog SHA256
  d5b66e12bfbc2ff6eea508948cc728da755cafde8685d647ff09d5da1a4fdf22.
- All 15 behavior+ecephys files; no neural-outcome-based selection. The seven
  other files are outside the navigation cohort, not failed recordings.
- Each payload must match its published byte size and SHA256. No unchecked
  cached file may be reused. Preserve all source files unchanged.

## Checks and limitations

Record unit-region/type counts, ragged spike-index integrity, sorted finite
timestamps, position shape and clock, available interval tables and label
counts, declared position units/conversion, and LFP sample-clock extent. Region
labels come from units/cell_area, not the electrode table's generic location.
Clock-range overlap alone does not establish millisecond synchronization.
Published pyramidal-cell labels do not establish stability across maze epochs.

Inventory HDF5 references as references, not strings. Compute LFP end time as
starting_time + sample_count / rate. Do not silently shift clocks or resolve
the position 'centimeters' / conversion=0.01 combination by assumption.

Do not infer maze identities from position gaps, dates, coordinate clusters,
neural remapping, or POST evidence. Any epoch candidates need an independently
documented source assignment. PRE/POST sleep must be assigned relative to those
epochs, not merely to whichever position samples are present.

The script exits zero when all 15 payloads are verified and inventoried. This is
a technical completion code, not a scientific pass. The separate context-decoder
readiness gate remains false pending documented epoch/scaling review. A missing
label is not evidence that the experiment lacked distinct contexts.

## Next gate, only after metadata resolution

Validate independent CA1/CA3/RSC experience readouts on held-out RUN trials using
blocked folds and balanced contexts, with rate/time-drift controls. Freeze all
encoding and selection choices before examining POST content. A failed RUN
readout is a feasibility failure, not negative replay evidence. Only thereafter
test whether independent experience evidence remains when thinning a separate
CA1 population makes a trajectory label disappear.

The motivating study already addresses context-dependent communication and
replay-related reactivation: Gonzalez et al., 2026,
https://pmc.ncbi.nlm.nih.gov/articles/PMC13317024/.
The proposed addition concerns visibility under controlled recording coverage,
not first discovery of interareal replay, causal transmission, or consolidation.

## Execution

Run on gpuserver6000 from a clean committed worktree, with raw files outside Git.
Launch detached with a durable log and record PID/process start time. Save the
published asset metadata, schema inventories, per-session and unit summaries,
code/input provenance and output hashes. Do not amend a producer while it runs.
