# Independent dataset readiness before another content diagnostic

Frozen before running all-session support or decoding analyses. The preceding
PF-trained RUN-risk diagnostic failed Tanni transfer. Its outputs and gates are
unchanged. This preflight seeks an independent validation resource, not an easier
replacement outcome or permission to mark that failed diagnostic successful.

Inspect all session_info.mat files in the existing Kleinman/Foster 2025 release
on gpuserver4090. No download or replay scoring. Preserve experiment, subject,
session, drug and novelty labels. Do not assume animal identity across experiments.
The prospective primary stratum is Experiment 1 control animals with drug=0.
All other strata remain visible; no picking sessions by an eventual decoder result.

Count units by (tetrode, cluster), since cluster IDs need not be globally unique.
Inventory missing spike files, raw unit counts, native sdes/ripple windows,
spike/behavior clock coverage and unresolved position timestamp conventions.
No automatic shift/truncation is justified by a one-sample length mismatch.

For each native event, inspect the last 20 ms without modifying its endpoint.
Compare a peak-centered 20 ms window only as a separate descriptive phase check.
Require the full window inside the native interval, and bracketed velocity
samples with <100 ms gaps and |speed|<5 cm/s for an immobile-support label.
Use three deterministic equal disjoint cell halves (seed 20260914, compound unit
IDs, split0 primary). Report both-halves support at >=3 spikes and >=2 active
units each, but do not use this to redefine or count validated replay events.

Raw units have not passed place-field/cell-type QC. Their support is an optimistic
upper bound on usable information after encoding validation. No session is
decoder-ready until position timing, encoding coverage and held-out known RUN
errors are separately checked. No diagnostic coefficient, threshold, selection
tier or claimed remedy is trained or chosen during this readiness inventory.

All outputs include source hashes, code commit, command and per-session failures.
Run detached on gpuserver4090. A missing or small usable cohort is a data-readiness
finding, not negative biological evidence or a positive remedy result.

Reader revision after the first inventory: Experiment 2 does not supply `novel`.
Keep this label missing, with missing-label groups retained in all summaries.
This is a schema correction, not a change to the prospective primary stratum.
Sessions with genuinely nonmonotonic velocity clocks remain excluded and visible;
do not silently sort or shift their timestamps. Preserve the initial inventory.
