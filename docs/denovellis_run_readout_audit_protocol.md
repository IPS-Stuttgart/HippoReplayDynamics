# Bounded Denovellis RUN-only audit

This audit diagnoses the failed RUN prerequisite without changing the original
post-error protocol, lowering thresholds, decoding replay, or testing behavior.
The four RUN maps and processing variants are frozen in the adjacent JSON before
new scoring. Cases illustrate previously recorded QC categories; they are not a
representative cohort and cannot establish a decoder success rate.

For each map, train on the first 70% of chronological RUN and independently decode
at most 300 equally spaced held-out moving windows per unique arm. Keep silent
windows. Use 20 ms non-overlapping windows, a 3 cm graph grid and a uniform prior
over occupancy-supported bins. Report the unchanged .80 balanced-accuracy/.75
arm-recall thresholds only as diagnostic comparisons, not cohort qualification.

Isolate spike-to-position interpolation, exact-timestamp deduplication, and a
gapped 1D spatial KDE. The latter is inspired by the author code, not an exact
reproduction: the authors' RUN HMM segment projection and grid construction are
not reproduced. A separate numerical check invokes only the pinned 0.7.7.dev0
Poisson mark equation on identical fitted intensities. It tests arithmetic, not
complete preprocessing or equivalence of biological observation models.

Audit all original target epochs for deterministic latest-matching-context
preceding RUN selection, without reading replay scores or choosing by decoder
performance. Preserve missing metadata, context mismatches and tetrode overlap.
Count newly metadata-eligible targets as potential, never as QC-qualified trials.

Write checkpoints, durable job logs, terminal status, hashes, source pins, window
accounting, confusion matrices, support distributions and deterministic posterior
examples. Bound runtime to two hours. No automatic extension or biological run.
The original no-go remains in force; a later revised protocol would require full
chronological validation and frozen eligibility before any association test.
