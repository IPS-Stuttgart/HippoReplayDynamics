# Pinned camera-conversion diagnostic

This is an input diagnosis for replay-order versus subsequent RUN coordination,
not a source-parser repair, cohort amendment or biological analysis.

The raw-clock census found 50 backward pulse steps across 36 recorded-global-clock
camera rows, with monotonic frame times and matching pulse counts. The pinned
SpatialAutoDACQ source uses `closest_argmin`, which sorts a lookup copy and returns
the original pulse indices. Our original biological input parser deliberately
rejects backward pulse chronology. Those rules remain unchanged.

This diagnostic executes only the two exact pinned clock functions plus the
already pinned geometry functions. It retains and counts all raw pulse defects,
checks frame chronology, prohibits upstream pulse-count truncation, and compares
every deposited ProcessedPos row with the reproduced output at the original
5-ms and 0.05-cm tolerances. Nonmonotonic converted frames remain unresolved.
No time offsets are fitted. No spikes, LFP or private Settings are read.

The parent source inventory supplies all 25 original recording identities; files
cannot be substituted based on results. The original protocol is supplied for
the source commit, documented clock units, geometry settings and unchanged
reconciliation tolerances. It is not overridden to admit backward timestamps.

Even exact source reproduction does not prove hardware synchronization, the
meaning of a pulse reversal, or which surrounding frames are scientifically
usable. It only tests whether the deposited result is consistent with the
published conversion. Every output has `eligible_for_neural_analysis=false`.

A subsequent interval-based amendment would need independently justified
invalid-region boundaries, no bridging of those intervals, and independent
reconciliation of every admitted timing/coordinate row. It must be frozen before
new event-order or coordination associations. This diagnostic authorizes none
of those changes.
