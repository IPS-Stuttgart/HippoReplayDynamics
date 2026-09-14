# Native audit runtime correction

The original `error-trained-context-pf-audit-20260914.service` was deliberately
stopped, along with its waiting finalizer, after identifying a redundant native
spike recount. It was live and consuming CPU, not stuck or failed; this was not
a restart due to an observation timeout. Its partial records and log remain in
`error-trained-context-content-20260914/pf-audit` and the corresponding server
log. Rat1/Open1 had passed. There is no completed audit result for this run.

The old auditor called `recount` separately for each event; that helper scans
and sorts each cell's native spikes on every call. Concatenating the original
interval arrays allows the SAME helper to count all intervals in one call per
native source, preserving exact half-open interval edges, duplicate spike
times, counts, original event order, every source, all splits and all posterior
checks. No decoder, features, fitted tree, selection, outcome or gate changes.

Regression tests compare batched and separate counts at exact boundaries and
duplicate timestamps, and the full synthetic producer/auditor roundtrip checks
every posterior and metric. A call-count assertion confirms one recount per
native source rather than per event. Independent model refitting is retained.

The optimized audit uses a NEW output directory `pf-audit-v2` and service
`error-trained-context-pf-audit-v2-20260914.service`. The report must reference
that completed audit. No prior partial directory is overwritten or called a pass.
The old finalizer must not be restarted because it references the partial audit.
