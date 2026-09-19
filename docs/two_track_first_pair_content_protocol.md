# RAT2 first-pair content diagnostic

Frozen after RUN-only validation and before inspecting any candidate-window
sequence or content results in this recording segment.

## Scope and Claim Boundary

Analyze RAT2_SESS1's chronological first and second supplied position epochs.
Both have separate source coordinates and pass the unchanged RUN preflight.
All samples from the third epoch onward are excluded. The two experiences
mean first versus second observed RUN, not inferred identities of epochs 3/4.
The original study code's first-pair POST definition, the 125-minute source
table and the supplied LFP endpoint corroborate this bounded view. They do
not recover the missing session-specific re-exposure lookup.

Consequently this is a separate `first_pair_source_scope_diagnostic_only`
cohort, not a retroactive addition to the frozen primary cohort. Never count
later exposures as another rat or independent session. No primary-threshold
change, alternate epoch-pair search, map fitting on rest spikes, or promotion
based on a favorable replay outcome is allowed. Session-specific provenance
remains needed before any primary-cohort claim. Existing primary results,
configuration, banks and manifests are immutable.

## Frozen Experiment

Use the unchanged `two_track_content_configuration.json` (seed 20260918):
20% detector-only cells; five inference/evaluation splits of remaining cells;
five nested subsampling repeats; fractions 1, .75, .5, .25. Evaluation cells,
RUN maps and candidate windows are fixed across sampling levels. Decode 20 ms
bins with independent Poisson likelihoods; 499 whole-bin and 499 independent
field-shift nulls, each p<.025; original active-cell and nonempty-bin criteria.
Retain the existing cell-identity-randomized inference controls and independent
evaluation map-swap/identity controls. No parameter is selected using results.

Candidate detection uses only reserved detector cells and the original MUA
kernel in PRE/POST remote immobile rest. Bound POST after epoch 2 and before
epoch 3. Keep the actual source LFP timestamps unchanged, classify supplied
envelope peak >=3 as ripple-positive only for fully supported windows, and
retain supported MUA-only candidates as the prespecified secondary stratum.
The common timestamp support is used per event; no edge extrapolation or
inferred offset is allowed. Whole-record overlap must remain >=99%.

The primary diagnostic contrast is the original full-to-half loss in POST
ripple candidates: does lost sequence acceptance retain fixed held-out-cell
context support, and does the independently measured selected-content mixture
change? Separate losses, gains and retained windows; sequence track labels
are not ground truth and mean posterior mass is not replay prevalence.

## Verification and Execution

Recount every bank spike histogram from raw inputs; verify source preflight
and LFP hashes, map identity and disjoint populations. A deterministic capped
pilot is technical validation only, never a biological estimate. Verify its
Poisson, correlation and shuffle calculations independently, then run all
5x5 repeats in ripple and MUA-only strata regardless of pilot signs.

Use detached server jobs, bounded concurrency, explicit terminal records and
stable PID/start-time handles. Do not restart on connection timeout.
Report event-level summaries and temporal-block uncertainty, not independent
copies for each cell split. Keep the matched-rate identity null and future
held-out prediction as separate requirements before a replay-content claim.
No primary biological confirmation can be inferred from this diagnostic alone.
