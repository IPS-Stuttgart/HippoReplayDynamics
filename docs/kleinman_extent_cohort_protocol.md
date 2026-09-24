# Pre-evidence extent eligibility audit across the full RUN-qualified cohort

2026-09-24. Freeze before auditing these additional sessions. No reward contrast.

The six first-session integrated-likelihood bank passes only six matched-timing
and five unseen-timing animal/end screens. That result is not replaced. It also
does not establish which of the other 121 RUN-qualified sessions are usable.
This audit applies the identical estimator, simulation conditions and thresholds
to **all 127 RUN-pass sessions**, not the strongest-looking replacements.

The old 135-session RUN CSV is the fixed denominator; its eight failed sessions
remain explicitly ineligible. All six animals are retained in summaries even
if no session survives. No replay score or reward response selects sessions.

For the original six sessions, reuse exact known maps, event IDs and spike draws
from the first bank and require identical fitted estimates. For each additional
session fit the unchanged full-RUN directional maps and generate the same 2,880
conditions/repeats (the full 100/200/400-ms, 24/48/96-spike, static/linear/cosine/
pause-step bank). Use seed 20260923 and unique event IDs starting at 17,280, assigned
in animal/session order. These are additional recordings, not extra seeds on a
failed session in search of significance.

The 10-ms integrated conditional likelihood, unknown origin/end/directional map,
306 pre-support-exclusion hypotheses, and alternating-bin prediction are unchanged.
Apply exactly the frozen matched-timing, unseen-timing and duration-only screens
from `kleinman_integrated_extent_protocol.md` to each session and both ends.

A session is eligible only if **both ends pass all three screens**. Requiring
both ends preserves the manipulated-vs-unchanged-end comparison; a good single
end cannot rescue a session. Source/model failures count as ineligible, never
disappear. Record reasons and per-condition diagnostics. No threshold revision.

Report three distinct readiness statements:

1. Technical accounting: all 135 source sessions accounted for, all 127 RUN-pass
   sessions attempted, no silently omitted simulation or schema failure.
2. Reward-content cohort coverage: at least one eligible session in each of all
   six spike-recorded animals. This is merely minimum feasibility, not statistical
   power, map stability, real-event validity or biological support.
3. Drug/context coverage: each of 24 animal x drug x novelty cells retains at least
   one eligible session. Report separately; it must not be inferred from (2).

Retain all count-24 and off-template timing results. Session eligibility is defined
using RUN maps and synthetic truth only. Its association with drug/context/units
must be reported because an eligible subset may not represent the full experiment.
Passing sessions must still undergo real-event model-fit and map-drift checks
before content claims. No reward, drug or replay-rate contrast is scored here.

Save source/map/code/protocol hashes, all session scores/screens, compact cohort
summaries and an independent verification. Run on gpuserver4090 in persistent
tmux. Do not interpret a failure as a biological absence of replay.
