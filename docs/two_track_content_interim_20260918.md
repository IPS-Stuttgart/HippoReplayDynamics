# Experience-content audit: interim, not a completed hypothesis test

The public release is pinned to Dryad 10.5061/dryad.ksn02v76h, version 238435.
Read the protocol and machine-readable configuration before interpreting results.

## Input problems found before replay scoring

- RAT1_SESS2 and RAT4_SESS1 have byte-identical spike AND position files despite
  different animal/session labels. Both identities are quarantined.
- RAT2_SESS1 has four track entries; the two-track/re-exposure mapping remains
  unresolved. No arbitrary pair is chosen.
- RAT1_SESS1's supplied LFP and spike/position clocks do not overlap. No guessed
  clock shift is applied. Ripple-conditioned analysis is blocked for this session.
- Three other inspected LFP files start 30-42 ms after position. This is handled
  by global >=99% clock-range overlap and strict per-event coverage/no-gap checks,
  without interpolation across missing LFP or changing timestamps.

## RUN-only calibration

Seven unambiguous two-track sessions have 28-50 common RUN-qualified units and
finite held-out RUN results. Only RAT3_SESS2 and RAT5_SESS2 meet the predeclared
every-track/every-fold context accuracy >=0.8 gate. These form the strict primary
cohort. Other sessions are diagnostic only; this gate was not loosened after
seeing results. Conditional-position errors are good, but that does not imply
equally good context discrimination in every fold.

## First completed scientific session: RAT3_SESS2

All 50 jobs (five cell splits x five nested-subset repeats x two ripple strata)
completed without failures. This is ONE animal/session, not a dataset conclusion.
The bank has 517 detector-only rest candidates, 230 ripple-positive; 196 of the
ripple-positive candidates are POST.

At full versus half inference coverage, POST ripple sequence acceptance averages
8.061% versus 2.633%. These are averages over frozen subsets, not independent
event replicates. Cell-identity-randomized acceptance is 2.143% versus 0.980%.

Across the frozen subsets, 35 distinct POST ripple events are accepted at full
coverage and rejected at half coverage in at least one paired realization. Their
event-median, held-out-population signed context z scores average +0.4096 under
Poisson decoding and +0.7096 under the conditional-count sensitivity. This is
preliminary support for residual context information, not proof of ordered replay,
and not yet a replicated effect. Lost-support and lost-sequence-significance
groups overlap at the event level across different subsets.

The stronger bias claim is NOT established. The fixed-evaluation track-2
selection shift at half coverage averages +1.27 percentage points (median across
split/repeat estimates +0.086 percentage points), with typically only four accepted
half-coverage events per realization. Do not call this either systematic content
bias or evidence of no bias. Do not choose a new hypothesis from the larger
quarter-coverage shifts, which are especially sparsely supported.

## Verification

- 60 focused tests passed after the independent content verifier was added.
- All 521,566 bank count entries across 517 events match an independent raw-spike
  recount exactly.
- The technical 32-event pilot has 192/192 rows. Direct, independent reconstruction
  agrees for eight opportunity-valid sequence/null rows and every nonempty
  evaluation content/null calculation in that pilot to floating-point precision.
- An exact-boundary test caught a bin-index rounding issue at large absolute
  clocks. Explicit bin edges now replace division/floor indexing. The real bank
  had zero count discrepancies, as established by the raw recount.

## Remaining work

Finish the second primary session, construct the explicitly quality-labelled
diagnostic cohort, verify both, and complete the event-weighted selection/null
analysis. Test neural temporal prediction before calling residual sequenceless
track activation independently predictive replay. PRE/POST interpretation still
needs state, event-strength and unit-drift controls. Primary n=2 animals remains
a material limitation regardless of event count.

No manuscript biological claim is promoted by this interim update. Daniel Bush
is the collaborator; the Tirole/Takigawa/Bendor papers are independent prior work.
