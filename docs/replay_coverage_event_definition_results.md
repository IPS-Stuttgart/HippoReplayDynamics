# Frozen Real Event-Definition Inputs

Status: preparation and reconstruction complete. Detector/window coverage-effect
decoding is still required; this is not completion of event-definition sensitivity.

## Provenance

All scientific computation used gpuserver6000 in
`/home/florianpfaff/HippoReplayDynamics-recording-coverage`.

- Initial frozen protocol/kernel: `8cba8b25`.
- Corrected unavailable-detector handling, preparation and audit:
  `489dc4cd4c6371d63638541f2252d0143dd81df6`, clean at invocation.
- Authoritative artifact:
  `/mnt/seagate10tb/florianpfaff/replay-coverage-event-definitions-all33-20260905-v2`.
- The original directory without `-v2` is a retained failed first pass. Two
  sessions had <60 s immobile baseline. The corrected pass retains their MUA
  windows and marks LFP detection unavailable instead of discarding sessions.
  The baseline threshold was NOT lowered. The 23 successful Tanni raw envelope
  caches were reused by verified hash and identical parameters; the auditor
  subsequently rehashed all consumed native arrays.
- All 90,132 window rows, source identities, direct spike counts, support flags
  and 5,665 many-to-many overlap edges verified. All 150 consumed Tanni HDF5
  arrays rehashed. All-channel raw refiltering reproduced the first Tanni
  session exactly; not every session was refiltered independently.
- 123 relevant regression tests passed, including 19 new event-definition tests;
  later focused tests also cover reporting, missingness and artifact linkage.

## Denominators

The 12,141 original high-MUA candidates are retained: 4,069 PF from eight
sessions/four rats and 8,072 Tanni from 25 sessions/five rats. Tanni includes all
arena sizes, NOT only the five large-arena recordings. The 90,132 rows are two
window variants of 45,066 source events, not 90,132 independent events.

PF supplies 15,782 native ripple rows across the full files, many outside the
RUN/position intervals used here. Their timestamps are retained before common
eligibility. Raw PF LFP is not available in these session directories.

Tanni supplies LFP with regular native 1,500 Hz clocks and 32 stored channels
whose literal IDs are members of the native CA1 channel map. The detector uses
150-250 Hz, absolute Hilbert envelope, Gaussian SD12.5 ms, channel averaging,
and mean/SD normalization during supported immobility. This is an amplitude
envelope z-score, NOT ripple power or the old robust-MAD z-score.

In 23 Tanni sessions, 17,143 mean-boundary episodes exceed peak z3 before the
duration/movement gates. The following two sessions lack the frozen >=60 s
immobile baseline and have unavailable, not zero, ripple counts:

- R2470 / 2018-08-10_14-08-50: 51.937 s.
- R2481 / 2019-04-29_16-26-33: 34.937 s.

All nine animals still have sessions with both detector definitions available.
Successful technical preparation does not imply ripple availability in all33.

## Common-Eligibility Counts

Primary Tanni ripple threshold z3. Available events must pass their source
duration criterion, fall within one supported RUN/position interval, and be
immobile throughout the entire window. Fixed windows are peak-centered 200 ms,
never clipped. These counts are candidates, not decoded or significant replay.

| Dataset | Detector | Core windows | Fixed 200 ms windows | Available sessions |
|---|---|---:|---:|---:|
| PF | Original high MUA | 4,001 | 3,707 | 8/8 |
| PF | Native ripple table | 2,581 | 2,567 | 8/8 |
| Tanni, all arenas | Original high MUA | 5,224 | 5,170 | 25/25 |
| Tanni, all arenas | New CA1 LFP ripple-like | 2,635 | 2,566 | 23/25 |

Threshold sensitivity is frozen, not tuned to decoded outcomes: Tanni z4 gives
1,540/1,503 core/fixed windows, and z5 gives 986/968.

The common position-speed calculation and whole-window gate intentionally
expose a difference from legacy source selection. PF: 4,057 MUA cores have
tracking support, 4,060 have an immobile peak and 4,001 are immobile throughout.
Tanni: corresponding counts are 8,019, 5,988 and 5,224. Original source MUA
inclusion used different speed processing/window rules. The source cohorts
remain unchanged on disk; do not silently treat the new eligible cohort as a
reproduction of the older 1,264/226 trajectory counts.

## Detector Overlap

For eligible cores in sessions with available ripple detection:

- PF: 2,295/4,001 MUA windows overlap an eligible native ripple (57.36%).
- Tanni: 449/5,193 MUA windows overlap an eligible new ripple-like episode (8.65%).

These are pooled unique-MUA fractions, not equal-animal estimates. The figure
instead displays per-animal means of available-session overlap fractions.
Many-to-many edges, each source identity and overlap durations remain in the
output. Two-detector windows are not independent replications. Excluded ripple
episodes (duration, movement, clock) are not part of these overlap denominators.

Low Tanni overlap is a detector discrepancy requiring inspection, not a
biological finding or evidence that one detector is accurate. Native clock
preservation and regular sampling do not independently prove hardware timing
alignment, correct LFP event morphology or absence of artifacts. Likewise,
technical rail/flat-channel screening does not establish SWR authenticity.

## Next Required Work

1. Use these exact frozen windows and source maps for paired full/half-cell
   decoding, keeping detector, window, overlap, estimator and per-bin support
   strata separate. Preserve zero-denominator and unavailable-session rows.
2. For speeds, require every intermediate 5 ms support bin to be valid between
   non-overlapping 20 ms windows. The old real-subsampling helper checked only
   endpoints for this speed sensitivity; do not silently reuse that helper or
   present its filtered speed summaries as having this stricter protection.
3. Inspect representative LFP/MUA traces before treating new Tanni ripple-like
   candidates as a validated assay. Keep within-dataset detector contrasts
   separate from differences between PF native and Tanni re-detected events.
4. Compare geometric continuity to established shuffle-significant replay
   baselines. A candidate-count or overlap table alone cannot validate replay
   continuity, speed, or biological equivalence.

The main study objective and all remaining completion requirements are unchanged.
