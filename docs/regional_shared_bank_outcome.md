# Shared geometry-blind bank: outcome and claim boundary

2026-09-16. This supersedes the old forced-boundary-crossing generator as the
foundation for terminal-content, cell-participation and inferred-dynamics
diagnostics. It is a simulation-bank validation, not a regional-content result.

## Frozen run

- Server: gpuserver6000.
- Code: /home/florianpfaff/HippoReplayDynamics-content-stability-20260914
- Bank: /mnt/seagate10tb/florianpfaff/regional-blind-shared-bank-pf-20260916
- Audit: /mnt/seagate10tb/florianpfaff/regional-blind-shared-bank-pf-audit-20260916
- Reproduction: /mnt/seagate10tb/florianpfaff/regional-blind-shared-bank-reproduction-20260916
- Gaps: /mnt/seagate10tb/florianpfaff/regional-blind-shared-bank-population-gaps-v2-20260916
- Old-bank audit: /mnt/seagate10tb/florianpfaff/regional-terminal-proposal-audit-20260916

The original 1600 candidate identities remain in the cohort manifest. The common
100-ms-capable cohort contains 1139 count-trace templates across eight sessions
and four rats. The 461 shorter events are excluded explicitly, not padded.

Seven dynamics/scale strata x four segment durations x eight sessions yield
224 cells. Four replicas per calibration/null/validation prevalence configuration
give 8064 panels and 1,148,112 simulated event samples. These are repeated
simulations of 1139 templates, NOT 1.15 million independent recorded events.
Full bank size is approximately 330 MB. Separate seeds, spike identities,
native cell/tetrode identities, latent paths and geometric truth are saved.

## Checks

All bank technical checks passed:

- Original spike timestamps/counts, endpoint convention and active support.
- 56 representative full-session native-detector reruns preserve selected windows.
- Both labels and requested rounded prevalence quotas in every conditioning cell.
- 1,148,112 unique event seeds across phases and strata.
- 32,256 independently discretized saved-path dwell checks; maximum error 0.046 ms.
- Exact regeneration of spikes, paths and labels for 224 representative event seeds.
- Source/output hashes and complete raw cell-to-tetrode metadata.
- No unexpected near-degenerate positive occupancy cell at the predeclared 95%
  concentration check. Stationary and 20-ms-positive degeneracy are structural.

The focused regression subset passed 66 tests; Ruff passed for the new files.

## What changed

In the unconditioned geometric jump reference, pooling jump counts over the
eight sessions and three amplitude scales in 100-ms segments:

| Jump type | Fraction |
| --- | ---: |
| non-Home to non-Home | 93.42% |
| non-Home to Home | 3.18% |
| Home to non-Home | 3.17% |
| Home to Home | 0.23% |

These are simulated reference frequencies under the frozen generator, not
observed replay frequencies. Conditioning on content intentionally changes them.

Mean Home occupancy among positive segments, scale 1, averaging session means:

| Dynamics | 20 ms | 40 ms | 60 ms | 100 ms |
| --- | ---: | ---: | ---: | ---: |
| stationary | 100.0% | 100.0% | 100.0% | 100.0% |
| moving | 100.0% | 87.1% | 73.3% | 54.8% |
| jumping | 100.0% | 79.1% | 61.3% | 41.5% |

The 20-ms column must be 100% because the positive definition requires at least
20 ms cumulative Home occupancy. Longer moving/jumping segments no longer have
the old all-Home/forced-crossing truth degeneracy. Late crossings within 5 ms
remain explicitly labeled in every event and in the reference.

## Population gaps

The old bank's targeted-population signed gap was 7.75 pp for stationary,
8.07 pp for moving and 16.40 pp for forced late crossings. The last figure
describes an adversarial boundary-crossing case, not general jumps.

In the replacement bank, scale 1, conditioning on the 20-ms dwell label, the
unchanged 20-ms endpoint readout gives targeted-population signed gaps of
7.27/7.80/7.30 pp for stationary/moving/jumping. Whole-tetrode counterparts are
3.19/3.62/3.26 pp. Targeted pairs cover three rats; whole-tetrode pairs cover two,
with the original overlap/legacy eligibility limitations.

This is NOT a controlled old/new effect estimate: the new bank uses a common
longer-event cohort and a different minimum-dwell label, rather than just the
old instantaneous endpoint label. Longer conditioning Delta also changes
endpoint prevalence while the readout remains 20 ms; declining endpoint mass
or population gap is therefore NOT improved segment-content calibration.

## Important limits

The anchors use full-population smoothed decoder paths from 108 selected
clean-IMM events, not all 1600 MUA candidates and not known latent dynamics.
Raw and episode-level statistics are retained; their 4-ms resolution is explicit.
The source's shortened final-bin timestamp is handled separately from internal
uniform time steps. Moving and jump-amplitude scales are 0.5/1/2.

At the 2x amplitude scale, about 30.7% of requested jump distances differ from
their nearest supported realization by more than 4 cm (equal-stratum mean).
This is geometry-limited stress testing, not an exact 2x realized-distance model.
Requested and realized distances are saved; do not suppress this caveat.

The four replicas per configuration are development only. They cannot certify
90% interval coverage, a 5% false-flag rate, or a worst-case guarantee over a
continuous mixture space. Increasing independent replicas is needed for those.

At fixed total spike counts, common gain and absolute additive Hz are not jointly
identifiable; later cell diagnostics must use relative quantities or a separate
unconditional-count control. Existing population definitions still overlap and
must not be treated as conditionally independent tests by default.

No real-data prevalence estimate, endpoint recovery claim, IMM calibration,
cell-diagnostic validation or hc-11 transfer was performed here.

## Next use

Use this immutable bank for the terminal-segment calibration experiment first:
keep prevalence fixed within each conditioning Delta, vary dynamics mixtures,
stratify by occupancy, and report finite-bank uncertainty. Then reuse the same
saved spike identities for split-internal inferred-dynamics and cell-level
false-flag controls. Do not substitute the true generator label for inferred
type without labeling it an oracle diagnostic.

The preliminary benchmark directories are interface tests only and are not
pooled into this bank. The audit is a separate artifact referring to the frozen
bank manifest; the bank's generation manifest is intentionally not rewritten
after the independent audit, because doing so would invalidate its recorded hash.
