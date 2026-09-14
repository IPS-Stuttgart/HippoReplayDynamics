# RUN-balanced sampling: independent validation outcome

Status: technical audit passes, complete scientific validation FAILS.
This result does not complete the validated-remedy goal.

## Frozen comparison

Producer c0a8be70, protocol frozen earlier in 20571865. The producer was
committed before evaluation. PF development completed before external selection
outcomes. Neither selection nor thresholds were changed after those outcomes.

- Development: 4,069 candidates, eight PF sessions/four rats.
- External validation: all 741 frozen high-MUA candidates in five
  Blackstad--Moser open-field recordings, five animals.
- Equal disjoint halves from the identical cell universe in both conditions.
  Drop at most one seeded odd cell. No event abstention or continuity selection.
- Random split versus RUN-map pairing and RUN-only partition optimization.
  First-half rate maps; third-quarter calibration; fourth-quarter position test.
- Independent 20 ms flat-prior Poisson decoding. No temporal trajectory prior.
- Three splits; split 0 primary, two fixed sensitivities. Equal animals after
  equal sessions and events, not pooled draws or pooled event counts.

## Primary external result

| Measure | Random | Balanced | Difference |
|---|---:|---:|---:|
| Regional posterior TV | 0.276256 | 0.262636 | 4.93% reduction |
| Mean endpoint separation | 20.9888 cm | 19.6502 cm | 6.38% reduction |
| Mean A/B normalized entropy | 0.939861 | 0.937950 | -0.001911 |
| Held-out RUN true-position error | 51.9169 cm | 51.8897 cm | -0.0272 cm |
| Matched-map simulation error | 50.3211 cm | 50.2731 cm | -0.0480 cm |
| Map-drift simulation error | 51.4184 cm | 51.5009 cm | +0.0824 cm |

Both disagreement metrics improved in all five animals. Descriptive animal
bootstrap intervals for reductions: regional TV [0.00701, 0.02023], separation
[0.4937, 2.1835] cm. Five animals limit population generalization.

Nonetheless, both primary improvements missed the frozen >=10% threshold, and
the primary drift/gain simulation slightly worsened mean true-position error.
The +0.082 cm increase is small; the conclusion is not that balancing is harmful,
but that the complete predeclared non-worsening criterion did not pass.

Other splits reduced real separation by approximately 1.47% and 6.27%, and TV
by 3.42% and 5.33%. Neither sensitivity replaces the primary split.
PF development improvements were also small: primary TV 2.26%, separation 2.24%.

## Information limit exposed

External final-20-ms endpoints contain only 1.106 eligible spikes across both
halves on average. Both halves are silent at 28.0% of endpoints; at least one
half is silent at 85.9% in random partitions and 83.3% after balancing.
No primary external event gives both halves >=2 active units and >=3 spikes.
These are equally weighted animal summaries, not exclusion criteria.

Thus event-level high MUA does not imply a high-information endpoint after
RUN unit QC and population splitting. Low A/B separation can coexist with poor
known-position decoding and broad posteriors. The 20 ms RUN errors here must
not be compared as if they were the original 250 ms RUN quality-matching errors.
These results do not establish the cause of disagreement or true replay content.

## Verification

- Native Blackstad cache covers five animals and 741 unchanged events; stable
  integer unit IDs preserve native sorted-unit filename provenance and clocks.
- Separate auditor reconstructed all 207,528 real/RUN/simulated readouts across
  13 recordings, including all 4,810 candidate endpoints and RUN spike counts.
- The auditor independently reconstructed RUN-only pair assignments, population
  disjointness, shared simulated counts, means/widths/entropy/regional masses,
  known-position errors, and equal-session/animal summaries.
- Immutable input/producer/protocol hashes and PF-before-external order passed.
- Independent per-session/per-animal/overall aggregation and every gate were
  reconstructed; verified table hashes bind the non-rescoring report to the audit.
- 54 relevant tests passed; Ruff and git whitespace checks passed.
- Raw native spike sorting and biological replay truth were not validated.
  The producer and auditor share upstream encoding caches; this is not an
  independent re-estimation of every place field.

## Decision

This is partial evidence for a small population-balancing benefit, not a validated
remedy and not a destination-certification method. Preserve the previous
PF-to-Tanni failure: its apparent 20.8% agreement improvement was accompanied by
more diffuse posteriors and worse A-position accuracy.

The next useful diagnostic should target information in the exact decoded window,
not just event-wide MUA or global RUN quality. Any window-selection or
resolution change needs a new frozen protocol, known-position accuracy checks,
and independent evaluation. It must not silently replace the endpoint estimand
or discard difficult events and claim that all events became reliable.

## Artifacts

Server: gpuserver6000

- /mnt/seagate10tb/florianpfaff/balanced-content-remedy-20260914
- /mnt/seagate10tb/florianpfaff/blackstad-content-cache-20260914
- /home/florianpfaff/HippoReplayDynamics-content-stability-20260914

Primary tables: summary.csv, by_animal.csv, paired_sensitivity_summary.csv,
endpoint_information_summary.csv, gates.csv. Audit: independent_audit.json and
independent_reconstruction.csv. Figure: external_sampling_validation.png/pdf.
