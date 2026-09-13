# PF Matched-Population Content Intervention

## Outcome

This bounded experiment found substantial differences in decoded Home content
between equally sized populations that passed the frozen matching and independent
RUN-confirmation gates. It supports a methodological statement about global
decoder QC, not a biological claim about replay planning.

Completed on gpuserver6000 in the detached user service
`pf-matched-population-content-20260913`, exit status 0. Producer commit `05ebb0f3`.

Results: `/mnt/seagate10tb/florianpfaff/pf-matched-population-content-20260913`

## What Was Held Fixed

- Original PF MUA candidates, original timestamps, spatial support and grid.
- Exact cell count within every population pair.
- Early RUN mean firing rate within 10%, median field stability within 0.05,
  median half-peak field area within 20%.
- Global RUN posterior-mean median decoding errors within 2 cm, p75 within 5 cm;
  both median errors <=20 cm and p75 <=40 cm.
- These RUN-error gates had to pass both a matching block and a later confirmation
  block. Subsets were frozen before confirmation and never replaced if they failed.

Population choices used RUN only. Replay was decoded independently in 20 ms
windows with a flat prior and no HMM/momentum model. The endpoint of each fixed
candidate and the continuous-segment endpoint of each previously accepted event
were fixed from the earlier full-population benchmark, not reselected here.

## Matching Feasibility

| Population comparison | Confirmed sessions | Rats | Fixed candidates | Previously accepted trajectories |
|---|---:|---:|---:|---:|
| Targeted high/low Home tuning | 4/8 | 3 | 1,836 | 513 |
| First quality-matched random pair | 5/8 | 3 | 2,209 | 621 |
| Whole-tetrode high/low tuning | 3/8 | 2 | 1,235 | 337 |

Targeted matches: Rat1/Open1, Rat1/Open2, Rat2/Open1, Rat4/Open2. No targeted
pair qualified on the matching data in Rat2/Open2 or either Rat3 session.
Rat4/Open1 qualified initially but failed confirmation. All failures are retained
in `population_match_status.csv` and excluded from replay comparisons.

Whole-tetrode matches: Rat2/Open1, Rat2/Open2, Rat4/Open2. These kept complete
tetrodes among early-eligible units; the two populations within each pair still
had exactly equal unit counts. They were selected to maximize tuning contrast
among matching candidates, NOT drawn to estimate typical tetrode-sampling bias.

## Main Content Readouts

Values average events within session, sessions within rat, then equally weight rats.
Home mass is posterior probability within 20 cm of the inferred Home well;
these percentages are NOT counts of true Home replay events.

| Comparison | High-tuning population Home mass | Low-tuning population Home mass | Difference |
|---|---:|---:|---:|
| Targeted, fixed candidate endpoints | 11.38% | 2.80% | +8.58 pp |
| Whole tetrodes, fixed candidate endpoints | 8.18% | 4.10% | +4.08 pp |
| Random, fixed candidate endpoints | 8.93% | 8.33% | +0.60 pp |

The targeted contrast is positive in all three retained rats, with descriptive
rat-bootstrap interval [+5.82, +11.39] pp. For the frozen accepted trajectories
at their fixed continuous-segment endpoints, the contrast is +6.87 pp, interval
[+5.34, +9.81]. Thus it is not exclusively a property of rejected MUA candidates
or of noise after the continuous segment.

Whole-tetrode contrasts are +4.08 pp (candidates) and +2.41 pp (accepted segments),
but involve only two rats. Do not interpret their narrow discrete bootstrap
intervals as evidence of broad population precision.

## Compare Controls on the Same Sessions

The three overall cohorts differ, so direct subtraction of their pooled headline
values is not a clean family comparison. A non-rescoring paired-session summary
compares each targeted/tetrode effect with the random effect on identical sessions:

| Targeted family | Its contrast | Same-session random contrast | Paired excess |
|---|---:|---:|---:|
| Targeted, candidate endpoints | +8.58 pp | +1.41 pp | +7.17 pp |
| Targeted, accepted segments | +6.87 pp | +0.92 pp | +5.94 pp |
| Whole tetrodes, candidate endpoints | +4.08 pp | +0.43 pp | +3.65 pp |
| Whole tetrodes, accepted segments | +2.41 pp | -0.14 pp | +2.55 pp |

This descriptive paired summary was added during the audit to avoid confusing
differences in cohort membership with differences between population types.
Whole-tetrode and random families can have different total unit counts even on
the same session; exact count matching applies within each high/low pair, not
between those families. Their excess is not a pure tetrode-grouping effect.

## A Concrete Matched Example

Rat4/Open2, targeted populations:

| Readout | Higher Home tuning | Lower Home tuning |
|---|---:|---:|
| Cells | 68 | 68 |
| Global confirmation RUN median error | 12.37 cm | 11.91 cm |
| Local confirmation RUN median error within 30 cm of Home | 10.74 cm | 18.43 cm |
| Replay candidate endpoint Home posterior mass | 11.01% | 2.48% |

The local/global distinction matters: global error was matched, but local Home
accuracy was not. All four targeted pairs had better local Home RUN accuracy in
the Home-rich population. This provides a plausible spatial-decoding explanation;
it is not a mysterious biological change despite universally equivalent decoders.

## Sensitivity and Remaining Confounds

Using first-half-only maps for replay instead of the frozen full-RUN maps gives
targeted differences +9.00 pp for candidates and +7.30 pp for accepted segments.
Whole-tetrode differences are +4.31 and +2.30 pp. The direction therefore does not
depend on using full-RUN maps for the replay comparison.

Matching RUN firing rates did not match actual endpoint replay spike support.
Targeted populations averaged 3.71 versus 2.91 spikes per endpoint bin. Differences
in observed spikes may contribute; this study does not isolate spatial tuning
from every event-level signal-to-noise effect.

Original cell eligibility was established with the full-RUN QC pipeline. The new
subset-selection maps, stability and firing-rate descriptors were estimated in
early RUN, and error matching/confirmation used later disjoint temporal blocks.
This is conditional on the frozen cell eligibility, not an entirely new blinded
recording-level validation.

## Validation

- Independent audit reconstructed 84 RUN population/block error summaries,
  including confirmation failures, and 528 replay endpoint readouts.
- Verified source hashes, unchanged subset IDs before/after confirmation,
  exact equal cell counts, descriptor tolerances, intact tetrode groups,
  fixed event denominators/anchors and equal-rat aggregation.
- 21 pairs were frozen before confirmation; 12 passed. Failed populations were
  not replaced and contributed no replay rows.
- 48 relevant tests pass; Ruff passes. No ongoing evaluation remains.

## Defensible Conclusion

Equal cell counts and similar GLOBAL RUN decoding accuracy do not guarantee
stable regional replay-content estimates under this common-grid PF analysis.
Spatially different recording samples can produce substantially different Home
posterior readouts on unchanged events. A weaker version also occurs for complete
tetrode subsets, although external replication is limited.

This does not establish that the original PF goal-planning result is an artifact.
Home annotations remain inferred, sampling was contrast-targeted, matching failed
in half the targeted sessions, and only three rats contributed. The practical
implication is to inspect region-specific decoder recovery and sampling robustness,
not just report one global RUN error number.
