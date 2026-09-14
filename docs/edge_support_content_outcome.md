# Edge-support content benchmark: outcome and decision

Date: 2026-09-14. Status: completed, independently audited, **no validated
original-endpoint remedy and no validated A-only activity diagnostic**.
The overarching content-stability goal remains open.

## Frozen experiment

Protocol/inputs commit `540c1c99`; measurements `3d766066`; independent
auditor/reporter `e2efa438`, auditor boundary-arithmetic fix `86136544`.
All scientific computation ran on gpuserver6000 in detached user services.

- PF: 8/8 sessions, four rats, 1600 hash-selected candidates from 4069.
- hc11: 8/8 sessions, four rats, 1600 hash-selected POST MUA candidates from
  57460. These are not automatically curated replay, ripple or NREM events.
- Selection: 200 candidates/session, SHA256-derived rank seed 20260914, selected
  without spike counts or decoder outcomes before map fitting.
- Maps and unit QC use only first-half RUN spikes, including quarter-to-quarter
  field stability within that training half. Full-RUN tracking can define grid
  geometry. No previously full-RUN-QC-selected mask is reused.
- Fixed 8-cm occupied grid, independent 20-ms Poisson windows on a 5-ms grid,
  uniform spatial prior, no temporal model. Two equal disjoint populations,
  three fixed splits; split0 primary, the others sensitivities, not extra rats.
- Same cells, maps and candidate windows across policies. When a policy abstains,
  its baseline is calculated from exactly the same retained events, with the
  full denominator and abstentions also reported.

Tested policies: original endpoint; last window with two pooled A+B spikes;
last with >=3 A spikes/>=2 A cells; last with that support in both populations.
The two-spike baseline is motivated by Pfeiffer & Foster (2013) edge trimming,
not a reproduction of their full trajectory/shuffle procedure. Joint and pooled
timing use B counts and are not A-only predictive controls.

## Primary real-candidate results

Values average events within sessions, sessions within rats, then rats equally.
Changes are selected minus raw at the same retained events. Lower separation/TV
means better agreement, but does not by itself mean better spatial accuracy.

| Dataset | Policy | Available | Earlier by, mean ms | A/B separation change, cm | Regional TV change |
| --- | --- | ---: | ---: | ---: | ---: |
| PF | Pooled >=2 spikes | 100.0% | 1.1 | +2.184 | +0.0242 |
| PF | A supported | 97.8% | 12.7 | +2.457 | +0.0147 |
| PF | Both supported | 94.1% | 21.1 | +0.255 | -0.0077 |
| hc11 | Pooled >=2 spikes | 97.8% | 11.4 | +8.819 | +0.1040 |
| hc11 | A supported | 53.9% | 41.2 | +16.832 | +0.1762 |
| hc11 | Both supported | 26.7% | 55.3 | +19.533 | +0.1872 |

For hc11 A-supported timing, separation changes 30.24 to 47.07 cm and TV .3392
to .5154. Both increase in all four rats. A normalized entropy decreases .1440
and B entropy decreases .0360: the posteriors become sharper, not more diffuse.
Descriptive four-rat bootstrap intervals for these changes are +13.91 to +20.90
cm and +.1405 to +.2046 TV, respectively. They are not high-powered population
confidence statements.

Availability is heterogeneous. hc11 A-supported availability is 92.75% Achilles,
47.50% Buddy, 44.00% Cicero and 31.25% Gatsby. Thus the >50% pooled mean does not
pass the frozen requirement of >=50% in at least three of four rats. Joint
support retains 76.25%, 10.50%, 13.33% and 6.75%, respectively.

Neither secondary partition rescues the result: hc11 A-supported separation
increases +17.41/+17.05 cm and joint-supported separation +20.78/+23.01 cm. PF
secondary separation changes are also positive for every trimmed policy.

## Accuracy and time-target dissociation

There are five known-position sources: q4 held-out RUN and stationary, moving,
gain-shifted moving, and late-jump simulations. Simulation controls preserve
each candidate's entire whole-population 5-ms count timecourse before splitting
the one generated population into A/B. They do not preserve native noise
correlations. Paths use occupied-grid shortest paths, not independently verified
maze topology; first-half place fields are the generative encoding assumption.

For hc11 A-supported trimming:

- q4 RUN: A original-time error decreases 12.50 cm; B decreases only .19 cm.
- Moving simulation: A selected-time error decreases 14.00 cm, but original-time
  error decreases only 4.89 cm. B selected-time error decreases 6.10 cm while
  original-time error increases .13 cm.
- Late-jump simulation: A selected-time error decreases 51.60 cm and B decreases
  43.51 cm, but original-time errors **increase 20.70 and 19.26 cm**. The selected
  and original truth locations differ by 116.57 cm on average.

The late-jump control similarly fails for PF and every trimming policy. It is
not a model of ordinary replay frequency: it is an explicit falsification of
the assertion that an earlier estimate recovers the unobserved endpoint.

Crucially, even ordinary single-path controls can show more disagreement after
trimming. In hc11 stationary simulations, A-supported trimming increases A/B
separation 12.16 cm while decreasing A/B true-position errors 11.55/2.58 cm.
This demonstrates why population agreement alone is an inadequate objective:
diffuse, low-activity posteriors can agree more while localizing less accurately.
It does NOT establish multiple simultaneous real replay trajectories.

## A-only activity flag does not validate

The frozen flag was original A activity <3 spikes or <2 active cells. It did
not reliably predict greater independent B error across external animals:

- hc11 RUN: flagged-minus-unflagged B error -4.52 cm, positive only 2/4 rats.
- hc11 moving simulation: +1.65 cm, positive 2/4.
- hc11 gain-shifted moving: -.91 cm, positive 2/4.
- hc11 stationary: -14.95 cm, positive 0/4.

Some comparison strata are small (only 76 finite unflagged hc11 RUN endpoints
across the eight sessions). Conditional-total simulations also couple A/B
counts and are not evidence for physiological population independence. These
results reject a claim of a general useful error predictor from that count flag;
they do not justify tuning the support threshold after seeing hc11 outcomes.

## Verification and provenance

Independent audit v2 passed all 16 sessions and 229164 readout rows, recounting
213006 native 5-ms windows and checking 123880 RUN truth base windows. Real
candidate base-count reconstruction covered 7507263 cell-by-window counts.
Counts were reconstructed from cached unbinned observations, not the producer's
window-count helper; posterior normalization, errors, policy selection and
partitions were reconstructed by separate code.

An additional source/summary verification checked 80 source/cache file hashes,
all 3200 frozen IDs and timings against independent hash ranking of full source
catalogs, and 1152 paired summary rows plus equal-rat availability directly from
audited readout tables. It did not independently regenerate the simulation
random paths or bootstrap intervals; these are separately identified limits of
the audit, not claimed validation.

The initial audit failed because the independent auditor reassociated
`3 * ((x-min)/extent)` as `(3*(x-min))/extent`. At floating-point regional
boundaries this reclassified a grid column (22/535 bins in Rat4/Open1). Correcting
the auditor to preserve the frozen region convention resolves the discrepancy.
A decimal-origin grid regression test covers this case. The failed audit is
preserved; no measurement, selection or region rule was changed.

The final relevant suite passed 68 tests, including the new decimal-boundary
regression. Ruff passed for both producers, the auditor, reporter and their
tests. Optional Matplotlib Axes3D/backend and SciPy deprecation warnings
do not affect the 2D numerical outputs. The compact figure was visually checked.

## Decision and relation to the original result

No policy passes the original-endpoint remedy gates. Do not tune the activity
thresholds, select a favorable secondary partition, or describe earlier-time
accuracy as endpoint recovery. This is a bounded negative intervention test.

These raw-candidate endpoint assays are NOT the accepted trajectory endpoints
in the earlier matched-population Home analysis. They neither explain away its
+6.87 percentage-point accepted-endpoint contrast nor establish a correction
for it. A successful remedy must ultimately confront that fixed-content
population-sampling contrast, not only an easier readout or agreement statistic.

Next methodological direction: retain the original time/content target and
test an encoding-aware observability or aggregate-content calibration approach.
Any proposal must preserve nonuniform known content, avoid agreement through
flattening, be frozen before independent evaluation, and demonstrate both
useful availability and recovery under model-mismatch controls. This direction
is not yet implemented or validated; the goal remains active.

## Artifacts

Server root:
`/mnt/seagate10tb/florianpfaff/edge-support-content-20260914/`

- `pf-inputs/`, `hc11-inputs/`: frozen IDs and training-only encoders.
- `pf/`, `hc11/`: full measurements and source audit arrays.
- `audit/`: preserved first, failed auditor run.
- `audit-v2/`: corrected independent reconstruction, passed.
- `report/`: CSVs, Markdown, PNG/PDF and manifest; no rescoring.

Local compact copy:
`/mnt/c/Users/emper/Documents/codex/2026-09-14/edge-support-content-benchmark/`

No raw dataset, EEG or source spikes are included in the compact report copy.
