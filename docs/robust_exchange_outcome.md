# Robust calibration-gradient exchange: stopped before replay

2026-09-15. Development result, not a validated remedy or independent-data
diagnostic. The active objective remains unmet.

## Question and frozen design

The previous joint exchange improved population agreement but did not preserve
accuracy outside its selected native calibration samples. This new rule used
all 20-ms observations in training RUN-Q3 parents, plus Poisson-gain1 and
conditional-count calibration simulations. A fixed hash split reserved 25% of
parent groups per true-class stratum; all choices were frozen before scoring
the reserved observations. Native 250-ms parents were never split across
training and validation. These are development data, not independent recordings.

The four original pairs, their cell counts, union and intersection were kept.
The original 1,836 candidate and 513 accepted-segment endpoints were not
modified or rescored. Proposed exchanges used analytical cell-weight gradients
and bounded integer optimization, then exact Poisson scoring. Each proposal
had to improve native training agreement J while preserving physical error and
Home Brier separately for both true classes, both populations and all three
calibration sources: 24 exact accuracy guards. No threshold was relaxed.

## Result

| Pair | Exactly evaluated proposals | Improved training J | Failed accuracy guards per proposal | Accepted |
| --- | ---: | ---: | ---: | ---: |
| Rat1/Open1 | 12 | 12 | 11-15 of 24 | 0 |
| Rat1/Open2 | 13 | 13 | 9-14 of 24 | 0 |
| Rat2/Open1 | 12 | 12 | 10-14 of 24 | 0 |
| Rat4/Open2 | 14 | 14 | 8-13 of 24 | 0 |

All 51 proposals improved J; none passed exact accuracy. There were 604 failed
nonworsening comparisons among 1,224 reconstructed training risk comparisons.
These point-estimate guards are not significance tests and do not establish
that each small deterioration is a population-level effect.

The linear approximations predicted every risk change to be negative; the
largest risk-normalized predicted change was -7.88493e-5. Numerical tests
verify the derivatives against finite differences. Thus correct infinitesimal
derivatives did not provide a reliable approximation to whole-cell exchanges.
The optimizer's feasible proposals were not actually accuracy-feasible.

No exchange was accepted in any of the four pairs. Internal validation had all
96 expected risk rows and numerically unchanged accuracy because it retained
the original populations. That is not a successful correction:

| Gate | Result |
| --- | --- |
| Original pairs have grouped support | pass |
| All 96 internal validation risks present | pass |
| Internal accuracy nonworsening | pass, unchanged populations |
| Changed populations in every rat | fail |
| Native validation J improves in every rat | fail |
| Ready for held-out truth preflight | fail |

No Q4, test-bank or replay scoring followed this failure. No independent
recording was evaluated or promoted. The method did not reduce instability.

## Interpretation and next requirement

Broader calibration did not produce an admissible exchange with this proposal
method. It is a failure of this bounded gradient-based search, not an exhaustive
nonexistence result and not a failure of data ingestion. The original endpoints,
populations, scientific target and accuracy safeguards remain in force.

The next exchange proposal mechanism would need demonstrated accuracy for
finite membership changes, not only derivatives at the baseline. It must still
pass the same complete held-out truth, fixed-endpoint discrepancy, entropy,
random-control and independent-recording requirements. Decreasing disagreement
alone, or making unchanged populations pass accuracy, does not meet the goal.

## Audit and provenance

- Server: gpuserver6000.
- Root: /mnt/seagate10tb/florianpfaff/robust-exchange-20260915.
- Producer commit: 23cb6386103c914088fd93ca8f699f6223738ef3.
- Producer runtime: 44.123 s; terminal exit 0.
- Producer manifest SHA256:
  559862c67f75b01ed19968b39a0776f78faba61f880d92ff2494648240a4eb29.
- Auditor commit: 2073777f949bcdd4202fc1f8793e5eb559d6ef37.
- Independent full-Poisson reconstruction: 51 proposals, 1,224 training
  accuracy measures and 96 validation measures; status pass.
- Auditor independently reconstructs group membership, legal cell exchanges,
  exact admissibility, training winner and final validation gates. It does not
  certify the search as exhaustive or prove absence of other feasible exchanges.
- 49 focused tests pass, including analytic-vs-finite-difference derivatives,
  a numerical positive-control exchange, validation isolation, nonvacuous gates,
  and rejection of falsified validation values even after hashes are regenerated.
- Ruff and staged whitespace checks pass.
- Results are compact tables and provenance; no raw recordings are copied.

Primary files: calibration/selection_summary.csv, calibration/gates.csv,
calibration/frozen_assignments.json, audit/candidate_reconstruction.csv,
audit/risk_reconstruction.csv, audit/partition_support.csv and
audit/independent_audit.json. Producer and auditor run as detached systemd user
services; both reached terminal exit 0.
