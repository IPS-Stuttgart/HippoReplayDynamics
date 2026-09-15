# Finite whole-cell exchange: calibration gate failed

2026-09-15. Development-only result. No independently validated remedy or
diagnostic was demonstrated. The active objective remains unmet.

## What changed

The preceding gradient rule proposed 51 exchanges that all improved agreement
but each violated 8-15 of 24 actual accuracy measures. This experiment replaced
infinitesimal derivatives with measured costs of every legal whole-cell swap.
The same native/simulated training banks, atomic parent partitions, original
four populations, cell counts, union/intersection and accuracy safeguards were
retained. The previously fixed 1,836 candidate and 513 accepted-segment endpoints
were not changed or rescored. No new observations were selected for evaluation.

The full original single-cell neighborhood was enumerated on RUN-Q3 training,
cal_poisson_gain1 training and cal_conditional training. Each swap required
native J improvement and nonworsening physical error and Home Brier in both
true classes, for both populations and each source (24 point-estimate guards).

## Single-cell result

| Pair | Legal swaps measured | Native J improves | Minimum failed risks among improving swaps | Admissible |
| --- | ---: | ---: | ---: | ---: |
| Rat1/Open1 | 3,025 | 1,301 | 7 of 24 | 0 |
| Rat1/Open2 | 5,041 | 2,416 | 6 of 24 | 0 |
| Rat2/Open1 | 2,025 | 1,248 | 8 of 24 | 0 |
| Rat4/Open2 | 2,209 | 1,276 | 6 of 24 | 0 |
| Total | 12,300 | 6,241 | | 0 |

No single swap preserved all 24 risk measures, even among swaps that did not
improve J. Complete enumeration concerns this specific single-cell neighborhood
and fixed calibration protocol; it is not a universal impossibility result.

## Joint search and validation

Measured single-swap changes were used as approximate costs for disjoint joint
exchanges of 2, 4 or 8 cells. Seven feasible limited-run proposals were returned,
all for Rat1/Open2. Every one improved J but failed 1-6 exact accuracy guards.
All were rejected. The other searches returned no feasible incumbent within
their frozen constraints or resource limits. In detail, 4 solves reported
infeasibility of the summed-cost optimization and 7 limited solves returned no
feasible incumbent; 7 limited solves returned the proposals that were exactly
evaluated. Neither an infeasible approximate problem nor a missing limited-run
incumbent proves that all actual joint exchanges are infeasible.

All four final populations remained unchanged. The 96 internal validation
accuracy measures therefore remained equal to baseline. The required gates
for population changes and validation improvement failed, as did progression
to held-out truth testing. An unchanged population passing accuracy is not a
successful correction. No Q4, test-bank, replay or independent-recording
evaluation followed this failure.

## What this rules out, and what it does not

The failure is no longer explained merely by an inaccurate gradient ranking or
by checking only the first few single swaps: all 12,300 original single swaps
were measured. Agreement improvement alone was common but was not sufficient.

Joint optimization is still bounded and uses an additive approximation before
exact checking. The seven returned joint proposals failed fewer guards than
the previous gradient proposals, but these are different selected proposals;
this is not an independent validation of a generally superior search method.
It also does not justify weakening accuracy safeguards to accept them.

These are strict point-estimate safeguards, not tests establishing significant
population harm for each individual increase. Native calibration and internal
validation are development data from the same sessions, not external validation.

Any further exchange method must address joint interactions using actual
accuracy measurements, then pass the unchanged complete truth, replay,
entropy, equal-budget random-control and independent-recording requirements.
The data do not establish that a remedy cannot exist.

## Numerical audit and provenance

- Server: gpuserver6000; four CPU workers with one BLAS thread each.
- Root: /mnt/seagate10tb/florianpfaff/finite-exchange-20260915.
- Producer commit: 06b2fe4eb96ba5ced85a197756954d27a7396891.
- Runtime: 622.187 s; detached service reached terminal exit 0.
- Producer manifest SHA256:
  e63c07e8d8456ad4aab975d7a33f8e7df7e59bc93a41be52296ad8d9d92793cb.
- 33 single swaps per session were independently reconstructed: 32 fixed
  hash-selected swaps plus the minimum-J swap; every admissible single would
  also have been reconstructed, but there were none. Maximum absolute numerical
  discrepancy was 1.635e-13. This is a sampled independent numerical audit of
  the complete tables, NOT reconstruction of every rejected single swap.
- The independent final-choice audit reconstructed all 7 joint candidates
  (168 accuracy measures) and all 96 reserved validation measures, verifying
  partitions, memberships, choices, gates and input/output hashes. Status pass.
- 54 focused tests passed, including every synthetic finite-swap likelihood,
  positive-control selection, membership constraints and validation-spike
  perturbations that cannot influence selection. Ruff passed.
- Code, protocol, compact results and hashes are preserved; no raw recordings
  are copied. No large scale-up or change to the original cohort was made.

Primary outputs: calibration/*_single_costs.csv,
calibration/frozen_assignments.json, calibration/selection_summary.csv,
calibration/gates.csv, audit/candidate_reconstruction.csv,
audit/risk_reconstruction.csv and audit/independent_audit.json.
