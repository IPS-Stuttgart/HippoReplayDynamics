# Finite whole-cell exchange calibration

Frozen 2026-09-15 before evaluating the rule. This development experiment follows
the failed robust-gradient rule without altering its choices or calling its
validation independent. The same development banks have already been inspected.
Independent recording confirmation is still required by the active objective.

## Unchanged scope

Keep all four original PF pairs, cell counts, union/intersection, encoding maps,
1,836 fixed candidate endpoints and 513 fixed accepted-segment endpoints. Use
the exact hash-based train/validation partitions from robust-exchange-20260915.
Only training rows of RUN-Q3, cal_poisson_gain1 and cal_conditional may influence
selection. All 24 class/population/source physical-error and Home-Brier guards
remain nonworsening at tolerance 1e-10. Retain every pair, including failures.

## Finite costs, not derivatives

Enumerate the complete original single-cell-swap neighborhood: every
high-exclusive cell exchanged for every low-exclusive cell. Measure actual
full Poisson posterior statistics on all three training banks for every swap.
An incremental likelihood update is algebraically exact, not linearized.
The fixed source neighborhoods contain 3,025, 5,041, 2,025 and 2,209 swaps.

Before proposing any joint swaps, check the finite-cost tables against the
separate full-likelihood implementation for 32 deterministic SHA256-ranked
swaps per session, the minimum-J swap, and every exactly admissible single
swap. Hash ranking uses `20260915|finite-exchange|session|high|low`. Any numerical
disagreement stops the run. The table covers all single swaps but this numerical
audit is sampled, not an independent reconstruction of every rejected entry.

Use the measured single-swap changes as approximate coefficients for joint
swaps of size 2, 4 and 8, at most four proposals per size. Integer matching
constraints prevent using any cell twice. Require all 24 summed risk changes
nonpositive and summed J change in [-J, -1e-10], normalized as in the preceding
rule. HiGHS limits remain 20 seconds, 64 nodes, relative gap .001. Validate any
limited incumbent at tolerance 1e-7. Exclude repeated complete memberships,
not merely repeated pairings, after each feasible proposal.

These summed finite costs are still approximations for joint moves. Recompute
every joint move with actual likelihoods on all three training banks and
independently reconstruct its risks. Single and joint candidates compete only
if native J improves by >1e-10 and every actual risk is nonworsening. Choose
minimum actual J, ties within 1e-12 resolved by fewer swaps and lexicographic
membership. With no admissible candidate, keep the original population.

## Validation and progression

Freeze all four final choices before decoding any reserved observations with
them. Require all 96 validation risks nonworsening, changed populations in every
original rat and lower native validation J in each rat (equal session weights).
No alternate choice is allowed after validation. Failure stops before Q4,
test-bank or replay scoring. Success only permits the complete previously
frozen truth, replay discrepancy, entropy, random-control and independent-data
gates; it never establishes a remedy by itself.

Separate worker processes may evaluate different sessions on gpuserver6000,
with one BLAS thread each. The run must survive SSH loss and preserve complete
tables, hashes, worker outcomes and solver status. A numerical/worker failure
is a technical failure, not absence of a feasible remedy. No truncated search
may be described as exhaustive.
