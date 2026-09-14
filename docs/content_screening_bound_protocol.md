# Fixed-window screening attainability bound

Frozen 2026-09-15 before solving. This follows the FAILED reserve-cell remedy.
It is an optimistic feasibility analysis to direct the active research goal,
NOT a validated predictor, new decoder, external confirmation or biological test.

## Question

Can selection alone simultaneously reduce regional-content disagreement and
position disagreement without increasing posterior entropy or known-truth loss?
The acquisition experiment improved known-position error while increasing
between-population separation, so these requirements cannot be conflated.

Use only ORIGINAL baseline posteriors from the independently audited
reserve-cell-content-20260915 measurement: four PF pairs, three rats, 1,836 fixed
candidates and 513 accepted-segment endpoints. No added-cell results enter.
Original counts, maps, endpoints, Home region and likelihood are unchanged.
Early maps primary, full maps real-only sensitivity. This script does not
decode again. The audit verifies source hashes against the full prior audit.

## Two oracle classes

- free_event: each observation has its own retention probability, even when
  observed counts are identical. This can exploit known truth and is a leaky
  optimistic ceiling; never call it a deployable selector.
- count_pattern: observations with identical count vectors over the UNION of
  the original high/low cells must receive the same retention probability.
  This imposes observation indistinguishability within a session/source/map.
  It still uses outcomes to optimize and can memorize unique count patterns.
  Thus feasibility is NOT evidence of out-of-sample predictability. Rules are
  allowed to differ across sources; this is NOT a single transferable rule.

The variables are fractional probabilities in [0,1], not a realized integer
event subset. Primary retained fraction q=0.5; q=0.25 and 0.75 are frozen
sensitivities, not alternatives that can rescue the original half-cohort rule.

Solve separately per session/source/map. Real-data retention is exactly q.
For known-truth banks, retain exactly q within EACH true Home/non-Home class.
This avoids a gain obtained by removing one spatial region, but is an extra
constraint compared with the earlier empirical fixed-half screen. Likewise,
these bounds are per session, stricter than earlier per-rat mean constraints.
Consequently failure here does not prove every possible goal-level rule
impossible. Success only demonstrates compatibility under these constraints.

## Objective and safeguards

Maximize progress t in [0,5] subject to weighted retained metrics satisfying:

- absolute difference of mean Home posterior mass <= (1 - 0.2 t) times baseline;
- mean between-population position separation <= (1 - 0.1 t) times baseline;
- mean nine-tile posterior TV <= (1 - 0.1 t) times baseline;
- mean entropy for each population does not increase.

t=1 meets the 20% Home-gap and 10% separation/TV numerical reduction targets.
t=0 is always feasible by giving every observation probability q. Report that
zero-improvement solution correctly, never as a successful screen. Home gap is
the absolute signed MEAN difference, not mean absolute eventwise difference.
Constraints refer to the weighted population, not the expected absolute gap of
a finite random draw. Fractional solutions may require randomization in practice.

For all six known-truth sources (native Q4, matched Poisson, gain4, conditional
endpoint-total counts, map drift, shared assembly), solve both:

- agreement_only: no known-truth loss constraint;
- truth_guarded: neither population's class-balanced position error nor Home
  Brier loss may increase. The equal class-retention constraint makes these
  mean-loss inequalities linear.

The true labels are constraints in the oracle only, not permissible real replay
features. Truth-source agreement improvement is an additional diagnostic, not a
replacement for the original real-data content criterion. We cannot enforce
unknown biological accuracy on real replay. No oracle result establishes it.

## Verification and interpretation

Use scipy.optimize.linprog with HiGHS. Save every probability vector and primal/
dual certificate, solve status, counts of observations/unique patterns, full
source identities and all metrics. Independently rebuild grouping and linear
constraints, verify primal feasibility, dual signs, stationarity, objective gap,
and summaries from the immutable baseline rows. Solver success alone is not
an optimality audit. Preserve any timeout/failure; do not count it as a bound.

If even the free oracle cannot reach t=1, this particular selection problem is
too constrained; it is not proof against other data, windows or estimators.
If free passes but count-pattern fails, distinguishability of observations is
a demonstrable limitation under the stated constraints. If both pass, the
remaining task is to learn an OBSERVABLE rule and verify it on independent data.
None of these cases completes the user's validated-remedy/diagnostic objective.
No thresholds, oracle choices or sources are selected after inspecting outcomes.
