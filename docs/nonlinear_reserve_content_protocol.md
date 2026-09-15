# Nonlinear reserve acquisition protocol

Frozen 2026-09-15 before results. This is a new development rule, not a rescue
or relabeling of the failed guarded-reserve rule and not independent validation.

## Unchanged scientific scope

Retain all four original PF matched pairs, every original neuron, and the frozen
1,836 candidate / 513 accepted endpoints. Use the same maps, spatial grid,
20-ms independent Poisson likelihood and flat prior. New cells may only come
from the reserve outside BOTH original populations. Added cells must be disjoint
and equal in number between the two sides; original overlap must be unchanged.
No event, class, session or rat may be dropped to produce a pass.

Reuse exactly the audited grouped train/validation partitions of native Q3,
cal_poisson_gain1 and cal_conditional. All training rows are used, with native
250-ms parents atomic. Prior development has inspected these banks; they are
not pristine external data. All four choices freeze before validation scoring.

## Why a different search

Summed original single-cell effects need not predict the nonlinear response to
joint additions. The prior rule filled the maximum quota. This rule instead
uses the actual nonlinear training losses while proposing allocations and
compares budgets {1,2,4,floor(reserve/2)}, deduplicated and feasible. A smaller
acquisition budget does NOT relax any final scientific improvement requirement.
It is selected from training only; no budget may be chosen using validation.

## Fractional optimization is only a proposal mechanism

Each reserve neuron has high and low weights in [0,1], with high+low <=1 and
each side's weight sum equal to the candidate budget. Original cells keep weight1.
Weights multiply the cell's log-Poisson contribution. These fractional likelihood
powers do not correspond to actual neuron subsets and must never be reported as
a validated decoder or scored on replay as the remedy.

Losses and analytic derivatives are recomputed at each proposed weight vector:
all24 source/side/true-class risks (physical posterior-mean error and Home Brier),
and native J=mean_k(mean(q_high-q_low | true_class=k)^2).
Test derivatives against numerical differences and binary endpoints against
independently reconstructed full Poisson probabilities.

For each budget:

1. Evaluate20 fixed random integer allocations on TRAIN data. The seed is the
   earlier reserve seed SHA256(20260915|reserve|SESSION|DRAW), first8bytes little
   endian. Each draw permutes the reserve; the first budget cells go high, the
   next budget cells go low. These also join the final training candidate pool.
2. Use two starts: uniform weights budget/reserve_count per side, and the random
   allocation minimizing maximum positive risk change divided by
   max(baseline risk,1e-6), then J, then name.
3. From each start, run SLSQP minimax feasibility: minimize t>=0 subject to
   risk <= baseline+1e-10+max(abs(baseline),1e-6)*t and the linear allocation
   constraints. Keep the linearly feasible evaluated point with lowest actual
   maximum scaled positive violation, then J. It need not reduce J yet.
4. Starting there, run SLSQP minimizing J/max(baseline J,1e-6), subject to all24
   actual risk guards and allocation constraints. Keep the feasible evaluated
   point with scaled violation <=1e-8 and smallest J; absent such a point, keep
   the point with smallest violation, then J. This soft tolerance only chooses
   a proposal. Actual integer eligibility still requires the original1e-10.
5. Each solve: at most60iterations and120 distinct-weight loss evaluations;
   ftol1e-9. Preserve all evaluation traces, statuses, selected weights and times.
   A numerical failure or budget limit is not proof of infeasibility or optimality.
6. Round each selected point by exact linear assignment to budget high slots,
   budget low slots, and unused slots. Maximize assigned weights. Generate9
   roundings: unperturbed, four with Gaussian score noise SD0.05, four SD0.2.
   Seed SHA256(20260915|nonlinear-reserve-round|SESSION|SEARCH_NAME|DRAW), first8
   bytes little endian. Keep every proposal and its exact integer membership.

At most16 solves and224 named integer candidates per real pair. Duplicate integer
memberships may share a score cache, but all named proposals remain visible.
Repeated random assignments do not constitute independent statistical replicates.

Recompute every integer candidate's original full likelihood using two numerical
implementations. An admissible candidate must improve J by >1e-10 and worsen
none of all24 risks. Choose minimum J; ties within1e-12 choose smaller budget,
then lexicographic high IDs, low IDs, and name. None admissible means unchanged
baseline and failure, not removal of that pair or a successful zero-cost remedy.

## Validation and progression

Freeze all four choices before scoring any validation counts. Require all96
validation risk keys and nonworsening, all four pairs augmented, and strict native
validation-J improvement in every rat (sessions equally weighted within rat).
No validation results from alternative candidates enter selection. If this fails,
stop before Q4, test banks, replay or independent recordings.

An internal pass permits a separately audited192-guard Q4/test-bank accuracy
preflight. Only after that passes, score all original fixed replay endpoints with
early/full maps and20 random allocations matched to each selected budget. Retain
the original20% Home-gap reduction,10% separation/TV reductions, per-rat,
accepted-segment, entropy, random-comparator and truth safeguards. Independent
recording validation of the frozen rule remains mandatory for the active goal.

Accuracy guards are point-estimate safeguards, not tests of biological harm.
Development success, fractional feasibility, optimizer success, and improved
agreement alone are not validated remedies.

## Independent serialized audit

Reconstruct every named integer candidate's24 training risks and objective,
every selected fractional proposal state using full weighted Poisson constants,
and all96 validation risks. Check exact seeds, budgets, disjoint memberships,
unchanged original overlap, train-only winner selection, frozen partition/hash
chain and all-cohort gates. Check allocation feasibility and the selection rule
over every recorded optimizer trace. Intermediate unselected fractional losses
are not independently numerically reconstructed; the auditor does not certify
optimizer optimality or existence/nonexistence of other possible solutions.
