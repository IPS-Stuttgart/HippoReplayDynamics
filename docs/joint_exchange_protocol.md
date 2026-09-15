# Joint RUN-only exchange proposal experiment

Frozen2026-09-15 after the independent12,300-single-swap audit. This new
development experiment does not alter the original failed bounded exchange run.
Question: can simultaneous cell exchanges satisfy the unchanged regional
accuracy constraints where sequential single-cell moves get stuck?

Keep all four original PF populations, exact size, original union/intersection,
early RUN maps and first20-ms Q3 observation per250-ms parent. No replay, Q4,
synthetic test bank, extra cell, posterior average or changed window enters
selection. All earlier observations are development, not blinded validation.

Use all independently audited single-swap rows, including objective-worsening
ones, as APPROXIMATE joint-proposal coefficients. For each row e let dJ_e be
its exact single-swap J minus baseline J, and dR_e the eight separate physical
error/Home-Brier changes (both populations x both true classes).

For exact joint budgets2,4,8, generate up to four distinct proposals using
scipy.optimize.milp (HiGHS): binary variable per possible swap; sum variables
equals budget; each exclusive high/low cell participates at most once;
sum dR<=0 for each risk; -baselineJ<=sum dJ<=-1e-10. Minimize sum dJ.
Normalize each risk row by max(abs(baseline risk),1e-6), objective by
max(baselineJ,1e-6). Use all edges in high-ID/low-ID order. Node limit64,
time limit20seconds, relative MIP gap0.001. Record termination, gap, bound,
runtime and exact chosen edges. A feasible incumbent after a limit is a proposal,
not proof of optimality. Missing/invalid incumbents stop that budget, not the
whole experiment. Feasibility tolerance1e-7 is for the scaled proposal problem
only, NOT for the actual accuracy constraints.

Exclude each previously proposed MEMBERSHIP by a no-good inequality: for a
budget-k prior proposal with exchanged sets H,L, sum over chosen edges of
[high endpoint in H]+[low endpoint in L] <=2k-1. This excludes alternative
pairings with the same final populations. Budgets larger than either exclusive
set are explicitly skipped, not silently reduced.

Evaluate every proposed group with the exact full joint Poisson likelihood.
The sum of single-swap changes is not an exact decoder or a safety certificate.
Keep all seven individually admissible swaps from the complete prior audit as
additional candidates. A candidate passes only if exact J decreases>1e-10 and
all eight exact risks are <=their ORIGINAL baseline+1e-10. Pick minimum exact J
among admissible candidates; values within1e-12 tie by fewer exchanged cells,
then sorted exchanged high IDs and low IDs. Otherwise keep the original pair.

Independently rebuild each combined population and full Poisson posterior;
verify original membership, all proposal constraints, no-good exclusions,
singleton coverage, exact classwise risks/J, admissibility, chosen assignment
and source hashes. This verifies feasible proposals/outcomes, not global MILP
optimality or search completeness. Synthetic tests must include a genuine joint
move that passes exact risks even though every single swap fails if one can be
constructed; do not present mocked risks as scientific positive recovery.

Write selection.json, proposals.csv, summary.csv, report.md, manifest.json and
independent_audit.json. Freeze all assignments before any further evaluation.
This phase alone cannot satisfy the independent-data remedy goal. Any proposed
remedy must next retain1,836 original candidates and513 accepted endpoints,
pass all existing held-out RUN/simulation/discrepancy guards on the full cohort,
and pass a separately frozen independent-recording confirmation. No favorable
session restriction or relaxed guard is allowed to count as success.
