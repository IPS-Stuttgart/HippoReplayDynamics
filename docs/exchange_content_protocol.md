# Accuracy-guarded RUN-guided population exchange

Frozen2026-09-15 before selection. Development only; the active independent-data
remedy goal remains unchanged. Previous local screening failed held-out regional
accuracy, and reserve-cell acquisition improved accuracy without reducing the
original discrepancy. This intervention changes sampling at a FIXED cell budget.

## Fixed target

Four original matched PF pairs, three rats, all1,836 original candidate endpoints
and513 accepted-segment endpoints. No event selection, changed grid/maps/windows,
extra neurons, posterior averaging, temporal prior or likelihood correction.
Each original20-ms endpoint is decoded independently with flat-prior Poisson.
Early maps primary; full maps sensitivity. Home remains inferred, not truth.

Exchange cells between each original pair. Keep BOTH population sizes exactly
unchanged, their original UNION exactly unchanged and their original INTERSECTION
exactly unchanged. Shared cells can never be swapped. No reserve cell enters.
Thus agreement cannot be obtained by giving the populations more shared spikes.
High/low labels denote original ancestry after intervention, not current tuning.
This is a sampling-design remedy, not a decoder that can recover unrecorded cells.

## Selection, RUN only

Use each session's early maps and native Q3 first20-ms bin per250-ms parent, as
in the previous reserve experiment. Both true Home/non-Home classes must have
>=10 parents. Frozen full-RUN unit eligibility is inherited. These development
banks were already examined; even a pass requires a new external confirmation.

For each class k, compute D_k=mean(q_High(Home)-q_Low(Home)|true_class=k).
Objective J=(D_0^2+D_1^2)/2. This is a known-position response discrepancy,
not a replay outcome or a demand for constant replay prevalence.

Use at most10 accepted swaps. At each iteration compute the exact derivative
of J with respect to moving a cell's Poisson log-likelihood contribution from
low to high. For every current exclusive high cell i and low cell j, rank the
first-order swap change gradient_j-gradient_i (ascending, rounded10 decimals;
ties by integer i then j). Evaluate the first32 swaps exactly with the original
Poisson decoder. This is a bounded proposal search, NOT global optimization.

A swap is admissible only if J improves by>1e-10 and neither population's mean
physical error nor Home Brier rises above ITS ORIGINAL baseline in EITHER true
class (tolerance1e-10). These are eight separate local-accuracy constraints.
Among admissible swaps choose the smallest exact J; values within1e-12 tie by
integer i,j. If none is admissible, stop and retain the current populations.
Record every proposal, gradient, objective, classwise loss, and stopping reason.
Do not relax guards or enlarge the proposal pool after results.

Let k be each side's NET number of exchanged original cells in the final
assignment, which can be less than the accepted-step count. Freeze20 controls,
each exchanging k original exclusive cells per side chosen uniformly without
replacement. Seed:first8 SHA256 bytes little-endian of
`20260915|exchange|SESSION|DRAW`. Random and targeted final sizes/union/intersection
and net exchanged count must match. A zero-swap result is preserved, not replaced.
All assignments are frozen before Q4, simulated test banks or replay are opened.

## Evaluation

Evaluate unchanged native Q4 and five prior truth banks: Poisson, gain4,
conditional counts, map drift and shared assembly. Counts were generated ONCE
over the whole universe, so every allocation observes the same source spikes.
Average draws' metrics, not their posteriors; equal events within session,
sessions within rat, then equal rats. Report all20 controls.

Keep previous acquisition gates: exact source/event coverage; original candidate
Home gap>=20% reduction under early AND full maps with no rat worse; accepted
gap nonworsening; primary separation and regional TV>=10% reduction, improving
allthree rats; neither pooled entropy rises; allthree primary discrepancy
metrics beat mean equal-budget random exchange; neither population's balanced
error/Brier rises in any rat/source. Add the latest classwise safeguards:
physical error and Home Brier nonworsening for each true class, side, session
and truth source. No class/event can be discarded to pass. Baseline global RUN
quality matching is not asserted for newly exchanged populations; their actual
20-ms known-position performance is the relevant explicit safeguard here.

Independent audit reconstructs derivative ranking through a separate direct
cell-contribution calculation, every exact proposal and choice, random controls,
membership invariants, full likelihoods, unchanged event IDs, original baseline,
all summaries and gates. Numerical finite-difference tests check the derivative.
Hash protocol, code, sources and assignments. Detached server execution survives
SSH loss. Technical success or calibration improvement cannot complete the goal.

Any failed development safeguard blocks independent promotion. A stopped bounded
search is evidence about THIS rule, not a proof that no fixed-budget remedy exists.

Pre-run numerical clarification from synthetic zero-swap tests: beating the
random-control mean requires an excess>1e-10 in each metric. Identical assignments
cannot count as an improvement due to rounding when averaging20 identical draws.
This is an arithmetic tie tolerance, not a relaxed scientific threshold.
