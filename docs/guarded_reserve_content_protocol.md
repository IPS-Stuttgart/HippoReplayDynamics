# Guarded disjoint reserve-cell acquisition

Frozen 2026-09-15 before this rule's outcomes. Development follow-up, not
independent validation and not a relaxation of the active goal's accuracy gates.

## Distinction from earlier attempts

Posterior-only corrections have a proper-score cost when the original decoder
is model-correct. This experiment instead adds genuine recorded observations.
It does not give both populations the same new cells to manufacture agreement.
Unlike earlier reserve allocation, it uses all training rows and separate
accuracy guards, not first-row-per-parent Brier with forced greedy additions.
The previous failed rule remains failed; this is a separately frozen method.

Retain the four original PF matched pairs and all original neurons, timestamps,
grids, likelihoods and fixed 1,836/513 candidate/accepted endpoints. Reserve
neurons occur in NEITHER original population. Each side's quota is half the
reserve rounded down: 17,18,15,10 cells in Rat1/Open1, Rat1/Open2, Rat2/Open1,
Rat4/Open2. Additions are disjoint, keeping the original intersection unchanged.
One unused neuron remains where the reserve is odd. No original cell is dropped.

## Training and internal validation

Reuse EXACTLY the audited robust-exchange grouped partitions of native Q3,
cal_poisson_gain1 and cal_conditional. Each native 250-ms parent remains atomic
across its twelve 20-ms observations. Simulation rows are atomic. Include all
training rows, not the first row only. Require >=10 training and >=3 validation
parent groups containing each truth class. No validation counts enter selection.
These banks were inspected in prior development; they are not pristine external
test data. Existing full-RUN unit eligibility and original matched-pair selection
also remain limitations. All assignments freeze before validation is decoded.

Native objective J is the mean of the squared class-conditional mean
high-minus-low Home-probability differences: mean_k(mean(delta | class=k)^2),
not mean(delta^2). Training accuracy consists of 24 risks:
3 sources x 2 populations x 2 true classes x (physical error, Home Brier).
The forward likelihood remains flat-prior 20-ms independent Poisson decoding.

## Proposal search, frozen before outcomes

1. Measure every original single reserve addition to each side (242 total).
   Reconstruct each full likelihood and all risks independently. Unlike a
   derivative, this measures an actual added neuron's effect.
2. Sum measured single-addition changes only as an APPROXIMATION for proposing
   whole quota-matched allocations. Binary variables indicate cell/side use.
   Enforce exact per-side quotas, no repeated cell and disjoint additions.
3. Two finite-cost proposal families, at most 12 distinct allocations each:
   (a) predicted nonworsening of all 24 risks; (b) objective-only search. Both
   minimize summed native J change. The second avoids excluding a genuinely
   safe joint allocation solely because summed risk changes were inaccurate.
   NO approximate score certifies actual safety. Predicted negative J/risk
   values are permitted because the additive approximation is not a true risk.
4. HiGHS: 20 s,64 nodes,relative gap0.001 per solve. Accept a limited incumbent
   only if binary and constraint-feasible within1e-7. Exclude each complete
   previous allocation, not its score. Stop a family on absent incumbent.
5. Add 20 fixed random quota-matched disjoint allocations, seeded exactly as
   the previous reserve comparator: 20260915|reserve|SESSION|DRAW. These are
   also training-search candidates. Record every random assignment; never
   select using their validation/replay results.
6. Reconstruct every feasible candidate exactly on all three training banks.
   It is admissible only if J improves by >1e-10 AND all 24 risks are <= their
   original baseline+1e-10. Choose the minimum actual J; ties within1e-12 use
   lexicographic high/low added-cell IDs then proposal name. At most44
   candidates per session. This remains bounded search, not exhaustive joint
   optimization. If none passes, keep the original and record failure.

The proof implementation independently reconstructs every evaluated baseline,
single addition and joint candidate. Preserve the finite cost tables, solver
status/limits, exact candidate risks and choices. Run on gpuserver6000 with four
CPU workers and one BLAS thread each, detached from the SSH connection.
The separate audit additionally reconstructs serialized single-addition costs,
all candidate risks, the training winner, random memberships, all96 validation
risks and all80 random native validation objectives. It checks the source chain,
grouped partitions, unchanged overlap, temporal freeze and non-vacuous gates.
Accuracy guards are conservative point-estimate constraints, not independent
significance tests; failing them does not prove population-level harm.

## Validation and progression

Freeze all four choices before scoring any reserved validation rows. Require:
all four original pairs present, quota augmentation in ALL four pairs, all96
validation risks present/nonworse, and native validation J improvement in each
rat (sessions equally weighted within rat). Record random native validation J
for all20 allocations; this is descriptive development context, not a substitute
for the final real-data random-budget comparison.

Any failure blocks Q4/test/replay/independent-recording promotion. No partially
passing cohort may replace the original. An unchanged baseline is not success.
If this internal screen passes, a separate frozen-assignment preflight must check
all192 original side/session/class/metric accuracy comparisons on nativeQ4 and
the five frozen truth banks. After that, and only then, score all original
1,836/513 real endpoints with early/full maps and all20 random allocations.
The original20% Home-gap,10% separation/TV,rat-uniform,entropy,random-budget and
known-truth safeguards remain unchanged. Independent-recording confirmation
with this frozen rule is still mandatory before claiming a validated remedy.

This tests whether existing reserve neurons can improve sampling. It cannot
repair recordings in which these additional neurons were never recorded.
