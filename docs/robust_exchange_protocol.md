# Robust calibration-gradient population exchange

Frozen 2026-09-15 before evaluating this rule. This is a new development
experiment, not a retuning or replacement of the failed joint-exchange choices.
Previous evaluation banks have been inspected in development. Independent
recording confirmation remains mandatory for the active goal.

## Scope and rationale

Retain the original four PF population pairs, all cells per pair, each pair's
union and intersection, early/full maps, and all 1,836/513 fixed event endpoints.
Do not change the decoder, add shared observations, discard difficult pairs,
retime endpoints, or relax any final accuracy/discrepancy safeguards.

The previous rule protected eight empirical risks on only one 20-ms sample per
250-ms RUN parent. Its agreement gains transferred, but its safety did not.
This new rule protects native and simulated calibration risks, using all 20-ms
samples inside training parents, and separately checks unselected parent groups.

## Fit/validation split

Use native RUN-Q3, cal_poisson_gain1, and cal_conditional only. For native data,
keep every 250-ms parent intact; for simulations, a generated observation is a
group. Stratify groups by their first observation's true Home label. Order each
stratum by SHA256(`20260915|robust-exchange|session|source|group`). Reserve
max(3, round(0.25 * group_count)) groups per stratum for internal validation.
Require at least 10 training and 3 validation groups containing each true class.
If unavailable, retain the original pair with an explicit failure; never replace
the session. The split is label-stratified, not based on candidate outcomes.

All fitting uses the training rows only. Freeze choices for all four pairs
before scoring internal validation rows. These grouped rows remain within the
same sessions; they are not independent recording validation.

## Proposals and exact checks

Use analytical derivatives of the full Poisson posterior with respect to
continuous cell weights, evaluated at each original binary population. Native
training J is the mean squared high-minus-low Home-mass difference over the
two true classes. Separately differentiate physical posterior-mean error and
Home Brier for each class, population and training source: 24 risk constraints.
No held-out counts influence these derivatives or the proposal ranking.

Binary variables remove high-exclusive cells and exchange them for low-exclusive
cells. Use budgets 1, 2, 4 and 8 per side, up to four distinct memberships per
budget. Minimize linearized native J, subject to every linearized risk being
nonworsening and predicted J change in [-J, -1e-10]. Scale risk constraints by
max(abs(baseline risk),1e-6), objective by max(J,1e-6). HiGHS limits: 20 seconds,
64 nodes, relative gap 0.001. Accept a limited-run incumbent only if binary
membership and all scaled constraints independently pass at tolerance 1e-7.
No-good constraints exclude the complete membership, not an arbitrary pairing.
Missing incumbents stop that budget, not other budgets.

Derivatives are proposal approximations only. Evaluate each proposed membership
with actual Poisson likelihoods on all three training sources, independently
reconstructing risks with the existing full-likelihood auditor. A candidate is
admissible only when actual native J improves by >1e-10 and all 24 actual risks
are <= their original values +1e-10. Choose minimum actual J; ties within 1e-12
favor fewer exchanged cells, then lexicographic high/low IDs. If none qualifies,
retain the original pair, not a weaker candidate.

## Internal validation and progression

Apply the frozen assignments to all reserved native/simulated groups. Require
each side's physical error and Home Brier nonworsening within each true class
and source, in every session. Require a changed population in every original
rat and improved native validation J in each rat (equal session weights).
Unchanged pairs remain in all tables; zero changes cannot pass vacuously.
Do not choose another candidate after looking at validation results.

A failure stops this rule before Q4, test banks or replay scoring. A pass still
requires the previously frozen Q4/five-test-bank classwise safeguards, original
fixed-endpoint replay discrepancy reductions, entropy safeguards and equal-budget
random controls, then the unchanged method on independent recordings. Calibration
or numerical correctness alone never completes the active goal.

Report full input/code hashes, frozen group/choice IDs, proposal feasibility,
exact training and validation risks, failures, retained denominators and the
absence of external validation. Never interpret point-estimate nonworsening
checks as independent significance tests or biological ground truth.
