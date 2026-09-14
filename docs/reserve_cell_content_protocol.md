# RUN-guided reserve-cell augmentation

Frozen before selection or replay outcomes, 2026-09-15. Development only;
independent-recording confirmation is still required for the active goal.

## Motivation and distinction

The within-bin predictive screen failed the original candidate-content and truth
gates. The source artifact and outcome are preserved. Earlier global spatial
balancing repartitioned the full population and produced small benefits. This
test instead retains EVERY cell of each original high/low Home-tuning population
and asks whether acquiring additional, RUN-selected cells can reduce the original
content discrepancy. It is a sampling intervention, not a posterior correction,
and is not claimed as a new principle of experimental design.

Use the four original PF matched pairs (three rats), all 1,836 fixed candidate
endpoints and 513 frozen accepted-segment endpoints. Baseline overlap, timestamp,
grid, Home definition, likelihood, 20-ms bin and event denominators do not change.
Cell eligibility remains conditional on the original full-RUN QC. Existing
baseline population matching used Q3 and Q4; this follow-up on those fixed pairs
is not pristine independent validation.

## Intervention frozen before outcomes

The reserve is all eligible cells in NEITHER original population. This has 34,
36, 30, and 21 cells for Rat1/Open1, Rat1/Open2, Rat2/Open1 and Rat4/Open2.
Each side receives floor(reserve size / 2) new cells: 17, 18, 15 and 10.
Augmentation sets are disjoint and outside both originals. Thus interpopulation
shared-cell count is exactly unchanged. The odd reserve cell remains unused.
No cell is removed and no replay observation enters assignment.

Primary assignment is greedy, using early-RUN rate maps and native Q3 20-ms
counts. For calibration, use the first 20-ms observation in each original 250-ms
parent block. Both Home/non-Home classes must have >=10 distinct parents. Give
the two classes equal total weight. At each step, evaluate addition of every
remaining reserve cell to each non-full side. Choose the cell/side giving the
largest reduction of that side's class-balanced Home Brier loss. Loss concerns
the unmodified flat-prior Poisson posterior. Ties use side order high/low then
global integer cell index. Continue until each quota is filled, even if the
best remaining gain is negative; report the full gain trace. This does not fit
the A/B disagreement, compare replay results or force Home probabilities equal.

Comparator: 20 frozen random allocations of the same reserve, equal quotas and
unchanged overlap. Seed is SHA256('20260915|reserve|session|draw'), first eight
bytes little-endian. Their mean is the random-budget comparator; random draws
are not independent animals. Baseline has no added cells. A proposed advantage
must exceed ordinary addition of the same number of cells.

Freeze all assignments to files BEFORE evaluating any Q4, real endpoint or
known-truth test outcome. The selector's API accepts only encodings and Q3.
Do not use target or test arrays in the selector. Save all source/code hashes.

## Evaluation

Reuse the previously native-audited regional-prevalence cache without new spike
generation: native Q4; matched Poisson; gain4; endpoint-total conditional counts;
late-map drift; 80% shared assembly. Simulations were generated once for the
whole eligible cell universe. All added-cell observations must therefore be
the SAME native/simulated counts under targeted and random assignment.
Use early-RUN maps primary, full-RUN maps for real-data sensitivity only.

Report the original absolute per-session mean Home-posterior gap, nine-tile TV,
mean-position separation, entropy for each side, and each side's class-balanced
true-position error and Home Brier loss. For known truth, equalize true Home and
non-Home classes; no selective event omission. Average events within sessions,
sessions within rat and rats equally. Average random-allocation metrics within
session first; do not average their posteriors before measuring disagreement.
Also report Q3 selected gains and Q4 changes: Q3 gains alone are not validation.

## Gates for independent confirmation

- All original pairs, candidates and accepted endpoints preserved; every original
  baseline readout reconstructed; equal added-cell budgets and unchanged overlap.
- Targeted candidate Home gap decreases >=20% versus original, on early and full
  maps, with no rat worsened. Accepted-segment Home gap does not worsen.
- Primary candidate separation and regional TV decrease >=10% versus original,
  and both improve in all three rats. Each side's mean entropy does not increase.
- Targeted candidate Home gap, separation and TV are lower than the mean of the
  20 budget-matched random allocations. Report all random draws, not just mean.
- Each side's class-balanced Q4 physical error and Brier loss do not worsen in
  any rat versus original. The same safeguards apply to every frozen synthetic
  condition. A consistent false assembly is an explicit robustness challenge.
- All selection, counts, readouts, aggregation and gates independently audited.

Any failure prevents promotion of this remedy and does not justify retuning its
allocation rule or replacing candidate endpoints by favorable accepted segments.
Even passing does not complete the goal. Before calling the method validated,
freeze it unchanged for independent recordings, preserving their entire source
cohort and applying the same random-budget/truth safeguards. No conclusion about
the animal's true replay destination follows from A/B agreement alone.

If new-cell acquisition succeeds where posterior corrections failed, the scope
is improving sampling with available reserve neurons, NOT salvaging a recording
in which those neurons were never recorded. Record that practical limitation.
