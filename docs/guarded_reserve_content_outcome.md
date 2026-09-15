# Guarded reserve acquisition: no admissible allocation in the frozen search

## Decision

The new disjoint reserve-cell rule failed its development screen. Do not promote
it to Q4, frozen truth banks, replay, or independent recordings. No validated
remedy has been established, and the full goal remains open.

This is a bounded search result, not an impossibility theorem about collecting
more neurons. A training loss decrease is not a verified replay improvement.

## Frozen experiment and provenance

- Server: gpuserver6000.
- Code and protocol commit: c246f9a6b63d36e361ec20b9f34673ba64b52ac4.
- Root: /mnt/seagate10tb/florianpfaff/guarded-reserve-20260915.
- Producer manifest SHA256:
  be064246ed8c1d48f29fa418cc2eadb5d861c7b89f9d8b8d37be30e6f5d3bc5b.
- Producer service: guarded-reserve-20260915.service, exit status 0;
  2026-09-15 04:50:26--04:51:16 UTC.
- Independent audit: guarded-reserve-audit-20260915.service, exit status 0;
  2026-09-15 04:52:02--04:53:22 UTC.
- Both used systemd user services with RemainAfterExit, independent of SSH.
- Relevant test subset: 78 passed. Ruff and staged whitespace checks passed.

All four original PF pairs were retained. Original neurons were never removed,
and new reserve neurons could be assigned to at most one side. Original
interpopulation overlap was unchanged. Native grouped Q3 and the two calibration
banks used the exact previously audited train/validation partitions. These are
development banks, not independent recordings or untouched external data.

Each candidate had to reduce native training
J = mean_k(mean(q_high(Home)-q_low(Home) | true_class=k)^2)
while preserving all 24 source/population/truth-class accuracy risks. These
include physical posterior-mean error and Home-probability Brier score. The
point-estimate tolerances remained unchanged at 1e-10.

## Training search

| Pair | Added-cell quota per side | Single additions measured | Joint candidates | J improved | Minimum failed risks | Admissible |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Rat1/Open1 |17|68|32|21|6|0|
| Rat1/Open2 |18|72|32|30|5|0|
| Rat2/Open1 |15|60|32|24|5|0|
| Rat4/Open2 |10|42|32|16|3|0|
| Total | |242|128|91| |0|

For every pair, the additive predicted-risk MILP was infeasible. That result
concerns the APPROXIMATE additive constraints, not the true nonlinear risks.
The second, objective-only family produced 12 quota-feasible proposals per pair;
all 48 improved actual J but failed accuracy. All 20 frozen random allocations
per pair were also evaluated, giving 80 random candidates. Of those, 43 improved
J; none passed all accuracy guards. The search is not exhaustive over allocations.

Every proposed joint allocation was rescored with the full nonlinear likelihood.
Across 128 candidates x 24 risks, 807/3072 point-estimate guards failed. Even the
least-violating candidates failed 3--6 risks, so no final augmentation was frozen.

## What the failures mean

There are genuine improvements and trade-offs, not universally worse decoding:

- All 128 allocations improved non-Home physical error for both populations in
  each of the three training sources: 768/768 such comparisons improved.
- All 128 allocations improved the Home-poor population's true-Home Brier score
  in all three sources: 384/384 such comparisons improved.
- Conversely, all 128 allocations worsened that population's NON-Home Home-Brier
  score on both native Q3 and matched-Poisson calibration data: 256/256 failures.
  Median increases across the tested allocations were 0.000257 and 0.000217,
  respectively. For true non-Home observations this score is mean(q(Home)^2),
  a penalty on false Home probability, not a binary false-positive rate.
- Some failures also involved physical error at true Home. For example, the
  least-violating Rat1/Open1 allocation increased Home-poor Home-position error
  by 3.12 cm on native training rows and 3.63 cm on matched-Poisson calibration.

These are descriptive, selected-development comparisons, not independent
significance tests. They neither establish population-level harm nor imply that
the decoder is incorrect. Additional information can improve overall prediction
while changing conditional class risks. This experiment required all separate
guards to be nonworse and did not meet that requirement.

## Validation and audit

All four populations fell back to their originals. Accordingly all 96 validation
risks were unchanged, while all-pair augmentation and every-rat native-J
improvement failed. The readiness gate correctly remained false; equality with
an unchanged baseline cannot count as a remedy. The 80 native validation
objectives for random allocations are descriptive only and were not used for
selection. No fallback candidate was chosen after viewing validation results.

The independent auditor verified source/output hashes, exact grouped partitions,
all assignments and quotas, unchanged original overlap, temporal freezing before
validation, solver proposal coverage, all random memberships and training-only
winner selection. It reconstructed:

- 242 single-addition table rows, all 24 risks per row;
- 128 joint candidates and 3,072 joint risks;
- 96 validation risks and 80 random validation objectives;
- All non-vacuous gates.

Maximum single-addition risk reconstruction error was 1.49e-13. The audit passes
the numerical/accounting checks while explicitly preserving the scientific
screen failure. No Q4/test-bank/replay/external decoding occurred in this run.
The original 1,836 candidate and 513 accepted endpoints were not replaced or reduced.

## Solver-options correction

Before the real run, a test exposed that installed SciPy consumes `node_limit`
from the caller's options dictionary. The new implementation passes a fresh copy
to every solve; a regression test and the manifest verify the frozen 64-node,
20-second limits. This fix was included before commit and evaluation.

Older joint/robust/finite-exchange scripts pass their shared dictionary directly.
Their stated 64-node cap therefore cannot be assumed to have applied to every
successive solve in a process. The 20-second cap is not consumed. This corrects
the historical search-budget description, not their independently reconstructed
candidate losses or eligibility results. Those immutable scripts and artifacts
were not rewritten here; no previous failed method is promoted by this finding.

## Claim boundary and next requirement

This rules out THIS frozen acquisition rule as a validated remedy. It does not
rule out alternative acquisition designs, other observation models, or a useful
diagnostic. Any next approach must still preserve the original cohort and all
accuracy/entropy/random-budget safeguards, then demonstrate prediction and
reduction of instability on independent recordings. No such result exists yet.
