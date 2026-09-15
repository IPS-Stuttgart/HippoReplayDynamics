# Joint population exchange: calibration gains, failed accuracy transfer

## Decision

Stop this frozen rule before replay rescoring or independent-recording promotion.
It does not demonstrate a validated remedy. Agreement optimization generalized
to unused native samples, but its accuracy safeguards did not. This is not
evidence that no remedy can exist or that a particular replay estimate is true.

## Frozen Selection

The joint search retained all four original PF pairs, their cell counts, union
and intersection. It proposed exchanges of 2, 4 or 8 exclusive cells per side using
a time/node-capped mixed-integer surrogate, then checked actual full-Poisson
outcomes and all 8 classwise risks. Selection used only the first 20 ms per 250 ms
native Q3 parent with early RUN maps. Choices preceded held-out evaluation.

| Session | Exchanged per side | Q3-first objective before | After |
|---|---:|---:|---:|
| Rat1/Open1 |4|0.0436521|0.0089796|
| Rat1/Open2 |4|0.0219666|0.0131373|
| Rat2/Open1 |2|0.0071350|0.0068254|
| Rat4/Open2 |2|0.0674499|0.0156859|

The objective is the mean squared high-minus-low Home-mass discrepancy across
the two true position classes. It is not replay evidence, localization error,
or a measure of biological replay. All four choices passed all 8 calibration
risks. An independent implementation reconstructed 55 candidate outcomes (48 joint
proposals plus 7 previously admissible singles). This does not prove optimality or
exhaustive joint-search coverage.

## Held-Out Truth Preflight

Use unchanged native Q4 and five frozen simulated test banks, with two full
likelihood implementations. All four pairs and both true classes were retained.
Of 192 region/side/source/metric comparisons, 99 were nonworsening and 93 worsened.
These are point-estimate safeguards, not 192 independent significance tests.

| Source | Physical-error guards passed / 16 | Home-Brier guards passed / 16 |
|---|---:|---:|
| Native later RUN (Q4) |8|7|
| Poisson gain 1 |9|8|
| Poisson gain 4 |10|7|
| Conditional counts |9|8|
| Conditional with map drift |9|8|
| Conditional with shared assembly |8|8|

For true-Home native Q4 observations, high-population physical error changed:
Rat1/Open1 20.97 to 21.30 cm; Rat1/Open2 27.76 to 29.58 cm;
Rat2/Open1 34.05 to 34.96 cm; Rat4/Open2 38.31 to 44.14 cm.
For Rat4/Open2, Home Brier changed 0.64995 to 0.69673. This is not solely a tiny
roundoff failure. The 20-ms RUN errors are not interchangeable with the earlier
global 250-ms RUN-QC numbers.

The audit reconstructed 418,304 population posteriors independently across
104,576 paired truth observations and verified original cached baseline values.
No alternative choice was selected after inspecting these outcomes.

## Why the Calibration Result Was Misleading

A post-hoc fixed-choice transfer audit compared selected Q3-first observations,
unused nearby 20-ms samples from the same parents, and later Q4 observations.

| Native sample set | Pairs with better agreement objective | Classwise risk guards passed / 32 | Pairs passing all 8 risks |
|---|---:|---:|---:|
| Q3 first 20 ms, used for selection |4/4|32|4/4|
| Q3 remaining 20-ms samples |4/4|18|0/4|
| Q4 all 20-ms samples |4/4|15|0/4|

The agreement objective improves in every reported phase, including both
separate Q4 phases. Therefore the failure is not simply nontransfer of agreement
gains. It is nontransfer of the apparent safety of those gains. This is already
visible in nearby unused Q3 data, not just in synthetic model mismatch or later
recording drift. Q3-first included only 14 true-Home samples for Rat4/Open2, versus
150 in its unused adjacent samples. Selection against a small regional sample is
a plausible source of optimistic safeguards, not a proven unique mechanism.

Nearby samples share parents and are temporally correlated. All-bin summaries
overlap first/remaining summaries; they are not independent replications. Q4 is
a later RUN block, not an independent animal or dataset. This diagnostic was
specified after the truth failure and must remain labeled post-hoc.

## Provenance and Validation

All computation ran on gpuserver6000 in detached user services, now terminal.

- Selection: `/mnt/seagate10tb/florianpfaff/joint-exchange-20260915/selection`,
  producer `bf0fabaa`, 463.47 s. Manifest SHA256:
  `8f90b677c57db2d0586cf7836634e03b7f36e3efa39219d5fdd9b61ae6161c47`.
- Truth: `/mnt/seagate10tb/florianpfaff/joint-exchange-truth-20260915/measurement`,
  producer `678b229a`, 9.80 s. Manifest SHA256:
  `424719ee291ec354bd947ee88c6178ee75108aaf7fda13bf7f8efd08cfa051ce`.
- Transfer: `/mnt/seagate10tb/florianpfaff/joint-exchange-transfer-v2-20260915/measurement`,
  producer `55d482b6`, 6.65 s. Manifest SHA256:
  `dd03930f1006d01ad2ea31ea6487d9aff043340ee40d23171ea9c20b334d3b90`.
- The first transfer run exited on a provenance-schema check: Q4 is deliberately
  absent from RUN-only selection inputs. The corrected audit verifies Q4 through
  the linked completed truth audit. No data hashes or scientific gates changed;
  the first failed run is preserved at `joint-exchange-transfer-20260915`.
- 41 related tests pass; Ruff passes. Tests cover true joint-only synthetic
  improvements, exact constraints, leakage-resistant manifest handoff,
  classwise failure despite pooled improvement, nonvacuous coverage and source
  tampering. Separate-run Q4 classwise tables agree numerically; nullable CSV parent
  counts differ only in storage dtype.

Compact local exports, without raw recordings:
`/mnt/c/Users/emper/Documents/codex/2026-09-15/joint-exchange/`.

## Consequence for the Active Goal

An independently validated remedy/diagnostic remains unachieved. No replay
instability-reduction result is claimed for this rule, no external recording is
promoted, and no accuracy guard is relaxed. A future selection rule must test
accuracy transfer on group-held-out calibration observations before trusting an
agreement optimum, followed by the original fixed-event and independent-data
gates. This is a next research requirement, not a demonstrated solution.
