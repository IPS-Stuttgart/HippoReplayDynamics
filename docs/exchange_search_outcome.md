# Exhaustive single-swap calibration audit outcome

Completed on gpuserver6000 in detached service exchange-search-20260915, exit0,
2026-09-15T01:37:47.284394UTC,137.042seconds. Producer commit
b38fede14df3b9e36bec2b8f88bc7d082df7c393.

## Result

The original32-proposal exchange search missed admissible swaps. However, merely
expanding its single-swap search cannot supply a remedy across all four pairs.

| Pair | All legal swaps | Improve calibration objective | Pass all eight accuracy guards | First passing derivative rank |
| --- | ---: | ---: | ---: | ---: |
| Rat1/Open1 | 3025 | 1365 | 0 | none |
| Rat1/Open2 | 5041 | 2265 | 2 | 1278 |
| Rat2/Open1 | 2025 | 1224 | 0 | none |
| Rat4/Open2 | 2209 | 1201 | 5 | 116 |

All12,300 proposals were reconstructed independently from full population
Poisson likelihoods. Largest numerical discrepancy across session audits was
2.99e-13. The26 focused tests pass, including missing/tampered proposal detection,
all-pair enumeration, empty neighborhoods, original32-proposal correspondence,
classwise safeguards and an end-to-end synthetic provenance test. Original
bounded artifacts remain unchanged and FAILED.

Of6055 objective-improving swaps, only7 satisfy every separate physical-error and
Home-Brier nonworsening constraint for both populations and both true spatial
classes. Guard rejection counts overlap; proposal counts are NOT numbers of
independent replay observations. In particular, improving agreement alone is
easy and does not certify improved decoding.

The best admissible Rat4/Open2 swap changes calibration J from0.067450 to0.040993,
about39.2%. The best admissible Rat1/Open2 swap changes J from0.021967 to0.021309,
about3.0%. These are earlier RUN calibration objectives, NOT replay-margin
improvements, not held-out accuracy gains and not independent validation.
No identified swap was applied to replay, Q4 or simulated test observations.

## Consequence for the active goal

Validated remedy: NOT ESTABLISHED. Independent recording validation: NOT RUN.
This diagnostic explains one algorithmic failure but does not itself predict
and reduce replay-content instability. Original all-event denominators and the
requirement to preserve known-position accuracy remain binding.

An exhaustive greedy one-cell optimizer would still be stuck initially in
Rat1/Open1 and Rat2/Open1. Those pairs may require joint exchanges or a different
intervention; no global impossibility has been established. Any joint-search
experiment must be separately frozen, keep the original counts/union/overlap,
evaluate complete joint likelihoods rather than treating single-swap changes as
additive, retain all pairs and truth safeguards, and pass development before
new independent-data confirmation. Do not promote only the two favorable pairs.

## Artifacts

Server: /mnt/seagate10tb/florianpfaff/exchange-search-20260915/measurement
ManifestSHA256:c452faa5103ba77e582906721dbb7178f88f181f41f6058b3493c2dd2a33c361
Files:summary.csv,rejections.csv,four per-session proposal CSVs,report.md,
manifest.json,independent_audit.json. These contain diagnostics, no raw spike
trains or LFP. Prior failed rule:docs/exchange_content_outcome.md.
