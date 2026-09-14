# Error-trained temporal context: audited development failure

The frozen error-trained choice does NOT qualify as a remedy or as a diagnostic
to advance to independent data. The requested objective remains unachieved.
No HC-11 run or original targeted Home-content remedy is authorized by this
result. No event, source, split, threshold or primary method was replaced.

## Provenance and coverage

- Server: gpuserver6000.
- Run: `/mnt/seagate10tb/florianpfaff/error-trained-context-content-20260914`.
- Scientific code/protocol frozen at `021ae0b1`; independent summary verifier
  at `3e7e4229`; exact-count audit batching at `8ec46d5d`.
- Eight PF sessions, four rats, 200 original real events/session: 1,600 primary
  real candidates. Three population splits; split 0 primary, 1/2 sensitivity.
- All six sources and four methods preserved: 115,200 event readouts.
- 16,000 split-0 known-location population rows train four leave-one-rat-out
  model sets and one all-PF set reserved for external data. Each rat's model
  excludes all its sessions and splits. No real replay labels train a model.
- Each population chooses independently using its own features, predicted
  physical-error AND Brier gains >1e-10, >=3 endpoint spikes and >=2 active cells.
- Frozen target time: the same original final 20 ms; preceding context <=200 ms.
  No cross-population conditioning, averaging or endpoint retiming.

Authoritative completed artifacts:

- `pf/manifest.json`: measurement complete.
- `pf-audit-v2/independent_audit.json`: all eight sessions passed.
- `pf-report/manifest.json`: non-rescoring, primary_advanced=false.
- `independent_summary_check.json`: reconstructed statistics and gates passed.
- `independent_summary_tamper_check.json`: altered real summary rejected even
  after its checksum was updated.

The original slow audit and its waiting finalizer were deliberately stopped
after finding repeated native spike sorting. Their partial records remain in
`pf-audit` and the original log; they are NOT completed passes. See
`error_trained_context_audit_runtime.md`. The v2 audit and v2 finalizer both
finished with exit status 0. Neither reran the scientific measurement.

## Real agreement: smaller gains than the frozen requirement

Equal-animal means, primary split:

| Metric | Independent | Error-trained choice | Relative reduction |
|---|---:|---:|---:|
| A/B endpoint separation, cm | 45.449571 | 43.832287 | 3.5584% |
| Regional total variation | 0.509727 | 0.496097 | 2.6741% |
| A normalized entropy | 0.759295 | 0.736397 | 3.0157% |
| B normalized entropy | 0.743479 | 0.718600 | 3.3463% |

Separation improves in 3/4 rats; descriptive four-rat bootstrap reduction CI
[0.0682, 3.4136] cm. Regional TV improves in only 2/4 rats; reduction CI
[-0.00712, 0.03759]. Neither achieves the frozen 10% agreement reduction.

The primary beats the entropy-matched control on separation by just over 5%,
but by less than 5% on regional TV. Agreement is not being improved by
simply broadening the reported posterior: real entropy decreases on both sides.
Nevertheless, real agreement alone is not sufficient to demonstrate accuracy.

| Rat | Independent separation | Chosen separation | Independent TV | Chosen TV |
|---|---:|---:|---:|---:|
| Rat1 | 40.267687 | 35.848672 | 0.463586 | 0.412549 |
| Rat2 | 44.928849 | 43.015169 | 0.520170 | 0.502441 |
| Rat3 | 48.109259 | 47.711972 | 0.524130 | 0.535641 |
| Rat4 | 48.492492 | 48.753336 | 0.531022 | 0.533755 |

## Known-truth falsification

Mean physical error averaged over A/B, sessions within animal and equal animals:

| Source | Independent cm | Chosen cm | Error reduction |
|---|---:|---:|---:|
| Fourth-quarter RUN | 58.761845 | 57.266441 | 2.5449% |
| Simulated stationary | 45.706495 | 40.756940 | 10.8290% |
| Simulated moving | 39.431575 | 35.436893 | 10.1307% |
| Simulated moving plus gains | 39.191436 | 35.386062 | 9.7097% |
| Simulated late jump | 82.631388 | 85.573540 | -3.5606% |

All six late-jump safeguards fail. In every rat, both populations have worse
mean physical error, p90 error and regional Brier loss. The aggregate A/B Brier
increases are about 6.1% and 6.0%. Late-jump population disagreement also grows.
These are genuine known-location failures, not unavailable measurements.

Sensitivity splits do not reverse the result. Real separation reductions are
3.69% and 2.14%; regional TV reductions 3.50% and 2.17%. Late-jump physical error
again worsens in all four rats for both sides and both sensitivity splits.

The simulated controls preserve the original total spike count per 5-ms bin,
not exact observed active-cell patterns or real noise correlations. They are
falsification controls under the stated observation model, not replay truth.

## Does the predictor recognize when context helps?

AUROC for actual positive physical-error gain, among the original supported
endpoints; first average sessions within rat, then equal rats:

| Known source | A AUROC | B AUROC |
|---|---:|---:|
| Fourth-quarter RUN | 0.635325 | 0.564970 |
| Simulated stationary | 0.673190 | 0.695924 |
| Simulated moving | 0.653939 | 0.609712 |
| Simulated moving plus gains | 0.665695 | 0.616527 |
| Simulated late jump | 0.483563 | 0.504602 |

Only RUN-A passes the required diagnostic gates. Late-jump prediction is near
chance despite explicit inclusion of late-jump training examples from other
rats. The small real gains do not establish a reliable context-choice diagnostic.

Real context-choice fractions A/B are 41.31%/44.88%, versus eligible fractions
43.00%/47.06%. For late jumps they are 35.56%/30.44%, versus eligible fractions
47.94%/43.81%. All endpoints remain in evaluation, including fallback outputs.

## Verification and decision

- 115 relevant regression tests passed; Ruff and staged whitespace checks clean.
- The independent audit reconstructed all 16,000 training rows, refitted all
  model sets, rebuilt 230,400 posterior rows, checked 26,073 native context
  blocks, and verified all 115,200 event readouts.
- The independent reporter verifier imports neither the producer nor reporter.
  It reconstructed 6,144 session metrics, 3,072 animal metrics, 768 aggregate
  rows, all 288 session diagnostic rows, animal diagnostics, and all 43 gates.
- Report manifest SHA256:
  `754fea62acba6321770cd19a8ab7dcc152adf2f7d0227018c9373c2c7315ff96`.
- Summary verifier SHA256:
  `dfc12bbe3dec5d05ee3a7b098670525979a212422113164987c35b67523e238c`.

Decision: **do not advance this method to independent validation**. Learning a
context gate from known error improves some stationary/moving controls, but
this implementation still trades fidelity to abrupt changes for small real
agreement gains. This is not a disproof of all temporal methods. It is a reason
not to tune away the jump safeguard or claim a remedy from these gains.

The next candidate must address the missing information or calibration without
assuming continuity to create agreement. The original fixed-time, targeted
Home-content contrast and independent-recording validation remain mandatory.
