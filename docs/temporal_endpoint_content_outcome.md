# Temporal context stabilizes content but fails the endpoint-jump safeguard

## Decision

The frozen diffusion-plus-reset method fails the PF development advancement
gate. Do not run it on hc11 as a validated remedy, do not replace the primary
with a better-looking control, and do not relax the late-jump gate after seeing
these results. The original targeted Home-content contrast has not been reduced
by this experiment. The independent-data remedy objective remains unmet.

This is nevertheless informative progress: preceding observations improve real
population agreement and ordinary known-path recovery, but damage recovery when
the true represented location changes abruptly at the endpoint. A fixed 25%
reset probability permits jumps mathematically; it does not ensure that sparse
endpoint observations overcome the accumulated temporal prior.

## Frozen design and denominator

Computation ran on gpuserver6000 in detached systemd services. The producer was
9a9214ee; the completed independent auditor and reporter were at 0bab3dba.
The full protocol is temporal_endpoint_content_protocol.md.

Eight PF recordings, four rats, 200 previously hash-selected real candidates
per recording: 1,600 of the 4,069 source candidates. This is a development
benchmark, not the original four-session targeted Home-match cohort. Each of
five known-truth sources also has 1,600 events: held-out fourth-quarter RUN,
stationary simulation, moving simulation, moving with per-cell gains, and a
late distant jump. Three frozen equal-size disjoint population splits were
evaluated. Split0 is primary; splits1/2 are sensitivity, not extra animals.
Five methods x six sources x three splits x 1,600 events = 144,000 readout rows.

The final20ms window is fixed. Its spikes and timestamp are unchanged. The
primary adds preceding disjoint20ms observations, up to200ms total, using each
population's own spikes only, existing exact first-order filtering, sigma20cm
per20ms, and a25% uniform reset. No cross-population conditioning, future
observations, repeated-spike counting, or event reselection. Encoder/unit QC
use first-half RUN only. Known truth is the original final20ms mean location.

## Primary results

These are equal-animal means after within-session averaging, not pooled event
medians. Known physical error below averages the two populations; each side
was gated separately, including its p90 tail and regional Brier score.

| Measurement | Independent final20ms | Temporal context | Change |
|---|---:|---:|---:|
| Real A/B endpoint separation, cm | 45.450 | 37.479 | -17.54% |
| Real regional total variation | 0.50973 | 0.43028 | -15.59% |
| Held-out RUN physical error, cm | 58.762 | 46.379 | -21.07% |
| Stationary simulation error, cm | 45.706 | 25.120 | -45.04% |
| Moving simulation error, cm | 39.432 | 25.774 | -34.64% |
| Moving+gain simulation error, cm | 39.191 | 26.184 | -33.19% |
| Late-jump simulation error, cm | 82.631 | 102.341 | +23.85% |

Real separation improves in4/4 rats, by10.01,8.01,5.48,8.39cm respectively.
The descriptive rat-bootstrap95% interval for mean reduction is[6.21,9.51]cm.
Regional TV improves in4/4 rats; reduction0.07945, interval[0.05191,0.10380].
Four rats limit inference; these are not broad population guarantees.

Both real posteriors become sharper: normalized entropy A0.75930->0.68178,
B0.74348->0.66895. The entropy-matched independent control instead increases
real separation to50.582cm and regional TV to0.55915. Thus the primary's
agreement gain is not reproduced by matching posterior concentration alone.

The late-jump penalty occurs in4/4 rats for both populations. A mean error rises
79.35->98.48cm; B85.92->106.20cm. Their p90 errors rise136.60->173.07cm and
137.95->174.90cm. Regional Brier scores also worsen,0.50976->0.68042 and
0.56798->0.75504. All six frozen late-jump gates fail. The baseline itself has
large jump errors, so neither method gives reliable jump localization here.

Secondary splits reproduce both directions: real separation falls15.18-17.72%
and TV14.02-15.18%, while late-jump errors rise23.37-25.08%. They cannot replace
the primary failure. In RUN itself, truth accuracy improves while A/B separation
increases40.09->41.59cm and TV0.45111->0.46927, again showing that agreement
and accuracy are distinct quantities.

## Verification

All eight recordings passed independent dense posterior reconstruction:
288,000 per-population posterior rows,144,000 readout rows,26,073 native context
blocks recounted from original timestamps. Source hashes, cell IDs, partitions,
fixed endpoint clocks, posterior probabilities, entropy controls, and readout
metrics were checked. The independent flat-prior output also matches the prior
benchmark's unchanged endpoint. A separate aggregate calculation reproduced
all960 reported rows, including equal-animal means, paired reductions, seeded
rat-bootstrap intervals and per-rat improvement counts, directly from the
144,000 input readouts without importing the reporter. All66 relevant tests and
Ruff passed. The
Matplotlib Axes3D warning is unrelated to these2D figures, which were inspected.

Earlier failed audits are retained, not overwritten:

- Initial audit stopped for redundant spike-sorting performance; batching kept
  verification scope unchanged.
- v2 exposed an integer JSON-serialization bug in the audit status record.
- v3 exposed a one-spike half-open boundary discrepancy in the recount. Reusing
  original source edges fixed it; no spike timestamps or source counts changed.
- v4 used a raw squared-distance Gaussian cutoff; the actual runtime wrapper
  scales coordinates before subtracting. Some exact-radius fractional-grid
  pairs differ numerically. The v5 independent dense implementation reproduces
  that convention, with explicit source hashes and regression tests. Neither
  scoring output nor comparison tolerances were changed to obtain passage.

Tracked analysis files were committed. The dirty provenance flag reflects
unrelated untracked regional-content-bounds work, left untouched.

## Claim boundary and next mechanism

We have not established the true locations represented by real replay or proved
that its disagreement is decoding error. Conditional simulations preserve
whole-population5ms totals, not actual active-cell patterns/noise correlations.
Their abrupt-jump source is an intentional falsification condition, not a claim
about the prevalence of such jumps in real replay.

The result motivates a diagnostic of when temporal context is unsafe, not a
stronger universal continuity prior. Any new adaptive use of context must be
frozen separately, tested without moving endpoints, retain the abrupt-jump
control, and pass independent recordings before addressing the original Home
contrast. Do not relabel the current development improvement as a validated
remedy or narrower biological success.

## Artifacts

Server root: /mnt/seagate10tb/florianpfaff/temporal-endpoint-content-20260914/

- pf/: original measurements, every posterior, input hashes and frozen IDs.
- pf-audit-v5/: completed independent reconstruction and source provenance.
- pf-report/: gate_summary.csv, summary.csv, by_animal.csv, by_session.csv,
  denominators.csv, temporal_endpoint_falsification.png, report.md, manifest.json.

Compact copies, without raw spikes or large posterior arrays:
/mnt/c/Users/emper/Documents/codex/2026-09-14/temporal-endpoint-content/
