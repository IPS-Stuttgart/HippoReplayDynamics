# Common-pipeline 2D predictive replication

## Frozen Experiment

Completed on gpuserver6000, 2026-09-08. All 4,001 PF high-MUA candidates
(eight sessions, four animals) and all 5,224 Tanni candidates (25 sessions,
five animals) from the frozen common-eligible detected-core cohort. Both use
the same existing RUN-only encoding/QC, 8 cm grids, nonoverlapping 20 ms bins,
five 70/30 cell splits and fixed physical transition parameters. No selection
on trajectory, model evidence, ripple overlap, arena size or favorable animal.

These are retrospective, previously inspected candidate cohorts, not newly
held-out animals or a prospective replication. Candidate detection used all
cells; predictive inference inside each event used training cells only.

The endpoint is properly normalized held-out cell-identity prediction,
conditioned on the held-out total spikes per bin. Posteriors are frozen before
held-out scoring. This is a sum of marginal predictive log scores, not joint
event evidence or prediction of future observations.

## Primary Result

| Contrast | PF mean [95% CI], nats | Positive animals | Tanni mean [95% CI], nats | Positive animals |
|---|---:|---:|---:|---:|
| IMM - independent positions | +1.430 [1.257, 1.616] | 4/4 | +0.873 [0.628, 1.155] | 5/5 |
| IMM - static location | +5.787 [3.016, 9.085] | 4/4 | +2.812 [1.604, 4.411] | 5/5 |
| IMM - other-event composition | +11.329 [7.130, 15.905] | 4/4 | +1.741 [-0.237, 3.812] | 3/5 |
| Real - permuted-map IMM | +0.451 [0.301, 0.596] | 4/4 | +0.129 [0.087, 0.168] | 5/5 |

PF passes the four bounded primary predictive gates. Tanni passes three, but
not superiority over other-event composition. The declared full external
replication criterion therefore **does not pass**. Do not replace it with the
favorable temporal-versus-independent contrast or per-spike sensitivity.

This is a useful narrower result: temporal-spatial regularization improves
cross-cell prediction relative to independent/static position inference in
both datasets, including correct-adjacency sensitivity. It does not establish
a new IMM mechanism, that all high-MUA events are replay, or comprehensive
superiority to position-free explanations.

The real-map permutation is shared across all rate-map columns, preserving
population-code snapshots while changing adjacency. Independent and static
scores remain invariant. This is not an alternate-environment map, and there
is only one frozen permutation per session, not a sampled null distribution.

## Baseline and Normalization Caveats

The composition baseline learns relative cell activity from other candidate
events using five chronological folds and a one-second exclusion guard.
Training exposure differs from the RUN-trained spatial model. Its performance
can reveal encoding-transfer limitations; it is not proof that spatial content
is absent. The full readout contains both RUN and recalibrated compositions.

Tanni IMM-minus-composition animal means are R2470 +2.306, R2474 -0.930,
R2478 -0.235, R2481 +5.107 and R2482 +2.456 nats. Per-spike normalization gives
an overall +0.211 [0.030, 0.393] nats/spike, but R2474 remains negative.
Normalization changes the estimand; it does not rescue the frozen raw gate.

IMM-minus-diffusion is +0.170 [0.071, 0.289] in PF and +0.853 [0.321, 1.623]
in Tanni. Diffusion-minus-independent is +1.271 [1.065, 1.478] in PF but
+0.049 [-0.845, 0.765] in Tanni. These are descriptive model contrasts under
fixed settings, not model-identity or circuit-mechanism identification.

There are 174/20,005 zero-held-out-spike splits in PF and 143/26,120 in Tanni;
they are retained in raw scores. Three event medians in each dataset have zero
held-out spikes. Per-spike ratios are undefined, not set to zero in those
splits. Median event-median held-out spike counts are 13 PF and 11 Tanni.
The RUN-QC unit counts range from 63 to 213 PF and 48 to 165 Tanni per session.

## Aggregation

Pair scores within split, then take five-split event medians, event means per
session, equal-session animal means, and equal-animal dataset means. The 5,000
hierarchical bootstrap draws resample animals, sessions and events, keeping
maps, cell partitions and composition fits fixed. Their uncertainty is omitted.
Four/five animals limit population inference. Raw differences across datasets
are not a biological interaction test. Contrasts need not add after separate
median aggregation.

## Verification and Artifacts

Producer commit: `73167c097b51aa209f9d5703581415f64e897654` (clean).
Auditor commit: `0e62b586bfaae824d1180a67402f99a6529ed82f`.
The audit reports a dirty worktree because the two new reporter files were
untracked during it; the committed producer/auditor code was unchanged.
The first audit stopped on pandas converting a singleton excluded event ID
to float. The ID-column parsing was fixed with regression tests, and the full
audit was restarted. No scoring values, folds or event selections changed.

The final independent audit passes:

- 9,225 native event-count matrices reconstructed from PF MAT/Tanni NWB spikes.
- All 184,500 analytic independent/static and 184,500 global scores recomputed.
- 792 dynamic predictions checked with a separate forward/backward solver
  across first/middle/last events, two cell splits and both maps in all sessions;
  maximum absolute discrepancy 7.483e-11 nats.
- 415,125 split contrasts, 83,025 event contrasts and all 18 hierarchical
  aggregate/CI panels reconstructed.
- Source hashes, event selection, unit partitions, fold guards and proper
  predictive/no-update flags checked. RUN maps source-checked, not refitted.

The focused producer, auditor, reporter and frozen-prediction tests pass
(38 tests). Scientific computations and tests ran on gpuserver6000.

Run directory:
`/mnt/seagate10tb/florianpfaff/conditional-2d-mua-pf-tanni-all33-20260908`

Manifest SHA256:
`d16542d0bf935929a05a335f84cd2060167c76dd8e255378beb202fb84e97386`

Independent audit and non-rescoring report are sibling directories with
`-audit` and `-report` suffixes. The report includes all animal/session means,
raw/per-spike figures, a machine-readable decision and source/output hashes.

## Decision

Retain a bounded two-dataset temporal-versus-independent predictive result,
but do not claim the complete Tanni replication gate passed. No favorable
animal/arena selection, global-baseline weakening, parameter retuning or new
speed-uniformity claim is warranted. This advances the validation program;
it does not establish a new high-importance biological finding.
