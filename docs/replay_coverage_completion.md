# Paper Assessment and Completed Benchmark

## Answer

There is a credible methods-paper basis, but not an established biological
claim that replay speed is uniform or a newly discovered replay mechanism.

Suggested title: **Recording coverage shapes replay continuity and the
recoverability of spatial speed variation**.

The central result is a measurement chain: which neurons are observed and how
their spikes are decoded changes which fixed candidates count as trajectories;
those choices also affect recovery of known speed variation. A flat decoded
speed profile cannot be interpreted without a recovery and equivalence check.
This statement does not say that biological speed is nonuniform either.

## Strongest Evidence

- Same observed candidates, different cell subsets: with the transferred
  PF-style two-shuffle criterion, high-MUA acceptance falls from 22.19% to
  8.53% in PF and 5.77% to 1.70% in Tanni (equal-animal means). The primary
  change is negative in all nine animals, also under ripple candidate cohorts.
- Known-path simulations distinguish true trajectories from decoding errors.
  Longer windows can increase both genuine-path recovery and randomized-path
  acceptance. In one frozen large-arena slice, 20 to 40 ms gives 11.34% to
  64.34% recovery, but 0.52% to 18.23% randomized acceptance. Finer grids do
  not restore lost information in the tested gradient task.
- The completed candidate-budget test rules out a simplistic impossibility
  claim. At 300 candidates, matching local PF calibration can detect imposed
  gradients and sometimes establish synthetic equivalence. However, its
  primary coverage drops from 94.50% to 67.50% under disjoint-map/shared-gain
  stress. Tanni remains markedly less informative. More candidate events,
  more recorded cells and better model calibration are not interchangeable.

Each number belongs to its named experiment and denominator. Acceptance is
not biological replay prevalence; simulation truth is not real replay truth.
The detailed results include null acceptance, failed estimates, false
equivalence and contrary individual examples.

## What Could Be New

Not "fewer neurons make decoding worse" or "Bayesian decoders can be
overconfident": those have direct precedent. Silva et al. (2015) degraded
decoders and reassessed trajectories; Liu et al. (2023) removed units and
recomputed replay; Wei et al. (2024) studied uncertainty calibration. Ji et al.
(2026) explicitly describes coverage-induced apparent jumps. Links and
precise scope are in `replay_coverage_novelty_scope.md`.

The defensible candidate contribution is the quantitative connection between
recording coverage, continuity-based selection and speed-gradient inference,
tested on two 2D recording populations with an executable validation benchmark.
The resulting paper should state when the measurement is informative and when
it must abstain. A domain collaborator should assess whether that additional
connection is sufficiently distinct; publication is not guaranteed by the
number of analyses or passing software tests.

## Completed Deliverables

- Integrated methods/results/limitations draft:
  `docs/replay_recording_coverage_manuscript.md`.
- Claim-by-evidence and nine-requirement audit:
  `docs/replay_coverage_claim_evidence_matrix.md`.
- Exact experiment protocols and numerical results in this directory.
- Matched event atlas: 36 examples spanning nine animals and
  lost/retained/gained/rejected decisions; one middle-lexicographic member per
  category, not a random prevalence sample. Examples show timestamp rasters,
  2D posterior marginals, MAP/mean paths and the accepted geometric core.
- Final artifact registry: 15 production stages, linked manifests/audits,
  recorded top-level output checksums and explicit historical limitations.
- Test subset: 215 passed on gpuserver6000; newly added code Ruff-clean.

Server outputs are under `/mnt/seagate10tb/florianpfaff/`:

```text
replay-coverage-paper-index-20260905-final/
replay-coverage-matched-event-figures-20260905-v2/
replay-speed-panel-size-all33-20260905/report/
```

Local tables and figures are mirrored under `results/paper-index-final/`,
`results/matched-event-figures-v2/` and `results/speed-panel-size-report/`.

Visual checks covered the two new aggregate figures and representative event
panels from both datasets and all four decision categories, alongside the
previously inspected main figures and 15 detector traces. This is not a claim
that every one of the 36 examples received independent biological adjudication.
The registry checks existing artifact integrity; it does not rerun all prior
scientific audits or invent historical hashes for early artifacts.

## Stop Here Scientifically

The declared benchmark question is answered within its frozen scope. Further
experiments are not needed merely to turn mixed results into a positive story.
The next author decision is whether to develop this methods contribution with
Dan, supported by the manuscript and figure pack. An exact author-pipeline
reproduction, alternative sequence detector, prospective dataset or biological
equivalence analysis would be additional work, not already established evidence.

The historical IMM ladder may support a separate predictive model-comparison
paper, but it needs its own provenance/novelty assessment against existing
switching state-space methods. The surprise/backward-smoothing theory remains
a hypothesis. Neither should be presented as a discovery from this benchmark.
