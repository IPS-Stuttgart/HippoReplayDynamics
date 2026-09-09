# Exhaustive Path Integration: Numerical Bias Is Not The Whole Problem

## Decision

Exhaustive integration substantially improves low-fraction recovery in the
selected Pfeiffer/Foster recording, but it does not establish practical clock
identifiability. All four primary recovery gates fail. The largest-arena Tanni
recording remains weak and sensitive to the simulation's source path library.
An exploratory pooled-sample check makes some incorrect estimates more precise.
Do not interpret this as biological uniform-speed evidence, a neural-clock
prevalence estimate, or a successful uncertainty-aware continuity classifier.

This is a simulation calibration result on two recording encoders, not a
new biological or publication-novelty finding. It narrows the implementation
problem: random candidate-path sampling contributed to earlier failures, but
removing that approximation alone is insufficient.

## Frozen Primary Experiment

The protocol was committed before the run at
`887b95169694828ce3b64f0e5715706d962b737d`.
The run used gpuserver6000, 24 workers, and finished in 2,110.7 seconds.

- PF Rat1/Open1: 120,143 straight and 219,091 signed curved paths.
- Tanni R2470, 2018-08-10_18-11-37, largest arena: 562,463 straight and
  1,084,775 signed curved paths.
- Total: 1,986,472 admissible paths from 2,261,994 geometry proposals.
- Every ordered grid-center pair 40-120 cm apart, straight or signed
  sinusoidal bend, with the same supported local geometry as the original
  sampler. Both path orientations were computed explicitly.
- Straight and curved families each have prior mass 0.5, normalized within
  family. Reset models mix these families separately in every time bin.
- Exactly the existing 76,800 simulation observations were reused: two source
  banks, three mixture fractions, 50 repetitions, 128 events per population,
  in each recording. No real event was rescored or selected by model evidence.

The models distinguish constant traversal speed in physical coordinates from
constant traversal speed in normalized cell-identity probability space.
The latter uses the Hellinger arc length of the population code. Stationary
and two independent-per-bin reset controls also enter the mixture. All five
weights are fitted. The reported fraction is neural / (neural + physical),
not a speed in cm/s and not a fraction of all events.

Exhaustive means exhaustive for this finite endpoint/curve prior. It does
not cover every possible biological path, curvature, duration, or firing model.

## Primary Recovery

Same 128-event populations, not the earlier all-recording pooled ensembles:

| Recording | Source bank | True fraction | MC 8,192 bank 0 | MC 8,192 bank 1 | Exhaustive |
| --- | ---: | ---: | ---: | ---: | ---: |
| PF | 0 | 0.25 | 0.414 | 0.430 | 0.295 |
| PF | 1 | 0.25 | 0.438 | 0.470 | 0.333 |
| Tanni large | 0 | 0.25 | 0.284 | 0.303 | 0.163 |
| Tanni large | 1 | 0.25 | 0.388 | 0.431 | 0.333 |

The complete report retains all three true fractions and all seven numerical
integration methods. The table above illustrates the low-fraction bias; it
must not substitute for the full recovery curves.

Unchanged gates require worst absolute bias <=0.10, coverage >=0.90,
directional power >=0.80 and false direction at true 0.50 <=0.05.

| Recording | Source bank | Worst absolute bias | Min coverage | Min power | False direction | Pass |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| PF | 0 | 0.095 | 96% | 12% | 2% | No |
| PF | 1 | 0.095 | 92% | 12% | 6% | No |
| Tanni large | 0 | 0.291 | 98% | 0% | 0% | No |
| Tanni large | 1 | 0.108 | 98% | 0% | 0% | No |

Broad intervals can cover the truth while providing almost no directional
power. Fifty repetitions have appreciable Monte Carlo uncertainty; the
underlying tables preserve Wilson intervals. In particular, 3/50 false
directions versus a 5% target is not decisive evidence of miscalibration by
itself. The poor power is the much clearer practical failure here.

## Exploratory Larger-Sample Diagnostic

Added after inspecting the primary results, at
`f35680fa92d66e781436f470990292bb116da8f5`.
No likelihoods were recomputed and no primary gate was changed. Pooling all
50 repetitions yields 6,400 observations per recording, source bank and true
fraction. These observations still reuse the same 256-path source bank.

At true fraction 0.50, exhaustive integration gives:

| Recording | Source bank | Estimate | Conditional profile interval |
| --- | ---: | ---: | --- |
| PF | 0 | 0.597 | [0.533, 0.660] |
| PF | 1 | 0.599 | [0.530, 0.667] |
| Tanni large | 0 | 0.215 | [0.050, 0.383] |
| Tanni large | 1 | 0.503 | [0.326, 0.680] |

The first three intervals exclude the known truth. Pooling therefore does
not remove the problem. These are conditional model intervals, not intervals
validated under source-library mismatch or estimates of between-animal
uncertainty. One pooled fit per condition cannot estimate interval coverage.

A finite source bank is a sampled distribution of paths, not exactly the
exhaustive scoring prior. More observations drawn from that same bank reduce
spike-sampling noise without averaging away its particular path composition.
This is a plausible remaining bias source, not a causal attribution proven
by the present experiment. Other likelihood/model mismatches remain possible.

The next discriminating calibration would draw a fresh path for every
simulated observation directly from the scoring prior, rather than reuse
256-path libraries. Keep all five generators, both recording encoders, all
three mixture fractions and the unchanged recovery gates. Do not change
biological event selection or remove the weak Tanni recording to force a pass.

## Verification And Artifacts

The independent audit passed. It regenerated 59,785 paths in the predeclared
first/middle/last start-index blocks for both recordings, recomputed 139,176
partial likelihood sums, checked all 384,000 merged likelihood values, and
certified all 4,200 primary mixture/profile optima. Maximum recomputation
difference was 4.11e-10. All original observation vectors were verified equal
to the parent run. This does not independently rescore every path/event pair.
All 36 exploratory pooled fits also passed independent optimum and profile
certification. The non-rescoring reporter was committed at `2b12d8dc`.

All 132 primary output hashes, eight report hashes and two pooled-report
hashes verified. The 2040 x 1530 PNG was visually checked; pixel standard
deviation was 38.67. Final regression: 219 tests passed, Ruff check and format
passed for 21 relevant files. No changes were pushed.

Server artifact prefix, under `/mnt/seagate10tb/florianpfaff/`:
`exact-path-clock-recovery-pf-tanni-large-20260910`.
Sibling suffixes: `-audit`, `-report`, `-pooled-diagnostic`,
`-validation-pooled`.

Manifest SHA256 values:

- Run: `392208352fa8228982a082fea3d4697964149eb04fa52c754c501e83f04e1ac4`.
- Audit: `18188962c2d729389e05c48c5fc9895771fefea6eb2b4f42d9243955642fac2e`.
- Report: `a1f5e311c054aade92de6c2f27972b7808c009fc0108f76afe4b6b50f08efa70`.
- Pooled diagnostic: `de51245fd57df635b160f58342e9ff947d461b20055a58d4dc7475e8a2a34cea`.

This study is distinct from the earlier soft-continuity calibration tests.
It addresses unknown-path physical-versus-code-clock inference, not whether
fuzzy membership stabilizes trajectory labels under cell thinning. Neither
problem is solved by these results, and no uniform-speed claim is warranted.
