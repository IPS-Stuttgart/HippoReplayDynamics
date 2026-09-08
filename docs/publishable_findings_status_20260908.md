# Publication assessment after the assembly control

## Short Answer

There is a credible methods-paper direction, but no established new biological
mechanism or high-importance-paper finding yet. Technical success and a long
list of tests do not establish scientific novelty. Keep the stronger and weaker
findings separate rather than assembling a narrative from selected positives.

## Strongest Existing Direction

**Recording coverage limits inference of replay continuity and spatial speed
variation.** The result is a measurement chain, not simply that fewer cells
make decoding harder:

1. Hold the same observed candidate event fixed and hide cells.
2. Its trajectory classification changes systematically across both 2D datasets.
3. Known continuous paths can acquire apparent jumps; real injected speed
   gradients can be flattened by decoding and trajectory selection.
4. Longer time bins improve continuous-path acceptance but also accept more
   randomized paths. Finer spatial grids do not supply missing information.
5. Recovery/calibration identifies limited conditions where a gradient or
   equivalence claim is supportable, and otherwise must abstain.

The transferred PF-style two-shuffle MUA benchmark has 4,001 PF and 5,224 Tanni
observed windows, across 33 sessions and nine animals. Equal-animal acceptance
falls from 22.19% to 8.53% in PF and 5.77% to 1.70% in Tanni when decoding with
half the cells; the primary loss occurs in all nine animals. These are analysis
acceptance fractions, not measured biological replay prevalence. The common
encoding and transferred criterion are not exact reproduction of either
original authors' pipeline.

Evidence and integrated manuscript are maintained in the recording-coverage
worktree at `/home/florianpfaff/HippoReplayDynamics-recording-coverage`, including
`docs/replay_recording_coverage_manuscript.md`,
`docs/replay_coverage_paper_prospect.md` and
`docs/replay_coverage_novelty_scope.md` (baseline `37ea9f39`).

Novelty remains bounded: degraded-decoder controls, real-cell removal, decoder
overconfidence, and limited-coverage apparent jumps all have prior work. The
potential addition is a quantitative, reproducible link through selection to
the recovery and limits of spatial-kinematic claims, not any of those ingredients
alone. This needs domain-author judgment against the direct literature, not
just additional implementation.

## Secondary Predictive Result

PF's spatial models make useful held-out cell-identity predictions when
properly conditioned on total spikes. The new comparator study finds a +11.715
nat event-aggregated advantage over a separately calibrated global cell-
composition predictor [4.049, 19.382], positive in all four rats. It also beats
the tested persistent co-firing mixtures, but those are worse than their own
global baseline. Training exposure/capacity are not matched.

This is a bounded validation result, not a unique IMM mechanism. The strict
hc-11 sleep cohort has not replicated the spatial temporal advantage. RUN
positive controls and synthetic operating checks help identify limits, but
their success does not convert the negative sleep result into biological
absence or a statistically tested RUN-versus-sleep difference.

See `position_free_assembly_prediction_results.md` for the new audited result.
Its 480-event study passed raw-count and independent predictive reconstruction.

## Claims Not Established

- Replay physical speed is uniform throughout either 2D environment.
- Neural activity adjusts propagation speed to maintain physical uniformity.
- Bayesian backward smoothing explains replay after surprising outcomes.
- A new IMM mechanism is uniquely identified or independently replicated.
- Weak candidate cohorts imply no replay in the source datasets.

The absence of a robust observed gradient is not an equivalence result. The
recording/decoder/selection recovery controls are precisely why that distinction
matters here. Do not reopen closed negative hypotheses by choosing favorable
animals, thresholds, models or events.

## Direct Literature Boundaries

- [Maboudi et al. (2018)](https://elifesciences.org/articles/34467): HMM-based
  replay without behavioral position and distinguishing co-firing from order.
- [Wei et al. (2024)](https://doi.org/10.1523/JNEUROSCI.2158-23.2024): Bayesian
  neural-decoder calibration and overconfidence.
- [Ji et al. (2026)](https://doi.org/10.1038/s41467-025-68042-3): replay dynamics,
  firing-rate adaptation and limited-coverage decoding caveats.

The broader novelty note also covers the direct 2015 decoder-degradation and
2023 unit-removal controls. Nothing here certifies first publication or predicts
journal acceptance. The high-importance-paper search remains unfulfilled.
