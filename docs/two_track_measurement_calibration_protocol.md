# Known-generator calibration, frozen before its results

The real six-session result is mixed. Calibration must not change the frozen
real-data thresholds, select a favorable session or redefine success.

## Count anchors

Choose up to 20 supported PRE and 20 supported POST candidate windows per session
by a fixed hash of session/event identity. No replay or content score enters this
selection. Use their actual 20 ms lengths and population-specific per-bin counts.
Keep both primary sessions and all four RUN-quality diagnostic sessions.

## Known track content

Use the saved RUN maps as BOTH generating and decoding maps. The decoder gets
the maps, not the latent path or true track. Generate paired track-1 and track-2
events with the same sampled monotonic bin path through a contiguous interval
supported by both maps. These are piecewise-constant bin representations, not a
test of continuous replay speed or within-bin averaging. Both track identities
are equally represented before classification.

For each of five frozen cell splits, draw inference and independent evaluation
identities conditionally on their recorded per-bin totals. The nested half
subset uses repeat (anchor rank + split) modulo five. Whole-bin shuffled copies
preserve all spike vectors/context but destroy their order. Apply the existing
two-null sequence criterion unchanged at full and half inference coverage.
Report known-track discrimination, classification sensitivity, null acceptance,
and selected true-track proportions. Existing Poisson and count-conditional
content readouts are both retained. The generator conditions on counts: it is
not an exact Poisson-rate generative validation of the Poisson decoder.

## Known temporal generator

Use each anchor's already-fitted, held-out-event K50 parameters. Generate hidden
state paths from the learned transition or the occupancy/dwell-matched null.
Generate inference and evaluation identities independently conditional on their
recorded per-bin totals and the shared hidden state. The scorer gets the fitted
parameters but never the hidden state. Use half coverage, all five cell splits
and five fixed nested repeats; retain 20/40/80 ms and all original baselines.
Primary comparison remains the independently filtered matched null at 40 ms.

This is an optimistic model-matched measurement check, not refitted-model
recovery, validation of real replay labels, or evidence for biological bias.
Calibration failing at the observed count regime limits interpretation; more
simulation draws do not create additional animals or remove map uncertainty.
