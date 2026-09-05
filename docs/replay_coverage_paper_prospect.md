# Paper Prospect: Measuring Replay Under Limited Recording Coverage

Status: evidence-backed manuscript direction, not a finished manuscript or
novelty certification. This note concerns the recording-coverage study, not
all historical IMM results. Numerical claims are bounded by the linked audited
experiments and their frozen settings.

## Best Current Question

When can decoded hippocampal replay support an inference about spatially
varying speed, and how does recording coverage change that answer through
localization error and trajectory-event selection?

Working title: **Recording coverage limits inference of replay continuity and
spatial speed variation**. A less claim-heavy alternative is **Measuring the
reliability of decoded replay trajectories**.

The contribution should be a quantitative measurement study with a usable
validation benchmark. It is not a demonstration that replay speed is uniform,
an explanation of the neural mechanism generating speed, or a claim that the
original experimental papers were wrong.

## What We Have Established

1. Identical observed candidate events change classification when cells are
   hidden from the decoder. The primary geometric-continuity loss is present
   in every animal across the two datasets, and survives changing MUA/ripple
   candidate definitions and core/fixed event windows. Because the underlying
   biological event is unchanged, this establishes measurement sensitivity.
   It does not identify whether the full-population or reduced-population
   classification is biologically correct.
   The completed PF-style benchmark also applies 5,000 cell-identity and
   5,000 spatial-map shuffles per tested observation: MUA acceptance falls
   from 22.19% to 8.53% in PF and from 5.77% to 1.70% in Tanni under half-cell
   decoding (equal-animal means). The primary effect is negative in all nine
   animals for MUA and ripple cohorts. This is a transferred common-encoding
   test, not exact reproduction of the original authors' analysis.
2. Known-path experiments separate biological trajectory truth from decoded
   appearance. Continuous paths can fail the continuity rule; imposed spatial
   speed gradients can appear substantially flatter. A flat decoded profile
   therefore needs a recovery/equivalence analysis, not only a test against
   zero correlation.
3. Restoring total counts using true-position-dependent labels is not a clean
   count-only control. A data-only pooling control preserves all spikes while
   losing the identity of removed cells; much of the continuity loss remains.
   Spatially tuned identity information matters beyond aggregate counts in
   this surrogate. This is not a full decomposition of spatial coverage.
4. Analysis resolution has competing consequences. Longer time windows can
   improve recovery of known continuous paths while also increasing acceptance
   of randomized paths. Finer spatial grids do not create missing information
   and do not rescue the tested gradient attenuation.
5. Apparent uncertainty calibration can fail under map/observation mismatch.
   Matching-simulation interval coverage is near nominal before selection,
   but deteriorates with disjoint RUN maps and shared gain. Selected-event
   intervals frequently abstain. Neither calibrated baseline establishes the
   declared speed-equivalence band, even on constant-speed test paths.

These statements combine controlled simulations and real recording
perturbations. Simulated paths are not replay ground truth, and simulation
draws do not increase the number of animals.

Quantitative anchors retained from the earlier prospect:

- The original fixed-MUA perturbation includes 12,141 windows across all33;
  the new detector/core-window benchmark is a distinct cohort, not a revision
  of that denominator.
- In the fixed large-arena RatInABox slice, increasing windows from 20 to 40 ms
  raises eligible continuous-path recovery from 11.34% to 64.34%, but raises
  shuffled-path acceptance from 0.52% to 18.23%.
- Full-cell recovery with independently estimated rather than known generator
  maps changes from 35.82% to 19.60% in PF and 7.94% to 6.80% in Tanni. The
  smaller Tanni change is not rat-uniform; do not present both as equal effects.
- Bin-supported, all-data conformal coverage under disjoint maps plus shared
  gain is 76.75% PF/89.40% Tanni, versus approximately 95% under matching
  simulations. Primary selected-event finite conformal output is confined to
  Rat1, with all Tanni sessions abstaining at the declared panel budget.

Exact settings, denominators and provenance are in the geometry, independent
map, speed-identifiability and shuffle-baseline results notes. These are
different experiments and must not be combined into a single effect estimate.

## What Is Not New By Itself

Decoder quality affects replay analysis. Silva, Feng and Foster (2015)
deliberately degraded place-field decoding before reassessing trajectory
events. Liu et al. (2023) explicitly removed recorded units and recomputed
replay after matching/worsening behavioral decoder error. Population removal
alone therefore cannot be this paper's novelty claim.

Bayesian neural decoders can be overconfident, and post-hoc/conformal
calibration is established; see Wei et al. (2024). Replay validation without
ground truth, discretization sensitivity, and limited coverage causing
apparent jumps also have direct prior work. The literature and exact claim
boundaries are in `replay_coverage_novelty_scope.md`.

The candidate additional contribution is the connection between these
problems: how coverage and analysis settings alter continuity selection, how
that selection changes speed-gradient recovery, and when a calibrated
analysis must abstain rather than claim uniformity. A reader should obtain
an executable way to test an inference, not merely another warning that
decoding is imperfect. We have not established priority for this entire
connection through an exhaustive review.

## Proposed Results and Figures

1. **The same event, different recorded populations.** Matched full/half
   raster and independent posterior panels, accompanied by per-animal
   continuity acceptance. Include all candidates, not only successful examples.
2. **Sensitivity under established replay criteria.** Compare geometric
   continuity with the transferred PF-style two-shuffle criterion. Show MUA
   and ripple cohorts, full/half populations and order-randomized observations
   separately. Do not call randomized acceptance a measured biological
   false-positive rate. Do not present common 8 cm encoding as exact
   reproduction of the authors' 2 cm pipeline.
3. **Known kinematics versus decoded kinematics.** Constant, increasing and
   decreasing speed profiles; stationary/discontinuous controls; MAP and
   posterior mean before and after selection. Use paired latent paths and
   report failed/absent estimates, not only surviving profiles.
4. **Why the measurement changes.** Count-preserving pooling, information
   dose, field geometry, temporal averaging and grid resolution. Separate
   experimental contrasts instead of assigning the PF/Tanni difference to
   arena size or neuronal coverage alone.
5. **When an inference is supportable.** Interval coverage, width, finite
   availability, false equivalence and abstention under matching and mismatched
   maps/observations. Held-out-animal transfer is still required before a
   generalizable calibration procedure can be claimed.

The preferred narrative is not a chronology of failed hypotheses. It is a
measurement chain: recording -> decoder -> selection -> kinematic inference.

## Methods Spine

- Two independent 2D recording datasets: eight PF sessions from four animals
  and 25 Tanni sessions from five animals. These are biological replication;
  repeated cell subsets, simulations and time bins are not.
- Common RUN-trained encoding, explicit unit/grid inclusion, independent
  uniform-prior Poisson decoding, MAP and posterior-mean sensitivity. No IMM,
  momentum prior or temporal path prior is needed for these primary analyses.
- Frozen candidate cohorts and boundaries during real-cell removal. Preserve
  zero-success sessions and LFP-unavailable sessions explicitly.
- Spikes counted from timestamps; no speed step bridges unsupported decoding
  bins. Overlapping windows are not independent observations for inference.
- Fixed-total simulations decoded conditionally; unconditional Poisson on
  fixed totals is labeled mismatch. Independently fitted RUN halves and a
  declared shared-gain stress test address additional model uncertainty.
- Equal-animal summaries, with sessions and population replicates aggregated
  within animals. Bootstrap intervals remain limited by four/five animals;
  they do not certify universality or include all sources of pipeline choice.
- Frozen protocols, source/output hashes, code commits, tests and independent
  reconstruction audits. Report the actual verification scope of each audit.

## Remaining Before Submission

- Keep the completed two-shuffle comparison and its limitations visible:
  randomized observations also sometimes pass, especially relative to the low
  acceptance of original Tanni ripple candidates. Do not equate these accepted
  fractions with biological replay prevalence.
- Hold out biological populations during the surrogate calibration-transfer
  evaluation. Existing new simulation draws within known populations are not
  equivalent to this test. Retrospective transfer is not prospective validation.
- Complete the integrated methods/results/limitations document, representative
  event panels and a reproducible artifact index. Resolve any claim-to-table
  mismatch before describing the package as paper-ready.
- Discuss the specific added contribution against the direct 2015/2023
  decoder-degradation/unit-removal and 2024 calibration literature with a
  domain collaborator. More figures alone will not establish novelty.

## Other Ideas From the Conversation

The earlier held-out IMM and order/map controls may support a separate model
comparison study, but they need their own artifact audit and positioning
against existing switching state-space replay methods. They should not be
used as evidence for constant replay speed. The proposed surprise/backward
smoothing mechanism remains a hypothesis until behavior-defined surprise,
direction/content and matched decodability controls have been tested. Neither
belongs in this paper's established-results column.

## Bottom Line

There is a credible methods-paper direction here, stronger than the current
biological uniform-speed claim. The defensible contribution is to quantify
when the recording and analysis can or cannot support a replay-kinematics
inference. Acceptance and novelty are not guaranteed, and the complete study
must meet the requirements in `replay_recording_coverage_study.md` before the
goal is considered achieved.
