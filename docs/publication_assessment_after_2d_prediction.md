# Publication assessment after common-pipeline 2D prediction

## Answer

There is a credible scientific paper direction. The strongest established
contribution is presently a quantitative measurement/validation study, not a
new biological replay mechanism. A high-importance finding remains unestablished.

## Strongest Paper

**Recording coverage limits inference of replay continuity and spatial speed
variation.** The useful result is not just that a decoder is imperfect:

1. Keep observed candidate events fixed and hide recorded neurons. Continuity
   acceptance changes in both independent 2D datasets, across all nine animals.
2. Simulate known continuous paths and known speed gradients. Decoding and
   trajectory selection can create apparent jumps and attenuate gradients.
3. Changing time bins trades recovery of continuous trajectories against
   acceptance of randomized paths; a finer spatial grid supplies no new spikes.
4. Evaluate recovery, calibration and abstention under independent-map and
   observation mismatch, so an inconclusive decoded profile is not mistaken
   for evidence of uniform biological speed.

The transferred PF-style two-shuffle benchmark uses the same 4,001 PF and
5,224 Tanni high-MUA cohorts. Equal-animal accepted fractions fall from
22.19% to 8.53% PF and 5.77% to 1.70% Tanni under half-cell decoding. Those are
analysis outcomes, not estimates of biological replay prevalence. The source
study also has distinct detector/window cohorts and simulated-path controls;
their denominators and effects must not be pooled.

This study is maintained at
`/home/florianpfaff/HippoReplayDynamics-recording-coverage`, baseline `37ea9f39`,
with an integrated manuscript, novelty scope, matched event figures and
15-stage integrity index. Calibration has conditional successes and explicit
failures; it is not a universally validated correction or impossibility theorem.

## New Predictive Evidence

The common-pipeline experiment now covers 9,225 events, 33 sessions and nine
animals, with genuine held-out-cell prediction conditioned on spike totals.
Temporal inference improves over independent-position decoding in both PF and
Tanni, with positive animal means in 4/4 and 5/5 animals respectively. Correct
spatial adjacency also improves prediction in both datasets.

However, Tanni's advantage over a global cell-composition predictor calibrated
on other candidate events is not robust: +1.741 [-0.237, 3.812] nats, positive
in only 3/5 animals. The full frozen replication rule fails. PF passes those
bounded gates, but beating these specific comparators is not identification of
an IMM circuit mechanism. See `2d_count_conditioned_prediction_results.md`.

This is stronger evidence of predictive usefulness than merely comparing
which model best explains the same neurons used to infer a path. It supports a
secondary validation result, not proof that the biological dynamics are IMM,
that Tanni lacks replay, or that the full IMM story replicates externally.
The strict hc-11 sleep findings remain mixed/negative and are not superseded
by this different awake-2D candidate cohort.

## What Would Be New

The potential added contribution is a tested link from recording and decoder
resolution through event selection to the recoverability of a kinematic
claim, supplemented by properly separated predictive validation. It must
quantify a scientifically consequential failure mode and demonstrate when an
inference becomes supportable. Simply documenting sensitivity is insufficient.

Important direct precedents remain:

- [Silva et al. (2015)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6095134/):
  degraded place-field decoding before replay reassessment.
- [Liu et al. (2023)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10894649/):
  real unit removal followed by replay reassessment.
- [Wei et al. (2024)](https://pubmed.ncbi.nlm.nih.gov/38538143/):
  neural-decoder overconfidence and calibration, including hippocampal position.
- [Ji et al. (2026)](https://www.nature.com/articles/s41467-025-68042-3):
  explicitly discusses limited field coverage creating apparent replay jumps.
- [Maboudi et al. (2018)](https://elifesciences.org/articles/34467):
  cross-validated HMMs and distinguishing co-firing from temporal sequence order.
- [Takigawa et al. (2024)](https://elifesciences.org/articles/85635):
  replay validation using an independent content metric without known truth.
- [Huh et al. (2026)](https://www.nature.com/articles/s41467-026-74822-2):
  likelihood-based replay analysis using learned pairwise firing order.

The targeted literature check was refreshed on 2026-09-08. It is not an
exhaustive priority certification. Calling a method HMM/IMM, withholding cells,
or finding structured firing is not by itself a novelty claim.

## Claims Still Unsupported

- Constant physical replay speed throughout either environment.
- Neural-sheet propagation actively compensating for changing place-field size.
- Surprise-triggered Bayesian backward smoothing as the replay mechanism.
- A uniquely identified and fully externally replicated IMM mechanism.
- No replay in datasets where this specific predictive pipeline fails.

Do not assemble a positive biological story by selecting favorable historical
tests while omitting failed recovery, weak baselines or the Tanni replication
gate. The next publication decision is a focused assessment of the measurement
paper's added contribution with the domain collaborator, not another parameter
sweep until a preferred biological conclusion appears.
