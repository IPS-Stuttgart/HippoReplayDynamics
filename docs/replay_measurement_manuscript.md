# Recording Coverage Shapes Estimates of Replay Continuity and Spatial Speed Variation

Integrated evidence-backed manuscript draft, not a submission-formatted or
novelty-certified final paper. The declared experiments are completed; the
artifact index records verification scope and remaining scientific limits.

Updated 2026-09-10 with the independently audited training-only continuity
and held-out prediction follow-up. This draft preserves the earlier coverage
study at `37ea9f39` and adds a distinct validation experiment; it does not
rewrite its candidate counts, inference units or historical results.

## Abstract

Reconstructed replay trajectories are often selected for geometric continuity
before their kinematics are analyzed. We quantified how partial neuronal
sampling and decoding affect these two measurements in two independent 2D
recording populations. Hiding half the included neurons reduced acceptance of
identical high-MUA candidates under a transferred continuity-and-shuffle rule
in all nine animals. A separate training-only screen showed that geometrically
rejected Pfeiffer-Foster candidates still contained structure predictive of
held-out neuronal activity; the full comparator set was not passed in Tanni.
Prescribed-path simulations showed that longer decoding windows could improve
continuous-path recovery while increasing randomized-path acceptance. Imposed
spatial-speed variation was substantially attenuated, including with separately
estimated RUN maps. Calibration experiments found conditional recoverability
at larger candidate budgets, but failures under map and observation mismatch.
These results distinguish trajectory detectability, independent population
information and kinematic identifiability. They support recording-matched
validation before interpreting weak decoded speed variation as biological
uniformity, without establishing latent replay truth or a new replay generator.

## Main Question

How reliably can decoded hippocampal population events establish continuity
and spatial variation in represented speed when only part of the neural
population is recorded? We study the measurement chain from recorded cells,
through position decoding and trajectory selection, to kinematic inference.
We do not test a new biological replay generator or establish constant speed.

## Study Design

We reanalyze two independent 2D recording populations: eight Pfeiffer/Foster
sessions from four animals and 25 Tanni sessions from five animals. The Tanni
cohort includes several environment sizes and repeated small environments;
it is not 25 large-arena sessions. Tanni et al. reported structured variation
in field size and density with boundary proximity, motivating a distinction
between biological kinematics and location-dependent measurement resolution.
Their paper did not establish replay-speed uniformity:
[Tanni et al., 2022](https://doi.org/10.1016/j.cub.2022.06.046).

We combine four complementary comparisons. First, fixed observed candidates
are decoded after hiding cells, leaving the underlying event unchanged.
Second, known paths are used to generate observations under controlled and
empirically fitted rate maps. Third, independent simulated fit/calibration/test
panels assess whether uncertainty intervals can distinguish speed gradients
from an explicit equivalence band. Fourth, training-only continuity decisions
are evaluated against separate-neuron predictive scores. Real perturbations establish measurement
sensitivity; simulations establish error relative to surrogate truth. Neither
identifies the unknown biological replay prevalence of rejected candidates.

## Results

### Identical candidates change classification with recorded population

The initial real-cell experiment retains all 12,141 source high-MUA windows:
4,069 PF and 8,072 Tanni. Additional common tracking and whole-window immobility
rules define a separate detector-sensitivity cohort. The published-budget
two-shuffle benchmark uses 14,441 eligible core windows across MUA and ripple
cohorts; overlapping windows are not independent biological events.

With independent flat-prior Poisson MAP decoding and the transferred PF-style
continuity/significance criterion, hiding half the cells lowers acceptance:

|Dataset and candidate definition|Eligible windows|Full population %|Half population %|Paired change pp, animal-bootstrap 95% interval|
|---|---:|---:|---:|---|
|PF high MUA|4,001|22.19|8.53|-13.67 [-19.13, -8.02]|
|PF native ripple|2,581|24.86|10.94|-13.92 [-20.82, -8.70]|
|Tanni high MUA|5,224|5.77|1.70|-4.07 [-4.42, -3.73]|
|Tanni LFP ripple|2,635|2.21|0.64|-1.57 [-2.43, -0.82]|

Rates are equal-animal means, with session and population-subset averaging
within animals. The primary change is negative in all nine animals for both
candidate definitions. This is not a direct estimate of missed biological
replays, and it does not imply the full-population decision is ground truth.
The common 8 cm maps and explicit shuffle operators are not an exact
reproduction of the authors' original encoding pipeline.

Original-order acceptance exceeds whole-bin-order-randomized acceptance in
PF and Tanni MUA cohorts. That excess decreases after hiding cells in every
animal. However, Tanni ripple candidates do not establish the corresponding
original-order excess: +0.43 pp, interval [-1.55, 2.17]. Raw acceptance loss
therefore must not be relabeled as universal loss of order-specific replay.

### Geometric rejection and independent neural information can dissociate

The 4,001 PF and 5,224 Tanni immobile MUA candidates were also examined using
five fixed 70/30 cell partitions. In each split, only training-cell spikes
entered independent flat-prior Poisson MAP decoding and the geometric screen.
A further nested half of those training cells was hidden for the thinning
comparison. Candidate detection had previously used all cells; inference is
conditional on that frozen ascertainment.

For candidates failing geometry despite sufficient supported frames, the PF
IMM predictive advantage was +1.541 nats over independent positions, +5.419
over a static location and +11.672 over a nonspatial cell-composition model
calibrated on other events. Original-order prediction exceeded whole-bin
shuffles by +1.008 nats; the order-by-map interaction was +0.441. Each interval
excluded zero and each effect had a positive mean in all four animals.
The group comprised 3,542 distinct events qualifying in at least one split,
not 3,542 individually significant replay events. The 1,477 PF events whose
geometric acceptance was lost under nested thinning also passed all five
group-level predictive checks. Predictions used the larger training population,
not the reduced one.

Tanni showed positive contrasts against independent and static positions,
and positive order and order-by-map effects. However, its rejected-group
composition contrast was +1.773 [-0.245, 3.865] nats, positive in only three
of five animals. The lost-under-thinning group also failed this comparator.
Tanni therefore does not pass the joint validation rule. All original
whole-cohort replication decisions remain unchanged.

This validation does not relabel geometric failures as true continuous
replays. Structured but discontinuous activity remains possible, and a
positive group mean does not certify each event. Diffusion also showed
positive descriptive contrasts in the PF rejected group, so these data do not
identify a uniquely IMM mechanism. Results and all five primary intervals
are in `training_continuity_prediction_results.md`.

### Spatially informative identities matter beyond aggregate spike counts

In the empirical-map surrogate, pooling removed cells preserves every
fine-bin population total while withholding their individual spatial labels.
Continuity recovery remains much lower than with full labels: PF 10.50% versus
29.30%, and Tanni 2.98% versus 7.78%. This is more informative than restoring
counts with labels drawn using the known true position, which can introduce
spatial information. It supports a role for individually tuned identity
information beyond aggregate counts, not a complete decomposition of field
coverage or an explanation of all biological PF/Tanni differences.

### Continuity recovery and null acceptance can improve together

Paired RatInABox experiments isolate population count, field width, area,
aspect ratio and analysis resolution. In the declared large-arena slice,
increasing the decoding window from 20 to 40 ms increases eligible continuous
path recovery from 11.34% to 64.34%, but increases randomized-path acceptance
from 0.52% to 18.23%. Thus more continuous-looking output is not an unqualified
improvement. These are simulated-control acceptance rates, not biological
false-positive estimates. Finer 4/8/16 cm grids do not restore the tested
speed-gradient response; adding state bins does not add neural information.

### Known speed variation is attenuated by the observation-to-selection chain

Injected positive, negative and zero spatial-speed gradients distinguish
measurement attenuation from genuinely uniform speed. In the independent
RUN-map benchmark, the decoded response to a difference of 1.0 between the
two imposed gradient coefficients is 0.321 for PF and 0.121 for Tanni in the
reported full-cell, known-coordinate readout. This is a surrogate response,
not a biological wall-distance effect or a universal correction factor.
MAP, posterior mean, support filtering and selected-core readouts are retained
separately. Missing selected estimates are outcomes, not exclusions to hide.

Independent map estimation matters: using a separately estimated RUN map
instead of the generator-known map reduces full-cell constant-speed MAP
recovery from 35.82% to 19.60% in PF and from 7.94% to 6.80% in Tanni. The PF
change is rat-uniform; the smaller Tanni change is not. These comparisons
separate decoding-map values while preserving the observations and state grid.

### Calibration and candidate availability are different questions

Held-out RUN validation reconstructs actual position, but nominal 95% decoder
regions cover about 51% of PF and 61% of Tanni held-out 250 ms positions.
Localization ability is not calibrated uncertainty, and RUN coverage does
not establish replay coverage. This general issue has precedent:
[Wei et al., 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11063820/).

Matching-simulation inverse-conformal coverage is near 95% before selection,
but is 76.75% PF/89.40% Tanni under disjoint maps plus shared gain. Retrospective
excluded-animal pooling does not uniformly worsen coverage and can widen
intervals. Its primary selected-event conformal output entirely abstains at
the initial panel budget. Importantly, every missing primary calibration
statistic is explained by fewer than five contributing events in the small
11-30-candidate panels. This is not evidence that additional candidates cannot
help. No calibrated method establishes the declared +/-0.25 equivalence band
in that experiment, including constant-speed truth.

The separately frozen nested 30/100/300-candidate experiment resolves that
availability question. All 743,688 panel summaries and 3,168,000 interval
decisions independently reconstruct. With matching observations, primary
statistic availability rises from 62.38% to 100% in PF and from 4.88% to
77.68% in Tanni. More candidates therefore help: the initial abstention was
not an immutable recording-population limit.

At 300 candidates, matching local inverse-conformal coverage is 94.50% in PF,
but falls to 67.50% under independent-map/shared-gain stress. Excluded-animal
PF coverage is 91.38% / 82.25%, with wider intervals. Tanni local intervals
are finite in 67.80% of matching panels, but average width 1.353 in g, almost
the entire tested range of 1.5; excluded-animal primary output still entirely
abstains. Availability, precision and robustness remain different properties.

Matching local PF calibration detects the correct sign of g=+0.5/-0.5 in
64.38%/58.13% of 300-candidate panels, versus 1.80%/1.40% in Tanni. PF local
output declares +/-0.25 equivalence for true g=0 in 26.88%, concentrated in
two animals; no primary excluded-animal equivalence is established in either
dataset. False equivalence also occurs, particularly under mismatch, and is
reported rather than hidden. The benchmark therefore identifies conditional
successes as well as failures, not a universal inability to measure speed.
Details and uncertainty are in `replay_speed_panel_size_results.md`.

## Methods Summary

**Encoding and independent decoding.** Common occupancy-normalized maps use
8 cm bins, Gaussian smoothing SD 1.5 bins, running speed >=10 cm/s, minimum
occupancy 0.05 s and rate floor 1e-4 Hz. Pre-evidence unit QC requires >=30 RUN
spikes, mean rate <=4 Hz, peak rate >=2 Hz and split-half stability >=0.25.
PF additionally has a supplied excitatory designation; Tanni's rate-filtered
units are not described as waveform-verified pyramidal cells. RUN validation
performs map support and unit inclusion within its training fold. Replay
decoding is independent Poisson Bayes with uniform valid-bin priors: no IMM,
momentum prior or temporally propagated posterior enters the primary geometric
and speed analyses. The separate-neuron predictive validation is distinguished
below and does use temporal models.

**Events and windows.** Freeze source high-MUA identities/boundaries before
cell removal. PF native ripple tables are available, not the raw LFP needed
for common re-detection. Tanni ripple-like events use literal CA1 channel IDs,
150-250 Hz Butterworth/Hilbert amplitude envelopes, 12.5 ms smoothing and an
immobile mean/SD baseline. These are amplitude-envelope z scores, not power.
Two Tanni sessions lack the required 60 s baseline and remain explicitly
unavailable, not zero-event sessions. Whole-window <5 cm/s eligibility and
tracking/gap rules apply to core and peak-centered 200 ms windows. Technical
channel screening does not certify artifact-free physiological SWRs.

**Continuity and significance.** Geometric decoding uses 20 ms windows every
5 ms, longest continuous run with <20 cm adjacent displacement, at least ten
frames and >=40 cm end-to-end displacement. Internal >=2 cells/3 spikes is
a separate sensitivity. The PF-style benchmark adds edge support and 5,000
cell-identity and 5,000 per-cell circular spatial-map shuffles, requiring both
p values <0.02; 0.01/0.05 and ten/eleven-frame alternatives remain separate.
Numerical boundary tolerances and null operators are explicitly frozen.

**Speed and simulations.** Use every-four-frame displacement over 20 ms and
never bridge an unsupported intermediate frame. Speeds from MAP and posterior
mean are distinct estimators. Empirical-map simulations use fine 1 ms latent
paths/spikes and preserve declared source duration profiles. Conditional
multinomial decoding is used for fixed-total simulations; unconditional
Poisson on those totals is labeled misspecification. Independent RUN-half
maps and a mean-one shared-gain stress test probe mismatch; neither fully
models biological replay variability. Field-geometry factorials are paired
synthetic interventions, not additional animals.

**Kinematic inference.** Truth is v=1000*(1+g*q) cm/s for normalized horizontal
coordinate q and g in [-0.75,0.75], not physical wall distance. Fit and residual
calibration draws are disjoint from tests; B/gain observations never fit the
matching inverse models. Use Gaussian and split-conformal inverse baselines,
raw event bootstrap, fixed positive/negative/zero g tests and strict +/-0.25
equivalence with 0.10/0.50 sensitivities. Missing statistics remain infinite
calibration residuals and unbounded abstentions. All-panel coverage, finite
coverage, availability, interval width and false equivalence are co-reported.
Excluded-animal fits exclude all its sessions but still use its RUN maps to
decode its observations; no new-animal conformal guarantee is asserted.

**Inference unit and provenance.** Sessions average within animals and animals
equally; four PF and five Tanni animals limit inference. Paired animal
bootstraps are conditional and descriptive, not uncertainty from all pipeline
choices or overlapping transfer-fit refits. Simulation draws and overlapping
time bins are not biological replicates. Code commits, input/output hashes,
frozen protocols, unit tests and reconstruction audits accompany each stage.

**Separate-neuron validation.** The additional predictive experiment uses
count-conditioned multinomial observations and fixed temporal models, unlike
the independent Poisson decoder used for geometric classification and speed
analysis. The training-cell posterior predicts held-out cell identities
conditional on each held-out bin total; held-out spikes never update that
posterior. Scores sum marginal predictive log probabilities, not joint event
evidence. Classifier membership is determined inside each split; contrasts
are paired inside split, then summarized by event medians over qualifying
splits before session/animal averaging. The five simultaneous requirements
and 5,000 hierarchical bootstrap draws were fixed before stratification.
Intervals condition on the maps, ascertainment, splits and shuffles. The
geometric stage, not the full two-shuffle screen, defines these training-only
groups. Their event counts must not be substituted for those in the primary
all-cell continuity-and-shuffle experiment.

## Contribution and Limits

Decoder degradation, unit removal, replay validation without ground truth and
neural-decoder miscalibration all have direct precedent. This study cannot
claim those general ideas as new. The candidate contribution is their
quantitative connection to continuity selection and recoverability of spatial
speed variation across two independent 2D recording populations, strengthened
by separate-neuron predictive support for rejected PF candidates, with an
auditable benchmark rather than a new threshold chosen to produce replay.
The full prior-work comparison accompanies the source coverage study as
`replay_coverage_novelty_scope.md`; the concise assessment for this combined
draft is in `replay_measurement_paper_assessment.md`.

The predictive comparison also has relevant precedent: independent validation
is central to [Takigawa et al., 2024](https://elifesciences.org/articles/85635),
and alternative sequence likelihoods are studied by
[Huh et al., 2026](https://doi.org/10.1038/s41467-026-74822-2).
Neither held-out validation nor another replay metric is claimed as new in
isolation. The added evidence concerns recording-sensitive rejection and
its relation to neural information, without treating either as replay truth.

The closest methodological comparison is not simply another decoder.
Takigawa et al. compare detection yield and track discriminability after
matching an empirical randomized-data acceptance rate. Their work already
shows why more detections or a nominal significance threshold do not alone
validate replay. Our incremental question is whether recording perturbations
change continuity decisions, whether rejected groups retain separate-neuron
predictive structure, and whether a pipeline can recover prescribed physical
speed variation. These are different endpoints, not a head-to-head demonstration
that our classifier is better. The lack of biological replay ground truth
applies to both approaches.

Likewise, [van der Meer, Kemere and Diba, 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC7209917/)
already distinguish detecting nonrandom activity from comparing its content
between conditions, and identify sampling and tuning-curve differences as
confounds. We do not claim this distinction as new. The present experiments
provide quantitative operating limits for an explicit continuity-to-speed
measurement procedure in two 2D recording populations. In particular, a
nominally calibrated result under matching simulations can lose coverage
under observation/map mismatch even after additional candidates resolve
statistic availability. This limits biological interpretation, rather than
proving any existing biological speed result wrong.

The practical implication is to report three separate checks: stability of
classification under recording perturbation; independent neural evidence
conditional on candidate ascertainment; and recovery plus uncertainty coverage
for the kinematic effect of interest under plausible observation models.
Passing one does not substitute for the others. A fuzzy or adaptive continuity
rule would need its own recovery/null-acceptance and held-out validation before
being recommended; this study has not established such a rule.

In particular, these experiments do not establish: constant biological replay
speed; a neural mechanism maintaining physical speed; recording coverage as
the sole cause of dataset differences; the true replay prevalence of rejected
candidates; or a universal calibrated correction. A weak measured relationship
is not equivalent to a demonstrated small biological effect. The finite
candidate budgets, empirical duration cycling, map drift, simulated noise
families, unknown real paths and limited animal replication remain material
limitations. The study is exploratory with sequentially frozen follow-ups,
not a prospectively preregistered experiment.

## Figure and Artifact Plan

1. Identical candidate spikes under full versus hidden-cell decoding, with
   per-animal acceptance under geometric and transferred two-shuffle criteria.
   A deterministic category-selected atlas contains 36 matched examples,
   spanning all nine animals and lost/retained/gained/rejected decisions.
   Black segments distinguish the accepted geometric core from jumps elsewhere
   in the window; the x/y posterior heatmaps are 2D marginals, not 1D tracks.
2. Known continuous, discontinuous and randomized paths across the controlled
   recording/resolution factorial, including recovery/null-acceptance tradeoff.
3. Injected versus decoded spatial-speed variation, with count-preserving
   pooling and independent-map comparisons; failed estimates remain visible.
4. Calibration coverage, width and abstention under map/noise mismatch and
   excluded-animal transfer, followed by nested candidate-budget results.
5. Training-only geometric rejection versus separate-neuron prediction, with
   all five contrasts, within-animal direction and Tanni's composition failure
   visible. This panel uses a temporal predictive model; it does not supply
   the independent-decoder speed estimates in figures 1-4.

A compact four-panel overview now combines representative source-linked
results from the two-shuffle, predictive, controlled-window and independent-map
experiments. `report_replay_measurement_paper_evidence.py` reads their existing
tables without rescoring. Its CSV preserves the exact row selectors, fields,
scales, uncertainty columns and sources for every displayed value.

Detailed protocols and completed evidence are indexed by
`replay_recording_coverage_study.md` and `replay_coverage_claim_evidence_matrix.md`.
The machine-readable artifact index links exact server paths, scoring commits,
manifest hashes, reports and the actual scope of existing reconstruction
audits. It distinguishes early technical-only checks from independent audits.
The study does not guarantee novelty, editorial acceptance or biological
uniformity; those are not consequences of a technically completed benchmark.
