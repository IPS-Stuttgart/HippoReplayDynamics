# Unchanged-tuning MUA selection null

Frozen before outcomes, 2026-09-15; seed 2026091517. Question: can ordinary
activity gain and selection on spikes reproduce RUN-to-candidate calibration
failure without changing any cell's spatial tuning? This is a counterexample
and mechanism diagnostic, not a model of all real replay or calibrated truth.

## Inputs and invariants

Use all eight PF first-half training encoders from the prior frontier experiment,
four animals, same valid grid, inferred 20-cm Home disc, and decoding-QC mask.
Detector population is ALL recorded sorted cells, including units not eligible
for decoding. Rate maps and rate floor 1e-4 Hz remain fixed for every condition.
The externally located original select_pfeiffer_foster_speed_candidates.py is
loaded and hashed, not reimplemented. Parameters come from its frozen manifest:
1-ms bins, Gaussian SD 10 ms, threshold mean+3SD, mean boundaries, 50-2000 ms,
ceil(0.1 * all recorded cells) active. Synthetic physical speed is zero throughout.
This preserves the detector, not the actual animal's immobility schedule.

## Known-truth spike trains

Four independent replicas per recording, each 600 s, divided into 2-s epochs.
Within each epoch, represented location is fixed; Home probability is 0.30.
Positions within each class are sampled from training-encoder grid states with
weights from known-position calibration RUN (nearest valid grid). The same
position distribution is used for independent unselected synthetic calibration.
Latent states are shared across gain conditions within a replica. Random seeds
do not depend on decoded outputs. Epoch boundaries are not detected event edges.

Generate an exact independent inhomogeneous Poisson process conditional on
state and gain. Gain g(t)=1+(peak-1)*exp(-0.5*((t-epoch_midpoint)/0.04)^2), evaluated
piecewise at 1-ms midpoints. Peak gains 1,3,6; 6 is the primary burst condition,
1 the no-gain-change control, 3 sensitivity. There are no cell-specific gain
changes, tuning drift, altered rate floors, or shared assembly perturbations.
Preserve every generated spike and the sampled latent states. We do NOT fix
population spike totals or fit gain to observed replay mismatch.

## Three reads of the same experiment

1. Fixed windows: 20 ms ending 100 ms after each predeclared epoch midpoint.
2. Selected endpoints: original detector on realization A, last four complete
   5-ms bins from each event, same convention as real candidate endpoints.
3. Independent same-window control: independent Poisson counts at precisely the
   selected windows and identical latent states/gain integrals. No second detection.
   It removes selection on the decoded spikes while preserving the selected
   timing, activity profile, and within-region represented-position distribution.

Up to 200 windows per cohort/replica, ranked by deterministic hashes before
decoding. Retain all shorter cohorts, report them explicitly. Endpoints crossing
a latent 2-s state boundary are excluded from binary fixed-state controls and
reported; no original real endpoint is modified. The simulator does not claim
to reproduce moving trajectories, biological ground truth, or event frequencies.

Primary readout: frozen gain-one independent Poisson likelihood, regional odds
Bayes factor, ternary calls BF<1/3, neutral, BF>3; silent windows neutral.
Sensitivity: oracle integrated-gain exposure in the silence term, using the
known generator gain, not fitted replay spikes. Even the oracle readout does not
condition its likelihood on selection, and good spatial likelihood need not
imply transferable ternary sensitivity/false-alarm distributions.

## Calibration comparisons

Independent model-consistent gain-one calibration: 5000 samples per region class,
same class-conditional positions as target epochs, independent counts. Also use
the previous expanded native RUN calibration with its whole-bout partition.
Apply the existing full-population ternary mixture/bootstrap compatibility code,
200 repetitions, target blocks=30-s simulation intervals. Report fit residual
(total variation), estimated and true target prevalence, bias, compatibility
at slack 0/.05/.10/1, and whether intervals contain true finite-cohort prevalence.
These are compatibility diagnostics, not guaranteed frequentist coverage.
Do not choose new thresholds to obtain a desired outcome.

Pair the selected and independent counts event by event, and report their
class-conditional call distributions and mean spike counts. Baseline-versus-
selected failure includes activity, spatial selection and spike selection;
selected-versus-independent at the same windows isolates selection on spikes.
A mismatch reproduced by this null is sufficient to refute tuning change as
the only explanation, but not sufficient to explain all real mismatch.

## Verification and scope

Test the endpoint clock, gain integrals, independent spike generation, unchanged
maps, deterministic sampling, original detector gates and zero-selection states.
Independent audit must recount saved spikes, rerun the original detector,
reconstruct oracle mean counts and independent draws, and verify saved readouts
and cohort identities. Keep per-rat/session summaries and all failures visible.
No cell-participation correction, cross-dataset scaling, or biological claim.
