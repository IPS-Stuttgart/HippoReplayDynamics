# Known-label selection transport: fixed simulation protocol

Question: can neuron removal change the true experience composition of detected
trajectories under the already frozen two-track observation/selection model?
This is a calibration and sensitivity study, not evidence of real replay truth.

## Frozen inputs and counterfactuals

- Primary sessions RAT3_SESS2 and RAT5_SESS2 only.
- The original 40 score-independent PRE/POST count anchors per session, original
  RUN maps and original per-anchor/per-split monotonic path stay fixed.
- For every anchor and existing inference/evaluation split generate exactly 20
  independently seeded spike realizations, each paired on true track 1 and 2.
  Preserve each time bin's inference and evaluation population totals in both
  copies. This is count-conditioned, known-map simulation, not a Poisson truth
  validation or new independent animals.
- Each full-population sequence test is scored once. Compare it with five nested
  half-population subsets, fixed per session/split/coverage repeat across all
  anchors and spike realizations. Unlike the small original calibration, the
  subset must not vary as a function of anchor rank.
- Include whole-population-bin-shuffled versions, preserving the counts and each
  observed population vector but removing their generated time order.
- Preserve five active inference units/five nonempty bins, 499 time and 499
  field-shift nulls, and p < .025 on both tests per track. No threshold tuning.
- Evaluation cells never enter sequence selection. Save their sequenceless
  content readout for diagnostics; simulated truth, not a decoder label, defines
  the experience-composition endpoint. No actual event is rescored.

## Endpoints and weighting

Primary stratum: original POST ripple-positive count anchors. PRE and MUA-only
are sensitivities. Report ordered and shuffled generators separately.

For balanced generating tracks, report full, half, retained (full and half),
lost (full but not half), and gained (half but not full) acceptance by true
experience. Report both the true track-2 fraction in each selected set and the
fraction that the classifier labels track 2. Their difference measures
misclassification, separate from selective acceptance.

Primary contrasts: true track-2 fraction half minus full, and retained minus
full. Also report per-experience loss rates among full positives. Undefined
denominators remain missing, not zero or pass. Save equal-anchor statistics by
averaging coverage repeats, simulation draws and cell splits before session
aggregation. Each original anchor has equal starting weight for each truth.
Simulation rows and cell splits are not independent biological replicates.

Report independent evaluation-cell true-signed content for lost ordered copies,
but do not treat sequenceless content as independently establishing a trajectory.
Known simulated trajectories and nulls supply the calibration truth here.

Use 2,000 original-anchor bootstrap draws within each epoch/ripple stratum,
keeping all simulated representations of an anchor together. These conditional
intervals do not estimate animal-population uncertainty or map-estimation error.
Report estimates separately for even and odd spike draws as Monte Carlo stability
diagnostics, not as independent cohorts or a criterion for choosing a seed.

## Decision boundary

A simulated selective-composition change shows the mechanism is possible under
the frozen maps/generator; it does not establish a replicated real biological
bias. Report failures and disagreement between sessions. Do not fit or change a
mixture estimator in this experiment, automatically correct real fractions, or
replace the original failed composition-recovery runs.
