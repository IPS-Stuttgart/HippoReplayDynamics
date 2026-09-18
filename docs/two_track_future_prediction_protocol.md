# Two-track future-neuron prediction: frozen before execution

This completes the temporal-information check in the experience-content protocol.
It does not redefine sequenceless context support as ordered replay. It reuses
the existing learned-assembly and causal-forecast implementations used in the
earlier PF/Tanni audit; it is not an IMM novelty or speed analysis.

- Preserve every independent-detector window and every frozen inference label.
- Exclude detector cells from the model. Calibration may use the remaining
  neurons on OTHER events; target-event evaluation spikes never enter inference.
- Five chronological event folds, excluding calibration events within one second
  of any test event. Each whole target event is excluded from model fitting.
- Inherit K=50, two EM restarts, 500-iteration cap and regularization from the
  existing harness. Do not tune these settings to improve the new result.
  An unconverged fit is a reported failure, not a reason to silently change K.
- Use existing 20 ms nonoverlapping bins, full and half inference coverage,
  five fixed cell splits and five nested repeats. Keep evaluation cells fixed.
- Primary: 40 ms future-neuron conditional-identity log score over an independently
  filtered stationary-occupancy/self-transition-matched null. Also retain frozen
  posterior, shared-origin null, no-history and global population baselines;
  20/80 ms are labelled sensitivities. No future inference spikes or any target
  evaluation spikes can influence a forecast.
- Report event-median scores in nats per evaluation spike, then session means.
  Repeats are not additional events; sessions are not additional rats. Empty
  evaluation targets remain missing per-spike information.
- Primary group: POST ripple events that pass with the full inference population
  and fail at half coverage. Separate loss of opportunity from sequence-test
  failure; retain all events and full-pass summaries for interpretation.
- Independently reconstruct filtering and forecasts from saved parameters before
  trusting outputs. Synthetic tests explicitly perturb evaluation spikes and
  future inference spikes to check leakage. RUN/synthetic prediction calibration
  and uncertainty remain required before any biological promotion.

A positive future-prediction result would support temporal information in rejected
events, not prove their inferred context, a specific replay trajectory or memory
function. The stronger selective experience-content-bias question remains separate.
