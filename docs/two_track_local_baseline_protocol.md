# Frozen local-rest firing sensitivity

This follow-up is diagnostic, specified after the main readout but before its
own outputs. It does not change the primary acceptance thresholds or cohort.

- Keep RUN maps, detector/inference/evaluation roles, windows and labels fixed.
- For every scored event, use regular position samples within +/-60 s of its
  midpoint, in the same PRE or POST epoch, sleepbox, finite speed <=5 cm/s.
- Exclude all frozen-bank candidate windows with a 1 s guard on both sides.
  Other subthreshold bursts are not claimed to have been removed.
- Require >=20 s usable baseline. Insufficient exposure is missing, not zero.
- Estimate each evaluation cell's local rate from these outside-event spikes,
  with 0.5 pseudocount per cell. Condition on observed evaluation population
  count in each target time bin and draw 199 multinomial cell-identity nulls.
- Score observed/null arrays with the same sequenceless track readout, for both
  Poisson and conditional-count likelihoods. Positive odds indicate track 1;
  orient support using the full inference population's frozen track label.
- Use event medians across the repeated splits before session means. Report
  missing exposure and null variance explicitly. Joint 60 s occupied-time-block
  intervals remain descriptive, conditional on maps/candidates/cell splits.
- Repeat with +/-120 s as a predeclared bandwidth sensitivity, not a replacement
  chosen from results. Separate PRE/POST, ripple z>=3, z>=5 and MUA-only strata.

The local null preserves event count envelopes and slow marginal firing changes,
not assembly coactivation. Evaluation spikes inside target events never estimate
the baseline. This is not sleep-stage matching, spike waveform stability QC or
a proof that rate drift explains the main result. It can conservatively remove
real sustained context information. Passing it is not proof of ordered replay.
