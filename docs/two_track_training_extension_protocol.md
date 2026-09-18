# Training-only composition calibration extension

This is a prospective bounded measurement follow-up, not a real replay analysis.
The original cross-fit run stays frozen, including its failed primary gate.

1. Only RAT3_SESS2 and RAT5_SESS2, the two frozen RUN-qualified sessions.
2. Original 40 test count anchors/session and every existing synthetic test spike
   outcome, true path, split and sequence selection remain unchanged.
3. Select at most 80 new training count anchors per PRE/POST epoch from the frozen
   bank using a fixed hash of session and event ID. Require ripple-envelope
   coverage, but not a positive ripple threshold. Exclude every original test
   anchor and all candidate windows within one second of its interval. Never
   select by neural content, decoder score or trajectory acceptance.
4. Generate one paired true-track copy per training anchor and existing split
   using the frozen known RUN map, path generator and count-conditioned spike
   generator. Both copies share the same path and observed per-bin totals. Only
   simulated evaluation-cell content is read; no sequence test is run on these
   training anchors. No latent label or path is supplied to the decoder.
5. Fit exactly the previous five-bin, pseudocount 0.5 categorical score histogram
   per track and split, with at least five distinct nonempty training anchors.
   Test on the unchanged original count anchors. The candidate bank and cell
   splits are unchanged; training-simulation anchors and test anchors are disjoint.
6. Fit known track-2 fractions 0.25/0.40/0.50/0.60/0.75. Primary POST ripple targets
   are 0.25/0.50/0.75. Preserve equal total test-anchor weight across informative
   cell splits. Do not use known labels inside the mixture optimizer.
7. Use 2,000 separately seeded train-anchor and test-anchor bootstrap draws,
   refitting the training histograms. Keep the same finite-draw rule (95%),
   truth-in-95%-interval and directional separation from 0.5 requirements.
   The Poisson readout is primary; count-conditional remains sensitivity.
8. Keep PRE/MUA, count/path strata and actual full/retained/half selection stress
   outputs visible. An empty retained group stays undefined. Passing unselected
   simulated mixtures does not validate transport into selected real events.
9. No real-data inverse correction, additional biological replication, threshold
   change or new biological conclusion is authorized. If this fails, report the
   failure; do not keep adjusting the estimator to obtain a pass.

Purpose: distinguish limited calibration-training support from failure to recover
experience composition on the existing fixed test cases. It does not solve low
real event counts, limited independent animals, model mismatch, or selection
transport. Additional simulations are not additional biological evidence.
