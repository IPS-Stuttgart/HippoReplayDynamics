# Pfeiffer/Foster ripple strength and cross-cell coordination

## Frozen hypothesis

Within detected awake ripple events, stronger ripple expression indexes how
consistently a trajectory inferred from one neural subpopulation predicts a
disjoint held-out subpopulation, beyond map-specific posterior trajectory
content and ordinary event/decoder quality.

The primary predictor is native within-epoch ripple power z-score from column 6
of `Ripple_Events.mat`. The primary outcome is the event median leakage-free
held-out first-order-IMM-minus-fragmented score from the repeated 70/30 cell
splits. Primary controls are:

- training-only real-minus-wrong-map nonstationary mode mass;
- training-cell count;
- held-out spike count;
- training IMM posterior entropy;
- event time-bin count; and
- session fixed effects.

The extended model additionally controls training spike count and absolute
real-map nonstationary mode mass. All 160 frozen events are analyzed; no event is
selected from ripple power or held-out outcomes.

## Robustness gates

Primary support requires:

- positive adjusted partial Spearman association;
- rat-cluster bootstrap 95% interval above zero;
- one-sided within-session ripple-power permutation `p <= 0.05`;
- positive adjusted direction in all four rats;
- positive direction in all four leave-one-rat-out analyses; and
- positive association under the extended controls.

A stronger map-specific refinement additionally requires ripple power to predict
the real-minus-wrong-map held-out IMM-minus-fragmented score with a positive
bootstrap interval and significant within-session permutation. A paired
rat-bootstrap compares the coordination and map-content associations, but this
contrast is not allowed to change the primary support gate.

## Native-event identity

The analysis joins every frozen `session,event_index` pair back to the native
`Ripple_Events.mat` row. Native and cell-split event start/end times must agree
within the configured numerical tolerance. The join exports raw ripple power,
whole-session z-power, and within-epoch z-power.

## Claim boundary

The processed Pfeiffer/Foster release contains ripple power only for detected
ripples. It does not contain continuous LFP. Consequently this analysis tests a
continuous strength relationship within detected events; it cannot demonstrate
that promoted off-SWR windows are physiologically ripple-negative. Every event
is also truncated by the native detection threshold (`z >= 3`), which may
attenuate associations.

Passing supports an observational cross-cell coordination association. It does
not prove that ripples causally amplify or broadcast replay trajectories.
