# Edge-support content benchmark: frozen before decoding

## Purpose and boundary

Test an established sparse-edge baseline and support-aware variants against the
fixed raw-candidate endpoint. This is a controlled readout experiment following
the failed three-population diagnostic and hc-11 counts-only preflight. No failed
result is overwritten. Moving a readout earlier does not recover an unobserved
terminal location; assess both the new time and the original endpoint.

Pfeiffer & Foster (2013), DOI 10.1038/nature12112, adjusted sparse edges until
boundary decoding windows had >=2 spikes. Our pooled-two-spike rule implements
that activity idea on a shared fixed grid, not their complete trajectory/shuffle
protocol. Source: https://www.snn.ru.nl/~bertk/acns/pfeifer_foster_nature2013.pdf

## Frozen populations and inputs

PF: eight sessions/four rats, existing 4069 high-MUA candidate catalog. hc-11:
eight sessions/four rats, all 57460 native POST high-MUA candidates from the
audited v2 catalog. hc-11 is linear/circular track, not open-field replication.
These pools differ in behavioral state/event definition; report them separately.

For this pilot select up to 200 candidates per session by SHA256 rank of
seed=20260914, dataset, session, event ID; no spike count or decoded outcome enters
sampling. Freeze IDs before map fitting. Retain source denominator and cap
fraction. Do not replace failed sessions. Repeated cell partitions are not rats.

Refit the same 8-cm 2D occupancy-normalized maps, smoothing SD=1.5 bins, speed
>=10cm/s, occupancy>=0.05s, rate floor=1e-4Hz. Position gaps >100ms remain excluded.
Only FIRST HALF chronological RUN spikes fit the encoder and unit QC (>=30 RUN
spikes, mean<=4Hz, peak>=2Hz, first-vs-second-quarter stability>=0.25).
Native PF excitatory IDs/native hc11 CA1 pE labels are honored. Geometry may use
native full RUN tracking extent, but test spikes cannot enter rates or unit QC.
Do not use the previous full-RUN QC mask. q4 RUN supplies known-position tests.
Two equal disjoint populations, >=5 cells each, three deterministic splits;
split0 primary. All policies within a split use exactly the same cells/maps.

## Readout policies

Original candidate 5-ms grid, 20-ms overlapping windows, independent flat-prior
Poisson decoding. No HMM, momentum model, temporal smoothing or altered rate
prior. Policies choose only among existing complete windows:

1. raw_endpoint: last complete 20ms.
2. pooled_two_spike_edge: last window with >=2 spikes in A+B. This timing uses
   both populations' counts; it is NOT an A-only held-out prediction.
3. a_supported_edge: last window with >=3 A spikes from >=2 A cells. B cannot
   affect the chosen time; this is the A-only diagnostic candidate.
4. joint_supported_edge: last window meeting >=3 spikes/>=2 cells in BOTH A/B.
   This is an offline activity-support comparison, NOT held-out prediction.

No successful window means abstain; do not fall back to another policy, alter
thresholds, expand candidates or replace the split. Store each chosen index,
time shift and failure. Also store raw-window counterparts for exactly the
retained events, so dropping events cannot masquerade as a paired improvement.

## Measurements

Per policy/source/split: availability; A/B counts and active cells; posterior
entropy and width; A/B mean separation; coarse 3x3 regional posterior TV; A/B
regional probability vectors. Report both continuous uncertainty and support;
nearby means of diffuse posteriors are not proof of stable content.

Use raw A spike/cell inadequacy as a frozen, non-fitted diagnostic. Check its
association with independent B localization error and A/B disagreement. This
does not become validated merely because it detects low A counts by definition.

## Known-position and falsification controls

q4 RUN pseudo-events: fixed 200-ms windows every 500ms, supported movement
10-200cm/s, up to 200 uniformly time-spaced valid windows. Minimum32/session;
failure remains visible. Decode original and trimmed windows, comparing to the
time-averaged actual position in that window AND original endpoint window.

For each sampled candidate, retain its full 5-ms total-spike time course and
generate ONE whole-population count matrix, then split that SAME draw into A/B.
Do not simulate groups independently or redraw failed examples. Four sources:
- sim_stationary: one known occupied location throughout;
- sim_moving: 1000cm/s motion along an occupied-grid shortest path, reflected
  at its ends, continuous interpolation at 1ms resolution;
- sim_moving_gain: same type of path, per-cell lognormal gain SD=0.4;
- sim_late_jump: moving path with a new, distant location in the last20ms.

Use scipy sparse-graph shortest paths on occupied grid neighbors <=8sqrt(2)cm.
This is an occupied-map surrogate, not a verified native maze topology. Record
largest-component coverage. Interpolate adjacent path-state rates and average
within5ms; conditional multinomial sampling preserves observed totals. It does
not reproduce real noise correlations or establish replay ground truth.
Truth is the mean path position over the evaluated20ms. Report error against
both selected-window truth and original-window truth. Late jumps specifically
test the cost of claiming that an earlier readout recovers terminal content.

## Decision discipline

Aggregate events within source/session/split, sessions equally within rat,
rats equally within dataset. Show primary and sensitivity splits separately;
paired comparisons use common retained events, with all original denominators.
Bootstrap rats descriptively; with four rats per dataset avoid strong population
generalization. No selection policy or threshold is tuned on hc-11 outcomes.

A remedy for ORIGINAL endpoint content requires decreased real paired separation
and regional TV, no increased A/B entropy, no worse original-time truth error
in RUN and EVERY simulator, availability>=50% in >=3/4 external rats, and
independent reconstruction. Activity support alone cannot pass that claim.
A selected-time-only improvement must be labeled as such, NOT a terminal remedy.
The A-only flag is a validated diagnostic only if it predicts independent B
error/instability across animals beyond merely counting low-spike windows, with
useful availability. This pilot may expose another failure; report it honestly.

Freeze protocol, code, native sources and sampled IDs; detached jobs run on
gpuserver6000. Keep manifests, inputs/checksums, full output tables, failures,
independent audits and compact figures. No new biological replay claims.
