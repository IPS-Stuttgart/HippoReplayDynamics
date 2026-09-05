# Frozen Event-Definition Sensitivity

This protocol precedes new ripple-based decoding. It extends the existing
12,141-window, 33-session coverage study, not the old dataset-specific replay
counts. No evidence, speed, continuity or posterior outcome selects candidates.

## Inputs and Detector Conditions

- Retain every original high-MUA core event and its source identity from the
  frozen all33 cache. Do not rerun the detector after subsampling neurons.
- PF: read all rows of each native `Ripple_Events.mat`. Native ripple timing is
  available; raw LFP is not in these supplied session directories. Do not claim
  identical LFP re-detection across datasets, or infer undocumented power-column
  meanings. Keep out-of-RUN/native rows with explicit exclusion flags.
- Tanni: read native downsampled LFP and timestamps at 1,500 Hz. Select stored
  channels by their literal IDs in the native CA1 channel map, without index
  shifting or outcome-based channel choice. Clock irregularity fails the session.
- Exclude constant/nonfinite channels or channels with >0.1% samples at the
  integer dtype rails. This technical screen does not establish artifact freedom.
- Filter each channel independently with zero-phase fourth-order Butterworth
  150-250 Hz, take the absolute Hilbert analytic signal, smooth with a 12.5 ms
  Gaussian, then average channel envelopes (not voltages). Standardize that
  average with its mean and population SD during tracking-supported immobility.
  These are standard z-scores of amplitude envelopes, NOT power or robust-MAD z.
- Guard one second at each LFP edge; require >=60 s valid immobile baseline.
  If baseline duration is insufficient, retain the session's MUA windows, mark
  ripple detection unavailable, and use missing rather than zero ripple counts.
  Keep the >=60 s threshold unchanged. An all-session detector comparison is
  unavailable in that case even if technical preparation succeeds.
- Ripple-like episodes exceed the baseline mean. Merge gaps <=30 ms, require
  peak >=3 SD, and label duration 15-250 ms as passing. Retain amplitude-qualified
  episodes that fail duration, rather than hiding them. Fixed-window analyses
  also require their source core to pass duration. Expose >=4/5 SD sensitivity
  using these same episodes, not newly optimized onset/window rules.
- These are ripple-like candidates, not independently confirmed sharp-wave
  ripples or replay. No native replay-state labels are required.

The PF literature detector motivates the frequency band, envelope smoothing,
immobile baseline, mean-return boundaries and 3-SD threshold; the gap merge and
duration screen above are explicit study choices, not claimed exact replication:
https://pmc.ncbi.nlm.nih.gov/articles/PMC3990408/

## Common Position and Window Rules

Use cached native-clock positions and tracking-supported RUN intervals. Compute
speed separately in each contiguous segment: interpolate to its median-sampling-
interval regular grid, finite-difference XY, then Gaussian-smooth scalar speed
with SD100 ms. Never smooth/interpolate across a gap >100 ms or a RUN boundary.
This is an additional common eligibility layer; it is not a silent replacement
of the original detector's speed calculation.

Immobility is <5 cm/s throughout the entire scoring window. Record peak-only
immobility too, so exclusions relative to older peak-only pipelines are visible.
Keep every source row regardless of eligibility. Count support from native spike
timestamps, but do not impose event-level spike counts or active-unit cutoffs.
The existing >=2-cells/>=3-spikes per-bin decoding sensitivity remains separate.

For every source event retain two variants:

1. Detected core, with its original start and end.
2. Peak-centered 200 ms, with no clipping to tracking, immobility or LFP limits.

Flag out-of-clock, tracking-gap, movement, LFP-edge and detector-duration failures.
Only eligible windows are decoded. No gap or unsupported intermediate decoding
bin may be bridged when measuring speed or continuity. Explicitly retain sessions
with zero eligible windows in all summaries.

## Overlap and Interpretation

Compute strict positive-duration overlaps separately for each window variant
between eligible MUA and eligible ripple windows. Preserve every many-to-many
edge and each unique event identity. Both-detected does not imply exact matching;
peak offsets and overlap durations are reported. Do not call event overlap
precision/recall of true replay. Different detector windows can double-count
the same underlying activity, so analyses are paired by session/animal and do
not pool those windows as independent biological replicates.

Tanni mean/maximum envelope z within MUA windows and fraction above z3 are
diagnostics. PF has no corresponding raw envelope readout in these inputs.

## Planned Decoding and Baselines

Use the frozen common RUN maps and identical nested full/half cell subsets,
independent uniform-prior decoding, MAP/posterior mean, support sensitivities,
20 ms windows at 5 ms stride. Keep overlap-dependent and detector-independent
subsets distinct. Compare paired cell-removal effects and event/window selection
effects without claiming latent replay truth. No thresholds are optimized here.

The established PF/Foster baseline additionally requires edge trimming and
cell-ID/place-field-position shuffle significance, not just geometric continuity.
It must be implemented and validated separately. Published p-values use the
fraction of shuffled candidates passing the trajectory criterion; do not silently
replace this statistic with an event-score tail probability. The original paper
used 5,000 shuffles of each type; report the actual Monte Carlo budget used here.

This preparation alone does not finish event-definition sensitivity or establish
uniform speed, calibrated replay detection, or a paper-ready endpoint.
