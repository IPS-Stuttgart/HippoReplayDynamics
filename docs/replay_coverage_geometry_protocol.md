# RatInABox Geometry and Decoder-Resolution Protocol

Status: pre-production protocol. No effect-size-based parameter selection.
This is requirement 3 of the recording-coverage study, not its completion.

## Paired Spatial Factorial

- Gaussian RatInABox 1.15.3 place cells, `widths` is Gaussian standard deviation.
- Cells 64/128, sigma 20/30/40 cm, arena area 3.7/8.75 square meters, rectangular
  aspect 1/1.4, fully crossed. Peak 20 Hz, floor 0.02 Hz, no rate multiplier.
- Eight independent synthetic populations with 24 path draws per truth class.
  Root seed 20260911. Population seeds are simulation replicates, NOT animals.
- Uniform normalized centers shared across arena geometry, and first 64 cells
  nested within 128. Same physical sigma across arenas. Increasing area changes
  spatial density, while independent N and sigma changes separate their roles.
- Identical physical paths across arena/field/N settings, centered in a common
  140 x 140 cm square inside every arena. Paths reflect at this prescribed path
  domain, NOT at each arena's walls. This is not a wall-speed hypothesis.
- Duration 200 ms, fine integration 1 ms. Continuous speed field is
  1000*(1+g*x/70) cm/s for g=-0.5/0/+0.5. Same start/direction across gradients.
  Nulls: stationary; independent positions each 5 ms; whole-5-ms-bin permutation
  of constant-speed path and exact population spike vectors together.
- No redraw for truth eligibility, decoding quality, continuity, or speed.

## Observation and Support Controls

1. Native independent Poisson counts generated for 128 cells, then first 64
   removed by exact subsetting. Poisson decoding uses the true field rates.
2. Separate common-count experiment: same Poisson(300 Hz * 1 ms) total schedule
   across field geometries/N, identities sampled proportional to true rates.
   Decode with count-conditioned multinomial likelihood. This is a distinct
   generative family; it is NOT a data-only causal count-restoration intervention.
   It can introduce information through true-position-dependent labels.

Every observed fine-bin count array is reused across grids/time windows/support
domains/estimators. Native first-64 spikes are identical to those in full128.
Shuffled controls preserve all per-cell totals and population snapshots.

Two fixed decoder supports: whole arena and common path square. The latter
separates field-density effects from expanded decoder search space; it is an
optimistic known-support control, not information available in real replay.

## Decoder Settings

Baseline on all spatial configurations: 8 cm grid, 20 ms window, 5 ms stride.
Targeted full resolution factorial: N128, sigma30, aspect1.4, BOTH areas and all
eight populations, grids 4/8/16 cm, windows10/20/40 ms, strides5/10 ms.
Centered lattice states lie inside support; grids nest at common multiples.
Rates evaluated at state centers by RatInABox, no interpolation/temporal prior.
The baseline resolution setting is scored once, not independent replication.

Independent uniform-prior decoding, MAP/mean. Support variants: all bins versus
>=2 cells and >=3 spikes. No bridging missing bins in continuity OR speed.
Speed uses non-overlapping windows separated by window duration; overlapping
stride is used for continuity only. Report error against instantaneous center
position and gradient response against both arclength and window-mean chords.

Literal continuity: strict adjacent jump <20 cm, >=10 frames, >=40 cm net
displacement, earliest longest run. At 10 ms stride also evaluate time-scaled
threshold <4000 cm/s*stride, >=45 ms center-to-center span, displacement>=40.
At 5 ms the two rules coincide and are stored once. Neither is a validated
biological replay criterion. Report setting-specific truth eligibility, overall
acceptance, and positive recovery conditional on truth eligibility separately.

## Readouts and Inference

All failed/null events remain in denominators. Event-level equal weighting,
then paired population-level contrasts; bootstrap eight population seeds.
Report continuity recovery and null acceptance, position error, support,
non-overlapping apparent speed, large jumps, entropy/RMS, and gradient response
before and after selection. Gradient slope requires >=5 events and coordinate
variance>=0.01. Missing gradient pairs remain missing, not zero. Known-coordinate
gradient is optimistic; decoded-coordinate result is a separate sensitivity.
Do not infer biological equivalence, replay prevalence, or a neural mechanism.

Main contrasts: N64-full128 within each spatial configuration; large-small at
fixed N/sigma/aspect/support; sigma and aspect sensitivity; paired grid/window/
stride changes on unchanged counts. Show both signs/nulls, not only significant
contrasts. Same-simulation settings/rows are not independent sample sizes.

## Verification and Remaining Scope

Tiny tests cover direct count windows, exact subsetting/pooling-independent
common totals, known straight speed, shuffle conservation, arbitrary stride,
gap exclusion, true-path pairing, RatInABox Gaussian width/rate semantics, and
summary missingness. Run separate smoke before a clean-commit production run.
Archive protocol/code, hashes of every path/count/grid/map and final outputs.
Independently reconstruct all observations/paths and directly recount support
for all metric rows. Inspect figures. Technical pass is not scientific support.

Still required after this experiment: empirical map-estimation/observation
mismatch, independently validated calibration/abstention, event-definition
sensitivity, baseline-method comparison and final claim-by-evidence audit.
