# Exhaustive finite-path clock integration pilot

Frozen before production scoring. This is a numerical/identifiability pilot,
not a biological replay result or a substitute for the full 33-encoder study.

Use every frozen observation for these predeclared recordings:

- Pfeiffer/Foster: Rat1/Open1, first recording by identifier.
- Tanni: R2470, 2018-08-10 18:11:37, the largest arena of the first animal.

The largest Tanni environment is retained deliberately: using its smallest
environment would not adequately test the difficult spatial-integration case.
Both original teacher banks, all 50 replicates, all three mixture fractions and
all 128 observations per population are retained. No true generator paths or
labels enter likelihood scoring. Scorer geometry is exhaustive rather than
sampled, so any generating paths it contains enter only through the full prior.

## Exact prior, not a changed easier model

Enumerate every ordered pair of grid centers with distance 40-120 cm, and every
shape: straight, plus the two sinusoidal curves of amplitude +/-20% of endpoint
distance. Retain the original 801 geometry knots and locally supported-triangle
criterion. Visit both orientations explicitly, without a numerical reversal
shortcut. Unsupported geometry is counted and rejected exactly as in the
original sampler; degenerate clock integration is a hard failure, not silent
exclusion.

The original sampler fixes half of its library to straight paths and half to
curves, and rejection-samples separately within each. Therefore the exact prior
is 0.5/number_of_valid_straight_paths on each straight path and
0.5/number_of_valid_signed_curves on each curved path. A uniform distribution
over their union would CHANGE the prior and is not allowed. For reset models,
mix these families separately inside each time bin before summing log scores.

Integrate conditional cell probabilities using the same exact piecewise-linear
rate integrals. Marginalize over all paths, never maximize or restrict to
apparently favorable paths. Start-index blocks only parallelize numerical work.
Save block log sums, counts of proposed/rejected/accepted geometries, and every
accepted geometry descriptor. Independent combination must reproduce the final
five likelihood columns.

## Comparisons

Refit the population mixture on the same one-recording populations using both
existing Monte Carlo scorer banks at 1,024/4,096/8,192 paths and the exhaustive
likelihoods. Do not compare these one-recording fits with prior pooled-dataset
fits as though the event counts were equal. The 25%, 50%, 75% target mixture and
all nuisance weights/interval definitions remain unchanged. There are 76,800
frozen observations and 4,200 mixture/profile fits in this pilot.

Report bias, coverage, directional power and false directional claims with the
existing thresholds and Monte Carlo uncertainty, without requiring or promising
a positive pilot. Lower power in one-recording populations is expected; the
primary numerical question is whether estimates change materially when path
sampling error is removed. Even exact recovery in these two maps would require
extension to all encoders and broader geometry/model checks before biology.

## Verification scope

Test exhaustive geometry and weighted coherent/reset likelihoods against a
separate brute-force toy calculation, including unequal straight/curve counts,
block/chunk invariance, no valid geometry, and a tampered output.

For production, independently audit first, middle and last start-index blocks
of EACH recording, selected by block ordinal before inspecting results. Within
those blocks regenerate all geometry proposals using independent triangulation
and barycentric interpolation, and recompute likelihoods for the first two
event indices per teacher/scenario/replicate. Audit all production block hashes,
descriptor uniqueness/range/proposal counts and the complete weighted merge;
certify every population/profile optimum and recompute summaries. This is a
stratified numerical audit, not independent recomputation of every geometry or
every event likelihood in the production run.

## Scientific boundary

Neither a failed gate nor a nonsignificant speed effect proves uniform speed.
The neural clock is a conditional rate-code Hellinger metric, not all neural
dynamics. The finite teacher libraries themselves remain a possible conditional
model-mismatch source even with exact scorer integration. This must be separated
from insufficient spike information before interpreting failure.

Replay-speed scaling with path length/field spacing already has precedent:
[Forli et al., Nature 2025](https://www.nature.com/articles/s41586-025-09341-z).
That paper reports faster replay for longer flight trajectories alongside larger
fields and field separation. Our aim is a calibrated discrimination of candidate
mechanisms, not to claim novelty for a raw speed-spacing correlation.
