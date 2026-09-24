# Conditional spatial-expression readout calibration

2026-09-24. Frozen before running this synthetic assay. No actual
recruitment/outcome association or drug effect is scored.

## Precisely what is measured

For a cell with a reference rate profile r_hat(b), RUN exposure T(b), and
N > 0 observed spikes with spatial counts y(b), define

q_hat(b | T) = T(b) r_hat(b) / sum_c T(c) r_hat(c)
S(y,T,r_hat) = sum_b [y(b)/N - q_hat(b | T)] log r_hat(b).

The candidate change is S(target) - S(baseline), using one reference profile
trained strictly before baseline. This is reference-centered spatial firing
alignment, in nats/spike, NOT a distance between two maps and NOT proof of
stability. Sharpening the old field can increase it even though the map changes.
Moving a field can occasionally increase it as well in a multipeaked profile.

Multiplying a cell's generating or reference profile by a positive constant
leaves conditional spatial probabilities and this statistic unchanged. A gain
change does change total counts and therefore precision. Zero-spike windows
have no conditional spatial information; return missing, never a zero score.

With a truly known unchanged profile, E[S]=0 for arbitrary exposure and N.
For estimated reference profiles this need not hold. Different baseline/target
occupancy can therefore create a nonzero expected change despite an unchanged
generating map. Quantifying that bias is a core output, not an assumption.

## Fixed data-shaped simulation bank

Take the existing opportunity inventory with original_RUN_pass, three-reference
history and >=5 reference-selected cells. Choose exactly one opportunity per
animal x drug x novelty x direction, using the minimum SHA256 key with seed
20260924. Selection uses identities, not future spatial scores or recruitment.
Keep the selected epoch/session and all denominator information explicit.

Reuse full-RUN directional profiles as numerical generating truths only.
They are not known biological maps and are not used to score a real future
outcome. The model-bank unit universe is the full-RUN encoding-unit set.
Real reference, baseline and target RUN occupancy come from the selected
opportunity, with the already specified movement/interior criteria.

For each cell, draw 16 independent Poisson reference count maps from the same
fixed generating profile and actual three-reference-traversal exposure.
Use reference gain 0.25, 1, 4. Estimate the profile with the existing 2-cm grid,
4-cm Gaussian count/occupancy smoothing and 1e-5-Hz numerical floor.
Record reference-only >=10-spike / >=1-Hz unit inclusion; do not hide failures.

This is an initial Poisson calibration. It does not yet test burst dependence,
within-lap gain drift or the biological coupling estimator.

## Counterfactuals and exact evaluation

Keep generating map, baseline occupancy and target occupancy fixed. Target cases:
- unchanged spatial profile;
- unchanged profile multiplied by four (gain-only);
- field translated by +/-20 cm (sign fixed from unit identity, no wrap);
- profile sharpened with exponent 1.5;
- profile broadened with exponent 0.5;
- spatially uniform rate.

Evaluate the same target cases with the exact generating map (oracle) and each
estimated reference map. Give neither arm actual future spikes to train on.
Compute the exact conditional expected score and sampling variance, avoiding
Monte Carlo p-values or extra simulated repeats to obtain significance.
Report SD of target-minus-baseline at N=5,20,80 spikes per readout. The conditional
mean does not depend on N; variance does. The two RUN count draws are independent
in this initial model; real temporal dependence remains unvalidated.

The old ensemble decoder's >=0.1-s raw-occupancy mask is not reused as an
exclusion gate. This is a conditional single-cell spatial likelihood, not a
latent-position decoder. Report raw and smoothed occupancy support instead.
The smoothing/extrapolation choice must be inspected through estimated-map
bias, not declared valid because it provides wider support.

## Outputs and interpretation

Preserve selected opportunities, all reference fit/inclusion records and
case/arm expected-score rows, plus summaries by animal, drug/novelty, training
gain and target perturbation. These labels index bank coverage, not biological
treatment effects. Input/output hashes and exact code commit are mandatory.

Technical checks: unchanged oracle expectation near numerical zero; positive
gain invariance; sharpening/broadening signs for oracle; finite variances;
no test counts in training. These are algebra/implementation checks, NOT a
paper biological gate or a demonstration of power in real recordings.

Compare estimated unchanged-map bias with the specified perturbation effects.
Do not call an interval containing zero equivalence or infer field stability
from positive alignment. An actual observational association needs separate
controls for baseline coding, counts, exposure, recruitment and serial dependence;
a drug interaction additionally needs animal/group and track-assignment care.
