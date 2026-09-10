# Animal-Level Robustness Of The Replay Measurement Finding

2026-09-11. Retrospective, non-rescoring sensitivity of frozen endpoints.
No event definitions, classification thresholds, predictive scores or primary
decisions were changed. Source aggregation already gave animals equal weight;
this is not a correction of event-level pseudoreplication in those summaries.

## Main Finding

The main measurement effects are not carried by a single animal. The
continuity loss, PF independent predictive advantages and total simulated
speed-gradient attenuation retain their directions after each animal is
removed. This strengthens the bounded methods result within these recording
populations. It does not turn four/five animals into a large independent
replication sample, nor change Tanni's failed predictive comparator.

| Endpoint | Dataset | All-animal mean | Range after deleting one animal | Units |
| --- | --- | ---: | --- | --- |
| MUA continuity, half minus full cells | PF | -13.668 | [-16.235, -11.227] | percentage points |
| MUA continuity, half minus full cells | Tanni | -4.073 | [-4.189, -3.959] | percentage points |
| Ripple continuity, half minus full cells | PF | -13.919 | [-16.083, -10.638] | percentage points |
| Ripple continuity, half minus full cells | Tanni | -1.569 | [-1.831, -1.161] | percentage points |
| Rejected-group IMM minus nonspatial composition | PF | +11.672 | [+9.623, +13.344] | predictive nats |
| Rejected-group order-by-map interaction | PF | +0.441 | [+0.404, +0.504] | predictive nats |
| Estimated-map minus latent true gradient response | PF | -0.679 | [-0.705, -0.621] | gradient contrast |
| Estimated-map minus latent true gradient response | Tanni | -0.879 | [-0.940, -0.838] | gradient contrast |
| Estimated-map minus finite-window true response | PF | -0.493 | [-0.528, -0.431] | gradient contrast |
| Estimated-map minus finite-window true response | Tanni | -0.686 | [-0.748, -0.645] | gradient contrast |

These ranges are deletion sensitivities, not confidence intervals or new
decoder fits. Ripple/MUA cohorts overlap and are not independent replication
samples. Gradient endpoints concern prescribed paths using empirical maps,
not observed wall-distance relationships in biological replay.

## Negative And Limiting Results

- PF's five predictive contrasts are positive in all four animals in both
  geometry groups, also with the existing per-held-out-spike normalization.
  This is group-level predictive support, not certification of each event as
  a true replay or evidence that IMM is the unique mechanism.
- Tanni's rejected-group composition contrast is negative in R2474 (-0.967
  nats) and R2478 (-0.298); the other three animals are positive. Its group
  mean stays positive under every deletion, but that does not repair the
  negative animals or the original interval crossing zero. Leave-one-out
  mean stability alone is therefore insufficient for replication.
- The additional estimated-map penalty beyond the generator-known map is
  small and nonuniform in Tanni: mean -0.024 gradient contrast, only 2/5
  animals in the loss direction, deletion range [-0.039, +0.020]. Total
  attenuation and attenuation beyond finite-window averaging remain negative
  in all five. Do not claim that separately estimating maps always worsens
  this readout in Tanni.

## Statistical Resolution

An exact directional sign test on four nonzero independent animals cannot
have p smaller than 1/16=0.0625; with five, the minimum is 1/32=0.03125.
Two-sided minima are 0.125 and 0.0625. Thus even unanimous directions in these
small samples do not support an arbitrary claim of strong distribution-free
population significance. The two negative Tanni composition signs yield a
one-sided p of 0.5 (3/5 positive), not evidence for a replicated effect.

This does not invalidate the original conditional bootstrap estimates. The
procedures use different assumptions and test different estimands. Nor does
non-significance establish no effect or uniform speed. The sign diagnostics
are retrospective and unadjusted; the endpoints and their normalizations are
correlated. They must not be counted as 50 independent confirmations.
Implementation reference: [SciPy exact binomial test](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html).

## Verification And Provenance

- Reporter and fixed protocol commit: `9f8efc445dd4451228ce9fe4eccf0720eaaf4a1b`.
- Server: gpuserver6000, no event rescoring or biological scale-up.
- Output directory:
  `/mnt/seagate10tb/florianpfaff/replay-measurement-animal-sensitivity-20260911/`.
- Audit:
  `/mnt/seagate10tb/florianpfaff/replay-measurement-animal-sensitivity-20260911-audit.json`.
- All 225 animal values were independently re-extracted from their source
  tables using the standard-library CSV parser. All 50 endpoint means and
  exact p-values were reconstructed with arithmetic/combinatorial sums;
  all 225 deletion estimates also matched. Used input/output hashes passed.
- The complete relevant software subset passed 161 tests, including 18 new
  sensitivity tests. This validates the implementation, not publication
  novelty, biological truth or every assumption of the parent simulations.

## Publication Implication

Lead with recording-dependent measurement and conditional recoverability,
not population-wide uniform biological speed. The same-event perturbations
and prescribed-truth simulations supply evidence that accepted continuity,
independent predictive structure and recoverable kinematics are distinct
measurements. The additional animal check makes their scope clearer: robust
within these recordings, with independent-animal replication still limited.
