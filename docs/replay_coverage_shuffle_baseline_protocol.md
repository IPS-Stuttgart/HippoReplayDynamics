# Frozen PF-Style Shuffle-Significant Continuity Benchmark

This protocol precedes the benchmark outputs. It extends, rather than replaces,
the real detector/window perturbation and known-path recovery experiments.

## Literature and Scope

[Pfeiffer and Foster (2013)](https://www.snn.ru.nl/~bertk/acns/pfeifer_foster_nature2013.pdf)
use independent uniform-prior Poisson decoding, 20 ms windows at 5 ms strides,
edge trimming to >=2 spikes, and the longest MAP sequence with jumps <20 cm,
at least ten steps and displacement >=40 cm. Their two Monte Carlo controls
shuffle cell identity and each cell's spatial field. They count shuffles that
themselves meet the trajectory criterion, using (n+1)/(K+1), K=5,000 per family;
all reported trajectory events had p<0.02 for both.

Here the transferred criterion uses ten decoded frames, with eleven frames as
an explicit steps-versus-frames sensitivity. The paper does not specify a
spatial-shuffle boundary operator; we declare independent nonidentity circular
x/y translations, the repository's existing spatial-roll convention. This is
not proof of an exact implementation match to the authors' unpublished code.

The encoding remains our frozen common 8 cm RUN-QC maps, not the paper's 2 cm
maps, 4 cm smoothing, >5 cm/s RUN criterion and different unit inclusion. The
candidate cohorts remain the audited common-immobility MUA/native-PF-ripple/new
Tanni-ripple cores, not redetected after cell removal. No original-author event
count reproduction or criticism of their reported significance is claimed.

## Fixed Analysis

- All eligible core windows from all33 detector-preparation sessions. Primary
  Tanni ripple peak z3. Keep unavailable sessions and all geometric failures.
- Full RUN-QC population and the same three half-cell subsets (seed20260905).
  Maps, support and event boundaries stay fixed under cell removal.
- Independently decode original events and one order-randomized surrogate per
  event. Permute complete 5 ms population-vector bins before overlapping 20 ms
  decoding, leaving any genuinely partial final bin fixed. This preserves fine
  population vectors and spikes while changing order; it is not an assertion
  that every surrogate is a biological non-replay event.
- The surrogate permutation is generated once from all cells and reused for
  all subsets. Edge trimming to >=2 total retained spikes is applied separately
  within each subset/observation, before decoding/continuity testing. Do not
  discard internal low-count bins in the primary PF-style condition. Evaluate
  the >=2 cells and >=3 spikes per-bin rule as a sensitivity on the same paths.
- Score MAP continuity with ten and eleven frames, earliest longest tie rule,
  strict <20 cm jumps and >=40 cm displacement.
- For each population, use 5,000 independent cell-identity permutations and
  5,000 independent per-cell circular x/y shifts. Shuffle full rectangular maps
  before restricting to the original fixed state support. Never roll an already
  compressed list of occupied bins or impose toroidal trajectory distances.
- Cell identity shuffles may include fixed points; spatial translations exclude
  the all-zero shift per cell, following the existing repository convention.
- Each event gets K independent draws per family, shared across events within
  a session/population for efficient scoring. This creates dependence across
  events, accounted for by animal-level inference, not event-level CIs.
- Only geometrically passing original/surrogate observations need Monte Carlo
  tests. A geometric failure is explicitly `geometric_fail_not_tested`, with
  p missing, K=0 and final acceptance false. This cannot remove a possible
  accepted event. Do not treat uncomputed p-values as measured p=1.
- Acceptance requires geometric pass AND both (n+1)/(K+1)<0.02. Secondary
  alpha0.01/0.05 and bin/frame sensitivities are descriptive. No early stopping.
- A --shuffles cap exists for technical runtime tests only. Runs with K<5000
  are labeled smoke, never the completed published-budget benchmark.

Sparse SciPy matrix products may accelerate Poisson MAP likelihoods; all
calculations are float64 and checked against the existing dense decoder.
The independent auditor allows at most 1e-10 log-likelihood units below the
dense maximum for roundoff ties and counts differing tied MAP indices explicitly.
Non-tied path differences fail; primary decoding uses the sparse argmax without
post-hoc spatial tie smoothing.
No dynamics prior, speed prior or temporal path smoothing is introduced.

## Outputs and Verification

Retain every event/population/observation/criterion result, geometric and final
acceptance, shuffle successes, K, p-values, and actual selected cell IDs.
Save input count arrays, shuffle banks, binary null-acceptance arrays and a
fixed sample of decoded shuffle paths for independent reconstruction. Record
hashes, source/commit, seed, actual K, timings and versions. Technical gates
require nonempty eligible cohorts, all session/population rows and full K for
every eligible Monte Carlo test. A zero qualifying session is not missing.

Verify raw-bin counts and preservation under order permutation; verify shuffle
operators and seed/batch invariance; compare sparse and dense MAP likelihoods;
recompute p-values from saved null decisions. Report animal/session counts,
equal-animal acceptance and half-minus-full effects with animal-bootstrap CIs.
Report acceptance of order-randomized controls separately, not as an empirical
estimate of real replay false-positive rates or proof of a biological null.

## Decision and Remaining Work

Does the cell-removal effect remain after both shuffle tests? Does changing the
criterion change its size or select a different sample? Does the same test
accept appreciable order-randomized controls? Report all outcomes, with no
threshold chosen from these answers. Speed uniformity is not tested here.

An exact 2 cm author-pipeline reproduction, other published sequence detectors,
and held-out-population calibration are distinct claims. This common-encoding
benchmark must not silently stand in for them or for known-path recovery.
