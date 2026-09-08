# What could become a paper?

Assessment after the independently audited 2026-09-08 order by map factorial.

## Short Answer

Yes: there is a credible measurement/validation paper. There is not yet a
demonstrated new high-importance biological mechanism. This distinction is
based on the corrected, audited experiments rather than selecting the most
positive provisional numbers from the conversation.

## Strongest Candidate

**Recording coverage limits inference of replay continuity and spatial speed
variation.**

The important result is not merely that a decoder can be noisy. The existing
study links the same recording through localization, trajectory-event
selection and recovery of a speed claim:

1. Fix the observed candidates and hide neurons. Under the transferred
   PF-style two-shuffle criterion, equal-animal acceptance falls from 22.19%
   to 8.53% in PF and 5.77% to 1.70% in Tanni. The effect has the same sign
   in all nine animals. Biological events themselves have not changed.
2. Known continuous simulated paths can appear discontinuous, and imposed
   spatial speed gradients can be attenuated toward a flat decoded profile.
3. Longer time bins improve some true-trajectory recovery but also increase
   acceptance of randomized paths; a finer spatial grid does not supply
   missing neural information.
4. Calibration/recovery tests establish conditional operating ranges and
   explicit failures under map/noise mismatch. More candidates can improve
   availability, but do not guarantee correct uncertainty or equivalence.

These are complementary experiments with different denominators, not one
pooled effect estimate. Simulated trajectories are not biological replay
ground truth. The manuscript and integrity-indexed experiments are maintained
in `HippoReplayDynamics-recording-coverage`, baseline `37ea9f39`.

Potential added value: an executable benchmark for deciding whether a
recording and analysis can distinguish a meaningful spatial speed gradient
from uniform speed, with an explicit inconclusive outcome where it cannot.
This is not a universally validated correction method.

## New Positive Supporting Result

The common 2D pipeline now has proper held-out-cell prediction on all 9,225
high-MUA events from 33 sessions and nine animals. The completed order x map
factorial preserves each time bin's population counts and duration and never
uses held-out spikes to update the inferred path.

| Dataset | Benefit of original time order | Extra order benefit with correct adjacency |
|---|---:|---:|
| PF, four animals | +0.914 [0.698, 1.117] nats | +0.401 [0.271, 0.530] nats |
| Tanni, five animals | +0.565 [0.417, 0.725] nats | +0.172 [0.127, 0.222] nats |

Both contrasts are positive in every animal. These are equal-animal means
of paired predictive-score differences, with hierarchical 95% intervals, not
raw cross-dataset log evidence. The full result and limitations are in
`2d_predictive_order_map_results.md`.

This supports predictive usefulness of temporal order and spatial adjacency
in two independent datasets. It does not establish unique IMM biology or
accurate physical kinematics. Tanni still fails the parent comparison against
other-event cell composition (3/5 animals; raw interval crosses zero). Do not
present a passed shuffle test as rescuing a failed comparator. Diffusion also
loses prediction quality under shuffling, without a robust Tanni unshuffled
advantage over independent positions.

## What Is Already Known

- Deliberate decoder degradation and real unit removal before replay
  reassessment have direct precedents in
  [Silva et al. (2015)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6095134/) and
  [Liu et al. (2023)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10894649/).
- Bayesian neural decoder overconfidence/calibration is established by
  [Wei et al. (2024)](https://pubmed.ncbi.nlm.nih.gov/38538143/).
- Limited field coverage causing apparent jumps is explicitly discussed by
  [Ji et al. (2026)](https://www.nature.com/articles/s41467-025-68042-3).
- Cross-validated temporal models and controls separating co-firing from
  temporal ordering are established by
  [Maboudi et al. (2018)](https://elifesciences.org/articles/34467).

The novel contribution would have to be the demonstrated additional link
between coverage, selection, kinematic recoverability and a useful validation
procedure, not any one of those known ingredients. This targeted literature
check is not exhaustive priority certification.

## Claims Not Established

- Replay has equal physical speed throughout an environment.
- Activity adjusts its propagation across the place-cell network to enforce
  constant real-world speed.
- Surprise triggers a Bayesian backward-smoothing mechanism.
- The brain uses IMM, or the full IMM claim replicates in every dataset.
- A weak or unresolvable model result means a dataset lacks biological replay.

## Next Publication Decision

Take the integrated measurement manuscript to the domain collaborator with
three arguments: same observed events under different recording coverage;
known kinematics versus decoded/selected kinematics; and the tested range in
which an equivalence or gradient inference is supportable. Use the new
predictive results as a separately labeled validation contribution, not as
evidence for uniform speed.

Submission readiness requires a clear added contribution relative to the
direct comparator literature and a compact, usable benchmark. A high-importance
biological centerpiece remains unestablished. Further favorable model counts,
parameter sweeps or post-hoc event selection would not fix that by themselves.
