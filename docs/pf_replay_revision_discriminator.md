# Pfeiffer/Foster retrospective replay geometry audit

## Question and claim boundary

This analysis asks a deliberately narrower question than whether replay implements
Bayesian smoothing:

> Is the emission-only decoded path closer to the immediately preceding route
> traversed in reverse, or to the upcoming route traversed forward?

The paired score is

    (future path-fit error - reversed-past path-fit error)
    ------------------------------------------------------
      future path-fit error + reversed-past path-fit error

Positive values indicate retrospective route geometry; negative values indicate
future-route geometry. This is a retrospective-content gate only. It does not
measure a filtered belief before replay, a smoothed belief after replay,
acetylcholine, neural prediction-error signaling, or causal replay function.
A separate filtering-to-smoothing bridge is required for any smoothing claim.

## Frozen cohort and inference

The input is the 160-event Pfeiffer/Foster context table produced by
test_pf_replay_context_hypotheses.py. Both templates are available for 136
events; structural exclusions remain explicit in the event output. The primary
null and observed statistic use the identical eligible cohort.

The primary estimand gives each rat equal weight. Within a rat, events retain
their realized weights. The hierarchical interval resamples rats, sessions
within rats, and events within sampled sessions while retaining each sampled
session's realized eligible-event count.

The null keeps each decoded path fixed and shifts the paired reversed-past and
future templates by a nonzero circular offset only within the same session and
event-route relation. Singleton strata are excluded from both the observed
primary statistic and the null.

## Historical positive-control boundary

The exact PF2013 away-reward-well future-path preference cannot be reproduced
from the frozen top-20-per-session artifact. PF2013 selected hundreds of
confirmed trajectory events using its own 20 ms decoder advanced by 5 ms,
shuffle tests, event boundaries, and continuity truncation. The present event
identities and 4 ms emission/IMM posteriors do not recover that historical
cohort. The physical event position and behavior can support sensitivities, but
not the exact positive control. The manifest therefore fixes
`historical_pf2013_positive_control_exactly_reproducible=false`.

This availability flag is separate from computation integrity: the technical
gate can pass while `adjudication_ready` remains false. Candidate labels and
mechanistic interpretations require both the candidate-classifiability gate and
the historical positive control, so unavailable means
`technical_nonadjudicative`.

A post-hoc audit of the 43 paired next-movement/home-bound events (4 rats, 8
sessions) is context, not a preregistered gate. Emission-mean Euclidean geometry
was future-directed (-0.0866). A closer PF-style sensitivity used the physical
animal position, radii 15 cm then +2 cm, first segment crossings, route paths
truncated after both 10 s and 50 cm, and the longest decoded run with steps below
20 cm. Comparable ring crossings existed for only 9 emission-MAP events (3
rats/4 sessions) and 14 IMM-MAP events (3 rats/5 sessions); their equal-rat
past-minus-future angular indices were +43.95 and +21.98 degrees. Applying the
additional >=10-bin and >=40-cm displacement proxy left just 1 and 3 events.
These directional values at inadequate coverage do not substitute for the
unavailable exact control.

## Candidate recovery and abstention

Recovery uses the actual past/future templates, decoded path length, and each
event's RUN decoder error. It injects anchored AR(1) spatial noise with that
event-specific radial RMS into four pure candidates and a 50/50 mixture:

- past_reversed: the actual reversed-past template;
- future_plan: the actual upcoming-route template;
- pe_disordered: a time-disordered, high-transition-surprise surrogate built
  from an actual template;
- null_mismatched: a nonzero session-and-relation circularly mismatched actual
  template;
- mixture_50_50: the pointwise equal mixture of actual past and future paths.

The pe_disordered candidate is only a deliberately recoverable
high-transition-surprise surrogate. It is not a generative model of neural
prediction error.

Feature standardization, candidate centroids, and abstention thresholds are fit
only outside the held-out group. Both leave-one-animal-out and
leave-one-session-out confusion matrices are reported. Real events receive a
candidate label only if `adjudication_ready` passes, which requires technical
validity, candidate classifiability, and the exact historical positive control;
the two cross-validation schemes must also agree. Otherwise the output
abstains.

## Finite prediction-error diagnostic

A behavior-only Markov transition model is learned separately for every session
and held-out route fold. A 0.5 pseudocount and one explicit out-of-support target
category guarantee finite surprise. The ordered emission-only path is compared
with nontrivial within-event time permutations that preserve its point set and
first point.

Two one-sided directions are distinct:

- ordered surprise lower than the permutation null supports time-order
  coherence;
- ordered surprise higher than the permutation null is the
  high-transition-surprise direction relevant to the PE surrogate.

Neither direction by itself identifies neural prediction-error signaling.

## Reproducible commands

Quick validation:

    python scripts/test_pf_replay_revision_discriminator.py \
      --events /home/florianpfaff/HippoReplayIMM-replay-behavior-hypotheses/results/pf-replay-context-hypotheses-final/pf_replay_context_hypothesis_events.csv \
      --route-segments /home/florianpfaff/HippoReplayIMM-replay-commitment-composition/results/replay-behavior-route-primitives/replay_behavior_route_segments.csv \
      --route-points /home/florianpfaff/HippoReplayIMM-replay-commitment-composition/results/replay-behavior-route-primitives/replay_behavior_route_segment_points.csv \
      --output-dir results/pf-replay-revision-discriminator-quick \
      --permutations 99 --bootstraps 200 --pe-permutations 40 \
      --injections-per-candidate 20 --seed 20260825

Frozen run:

    python scripts/test_pf_replay_revision_discriminator.py \
      --events /home/florianpfaff/HippoReplayIMM-replay-behavior-hypotheses/results/pf-replay-context-hypotheses-final/pf_replay_context_hypothesis_events.csv \
      --route-segments /home/florianpfaff/HippoReplayIMM-replay-commitment-composition/results/replay-behavior-route-primitives/replay_behavior_route_segments.csv \
      --route-points /home/florianpfaff/HippoReplayIMM-replay-commitment-composition/results/replay-behavior-route-primitives/replay_behavior_route_segment_points.csv \
      --output-dir results/pf-replay-revision-discriminator-final \
      --permutations 2000 --bootstraps 4000 --pe-permutations 500 \
      --injections-per-candidate 200 --seed 20260825

The manifest records the code revision, dirty flag, exact command, input SHA-256
values, parameters, output paths, and output SHA-256 inventory. Do not publish
or copy conclusions from an output directory whose manifest reports a dirty
code tree or whose hashes do not match the frozen bundle.
