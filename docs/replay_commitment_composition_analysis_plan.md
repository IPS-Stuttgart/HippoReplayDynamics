# Replay commitment versus composition analysis plan

## Frozen hypothesis

Momentum-like replay represents a committed rollout along one route, whereas
clean first-order IMM replay composes multiple familiar trajectory primitives.

This analysis does not treat future-path overlap by itself as novel. The primary
question is whether a replay event's independently validated dynamical class
predicts a distinct behavioral function.

## Phase 0: non-outcome feasibility freeze

The exact five-model evidence artifact, event identifiers, calibrated margin of
5.5 log evidence, and code commit are frozen before behavioral outcomes are
joined. Clean IMM requires first-order IMM to be exact-core best and to beat
fragmented by at least 5.5. Momentum-like requires exact-sparse momentum to be
exact-core best and to beat its runner-up by at least 5.5.

If fewer than 10 confident momentum events exist, or they span fewer than three
rats, the primary predictor is fixed to the continuous
`logZ_momentum - logZ_first_order_IMM` margin. Categorical model-class results
then remain secondary and descriptive. Thresholds will not be loosened after
behavior is inspected.

The feasibility audit writes model classification and behavioral availability
to separate tables. It must not estimate a model-by-behavior association.

## Behavior-only route primitives

Route primitives are learned from RUN position without replay data. For the
Pfeiffer/Foster open field, behavior is segmented between well visits and into
local directed transition motifs. Clustering and transition probabilities use
cross-validation so the behavior immediately surrounding a target event cannot
define that event's templates.

For maze data, explicit route identities and directions are preferred. Sleep
events may test composition but cannot test commitment to immediate future
behavior. External confirmation of commitment therefore requires awake,
immobile events with position before and after the event.

## Independent event metrics

The primary composition analysis uses only continuous, nonstationary IMM bouts.
Stationary and fragmented/jump phases are excluded rather than counted as
evidence of composition.

`composition_index` is the best-single-route physical fit error for the complete
event minus the bin-weighted mean familiar-primitive fit error of its eligible
bouts. Positive values therefore mean the bouts fit familiar local primitives
better than one route explains the complete event. `switch_alignment` tests
whether IMM mode boundaries coincide with changes in directed behavioral route
class, defined by origin-well to destination-well identity rather than by a
fine-grained subpath cluster.
`transition_surprise` measures how improbable across-bout transitions are under
RUN behavior.

`future_commitment_index` is similarity to the animal's actual next path minus
similarity to matched alternative paths. Secondary outcomes are next-goal error,
past- and future-path overlap, choice entropy before the event, and time to
departure.

## Primary tests

1. Clean IMM has a larger composition index than momentum-like replay.
2. Momentum-like replay has a larger future commitment index than clean IMM.
3. IMM switch points align with route-identity changes more than circularly
   shifted switch times preserving switch count and dwell durations.
4. Dynamical evidence predicts the next route beyond conventional replay score,
   path length, endpoint, and decoder quality.

Event-level inference uses rat/session clustering, per-rat estimates, rat
bootstrap intervals, and leave-one-rat-out sensitivity. If categorical momentum
is underpowered, tests 1 and 2 use the frozen continuous momentum-minus-IMM axis.

## Required controls

Controls include event duration, spike count, active-cell count, posterior
entropy, RUN decoder error, current position, current and next well coordinates, route
frequency, time since reward, session, and rat. Nulls include event-time circular
shifts, mode-boundary circular shifts, whole-bin replay shuffles, wrong-map
decoding, and quality-matched IMM/momentum comparisons. Training-cell posteriors
and held-out-cell predictive checks are used where feasible.

## Decision boundary

Strong support requires a quality-adjusted IMM composition effect, switch-boundary
enrichment beyond the shifted-boundary null, greater momentum commitment to the
imminent path, no single-rat/session dependence, and the predicted direction in
an independent dataset.

Composition without commitment supports a compositional-trajectory claim only.
Commitment without composition supports a direct-rollout distinction only. If
neither survives, the validated model taxonomy remains a statistical result but
does not establish behavioral specialization.
