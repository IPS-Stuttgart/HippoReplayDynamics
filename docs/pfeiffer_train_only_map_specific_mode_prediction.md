# Pfeiffer/Foster train-only map-specific mode prediction

## Frozen question

Does map-specific first-order IMM mode allocation inferred from one neural
subpopulation predict independent replay spikes from cells that were never used
to infer the latent trajectory?

The primary analysis includes all 160 frozen Pfeiffer/Foster events. It does not
select events using all-cell IMM evidence or posterior content.

## Cell splits and maps

- Twenty deterministic repeated splits use 70% of encoding cells for training
  and 30% for held-out prediction.
- RUN place fields are fit independently per cell. The arena occupancy and grid
  are position-only quantities and may be shared across train and held-out cells.
- The real map is the fitted RUN encoding map.
- The wrong map applies one deterministic, shared permutation of occupied
  spatial bins to every cell's RUN rate map. It preserves the population
  codebook but destroys its relationship to physical adjacency.
- The permutation is generated without replay spikes and is shared between the
  training and held-out cell encodings.

## Leakage boundary

For each event, split, and map condition:

1. First-order IMM and fragmented posteriors are inferred from training-cell
   replay emissions only.
2. The normalized smoothed position posterior is frozen and hashed.
3. Held-out emissions are constructed from held-out cells only.
4. Held-out observations are scored directly under the frozen posterior as
   `sum_t log sum_x p(x_t | train) p(test_t | x_t)`.
5. No model is rescored with held-out replay spikes, and held-out observations
   cannot update the posterior.

The primary predictor is:

```text
real-map training-cell nonstationary mode mass
minus wrong-map training-cell nonstationary mode mass
```

The primary outcome is:

```text
real-map frozen held-out log score under first-order IMM
minus real-map frozen held-out log score under fragmented
```

This frozen marginal score is distinct from the existing exact conditional
joint-minus-train evidence and is used because the present test requires an
unchanged posterior.

## Primary analysis

Repeated splits are collapsed to one median predictor and outcome per event, so
events with more splits, cells, or spikes cannot dominate. The association is
reported as:

- raw Spearman correlation;
- partial Spearman correlation with rat fixed effects and controls for training
  cell count, held-out spike count, training-cell IMM posterior entropy, and
  event time-bin count;
- an extended control set that additionally includes training spike count;
- rat-cluster bootstrap confidence intervals;
- within-session predictor permutation;
- per-rat and leave-one-rat-out estimates.

Primary support requires a positive core-adjusted association, a rat-bootstrap
95% interval above zero, within-session permutation `p <= 0.05`, positive raw
direction in all four rats, positive leave-one-rat-out direction, and a positive
extended-control estimate.

## Secondary analyses

- Association with the real-minus-wrong held-out IMM-fragmented score.
- Split-level within-event association after removing event means.
- A training-defined clean-IMM sensitivity subset. A split is clean only when
  its training-cell real-map IMM-minus-fragmented evidence is at least 5.5. An
  event enters the sensitivity subset only when at least half its splits satisfy
  that training-only rule.

The all-cell frozen 108-event clean-IMM set is not used for selection in this
analysis.

## Claim boundary

Passing would support a population-generalizable relationship between
map-specific mode allocation and predictive dynamics. It would not identify a
behavioral function for IMM or prove that all IMM-winning events are replay.
Failure would leave the independently established time-order, posterior-content,
and held-out gates intact but would reject this proposed event-level mechanistic
link.
