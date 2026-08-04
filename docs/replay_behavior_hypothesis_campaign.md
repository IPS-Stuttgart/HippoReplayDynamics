# Replay behavior hypothesis campaign

## Scope and claim boundary

This campaign tests ten hypotheses named before outcome inspection. It does not
change the frozen five-model evidence rows, the 5.5-log-evidence confidence
threshold, or the independently validated Pfeiffer/Foster event cohort.

The primary model axis is continuous:

```text
momentum_axis = logZ_exact_sparse_momentum - logZ_first_order_IMM
```

Positive values favor one persistent-velocity process; negative values favor
the switching model. Categorical clean-IMM and confident-momentum results are
secondary because only seven of the 160 frozen Pfeiffer/Foster events are
confident momentum winners.

Whenever represented path content is an outcome, the primary path estimator is
the emission-only posterior mean. This avoids defining behavioral content with
the IMM path whose evidence appears on the predictor axis. IMM-derived paths
are sensitivity analyses only.

All event-level analyses control, where estimable, for duration, spike count,
active-cell count, posterior entropy, trajectory-minus-stationary evidence,
path length, RUN decoding error, session, and rat. Inference uses rat-cluster
bootstrap intervals, per-rat directions, leave-one-rat-out checks, and a null
that preserves the relevant grouping structure. Benjamini-Hochberg correction
is applied once across the ten primary hypothesis p-values. An unavailable or
underpowered test remains `insufficient`; it is never treated as support.

## Frozen hypotheses and primary tests

### H1: Replay commitment during a pause

Define a pause as the interval after the preceding route's movement ends and
before the next route's movement starts. Analyze pauses containing at least two
frozen events. Event rank is normalized from zero for the first event to one for
the final event.

Primary test: the within-pause slope of `momentum_axis` versus normalized event
rank is positive. Null: permute event order within each pause.

Companion test: the final event has a larger emission-only future-commitment
index than earlier events. These two components must both point in the predicted
direction for strong support.

### H2: Prospective planning versus retrospective reinstatement

Construct behavior-only templates starting at the animal's event location:

* future template: the upcoming route in its executed direction;
* past template: the preceding route in reverse, corresponding to replay from
  the current location back through the just-completed path.

Compute direction-preserving path-fit error using the emission-only posterior
mean and define:

```text
prospective_index_cm = past_template_error_cm - future_template_error_cm
```

Primary test: `prospective_index_cm` decreases with `momentum_axis`; clean-IMM
evidence is therefore associated with future rather than past content. Null:
circularly shift event-to-behavior-route assignments within session.

### H3: Novel route construction

Route novelty is learned from RUN behavior only while excluding the event's
enclosing cross-validation fold. The primary novelty metric is the minimum
direction-preserving distance between the upcoming route and any retained
behavioral route beginning near the same origin. Binary unseen origin-to-goal
identity and cross-validated route frequency are sensitivities.

Primary test: `momentum_axis` decreases as route novelty increases. Null:
permute route novelty within session.

### H4: Goal certainty controls dynamics

The fixed Home well is inferred independently as the unique well present in
every Home/Away route in a session. A route ending at Home is `home_bound`; a
route leaving Home for a changing destination is `away_bound`.

Primary test: the adjusted Home-minus-Away difference in `momentum_axis` is
nonzero. The direction is deliberately two-sided and must be reported rather
than assumed. Null: permute Home/Away labels within session while preserving
their counts.

### H5: Policy branching evokes IMM switching

For each cross-validation fold, compute Shannon entropy of outgoing RUN
transitions at each spatial bin. Map this behavior-only branching field onto
the represented position at every replay transition.

Primary test: within-event association between branching entropy and the IMM
stationary-to-continuous transition probability is positive. Fragmented-source
or fragmented-destination transition mass is excluded from the primary metric.
Null: circularly shift transition probabilities within each event, preserving
event duration, mode mass, and switch count.

### H6: Neural ensemble turnover is the biological switch

Repeatedly split cells 70/30. Infer IMM switch probabilities from training-cell
replay spikes only. At the strongest eligible nonfragmented switch, compare the
held-out-cell population vector before versus after the boundary with matched
nonboundary transitions from the same event.

Primary outcome is excess held-out assembly turnover at training-defined switch
boundaries. Held-out replay spikes never update the latent posterior. A companion
test asks whether this excess predicts held-out IMM-minus-fragmented score.

### H7: SWRs mark commitment rather than generate trajectories

Compare source-deduplicated, LFP-validated off-SWR candidates with detected SWR
events using the same behavior-only pause and route metrics. Primary outcome is
emission-only future commitment. Secondary outcomes are event rank within pause
and alternative-route dispersion.

The primary comparison is quality- and session-matched. If fewer than ten
source-deduplicated off-SWR events have complete behavioral coverage, the result
is explicitly an underpowered diagnostic rather than a biological claim.

### H8: Replay has a within-event grammar

Use fixed-duration local evidence windows and a duration-penalized semi-Markov
decoder over stationary, diffusion, exact-sparse momentum, and fragmented modes.
The primary statistic is the fraction of events assigned more than one mode
with an ordered trajectory segment. Null: shuffle whole population time bins
within event and repeat the identical grammar inference.

Strong support requires excess multi-mode trajectory grammar over the shuffle
null in all four rats. Motif labels such as stationary-to-momentum-to-stationary
are descriptive until they replicate.

### H9: Learning or reward change increases switching

Use the frozen hc-11 matched PRE/POST ripple-event analysis because it provides
an independent learning manipulation. The primary outcome is the POST-minus-PRE
change in validated map-specific, order-sensitive, held-out-predictive dynamics.
The existing full-ladder result is authoritative; PF within-session time is only
a sensitivity analysis because explicit surprise labels are unavailable.

### H10: Event duration reflects computational composition

Primary test: `momentum_axis / n_time_bins` decreases with log event duration.
This prevents longer events from winning merely because log evidence accumulates
over more bins. Companion outcomes are nonfragmented switch rate per second and
composition gain per eligible bin, rather than unnormalized segment counts.
Null: permute duration within session after stratifying by spike-count quartile.

## Decision vocabulary

* `supported`: primary effect has the predeclared direction, rat-bootstrap CI
  excludes zero, multiplicity-adjusted q <= 0.05, and leave-one-rat-out direction
  is retained.
* `selective`: pooled primary effect passes but rat or leave-one-rat-out
  robustness fails.
* `contradicted`: robust effect is opposite the directional prediction.
* `inconclusive`: technically valid but uncertainty spans zero.
* `insufficient`: required data, cohort size, null, or technical gate is absent.

The campaign report must retain negative and insufficient results. It may not
promote a secondary metric after a primary test fails.
