# Interpreting the Information Controls

This note explains the frozen counterfactual protocol; it does not change any
scoring input, threshold, generator, or result.

## What Is Removed

Let C be the full cell-count vector in a fine time bin, and S the retained
subset. The three primary observations are:

- Full: C.
- Pooled: (C_S, sum(C_removed)).
- Native subset: C_S.

These form deterministic data reductions. They require no true-position
information. The pooled intermediate retains every population spike but loses
the individual identities of removed cells; its last channel has the summed
rate map of those cells. Dropping that channel then loses both counts and
aggregate spatial information. It is not a pure spike-count-only intervention.

For independent Poisson fine-bin counts, sums of disjoint cells remain
independent Poisson variables. The pooled likelihood is therefore valid at
the fine-bin observation-family level. Conditional on the population total,
the pooled cell-identity probabilities are obtained by summing the removed
probabilities. Pooling does not change that total.

## Information Is Not Heuristic Performance

For any two candidate positions x and y, the KL divergence between their
full-count distributions cannot increase under this deterministic pooling.
The same holds for their conditional cell-identity distributions at fixed
population count. This is the usual data-processing property, not a claim
that every decoded event must become less continuous.

The MAP/posterior-mean estimators, grid, finite windows, and geometric
continuity threshold are not optimal decision rules for every target measured
here. Their finite-sample performance can improve or degrade after pooling.
In particular, the decoder treats a moving 20 ms window as a single position,
whereas simulated firing rates vary within that window. Losing cell identities
may change the effect of this approximation. An improved continuity fraction
does not imply an increase in available spatial information, a better
biological representation, or a universally preferable recording strategy.

## Why Oracle Restoration Is Secondary

The oracle-restored arm distributes removed spikes among retained cells using
their relative rates at the known simulated position. It preserves total
spikes and native retained spikes, but its new labels depend on latent truth.
It is not a deterministic reduction of observed counts and can introduce
information. It tests a mathematically specified alternative population, not
a correction usable for unknown replay paths or unrecorded neurons.

## What the Dose Series Can Establish

The same paths and populations are scored at nested Poisson exposures. This
asks how much additional spike information changes the performance of THIS
pipeline. Large multipliers do not assert physiological firing rates. A high-
exposure plateau can reflect spatial discretization or the within-window
approximation as well as insufficient population information; temporal/grid
sensitivities are needed to distinguish them.

Recovering an injected gradient with its known spatial coordinate is an
optimistic measurement check. It is not a usable real-data correction, and a
small positive response is not faithful recovery of the gradient magnitude.
Any eventual equivalence procedure needs separate calibration/evaluation and
must retain unavailable selected-event estimates in its denominator.
