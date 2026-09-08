# Learned Nonspatial Assembly Comparator

Frozen 2026-09-08 before real fits or comparator scores. Exploratory comparison
on reused events, not an independent replication or a new detector proposal.

## Question and Prior Work

Do behavior-derived spatial maps add cross-cell predictive information beyond
a fully learned model of co-active cells and their transitions? The earlier
three/eight-component mixture with imposed persistence was weaker than a
global composition baseline and was not an adequate skeptical temporal control.

[Maboudi et al. (2018)](https://elifesciences.org/articles/34467) already learns
Poisson HMMs from population burst events, including PF open-field data, and
tests held-out events and whole-bin time shuffles. No novelty is claimed for
learning assemblies, their transitions, or replay without place fields.
Here we use a count-conditioned multinomial counterpart for a matched proper
held-out-cell identity score. It is not an exact implementation of that paper.

## Frozen Data and Separation

Use the existing 9,225 MUA candidates / 33 sessions / nine animals, native
20 ms nonoverlapping count caches, five fixed neural 70/30 partitions and five
chronological event folds with a one-second guard. Every event is evaluated;
no continuity or previous evidence filtering. Parent manifest:
`d16542d0bf935929a05a335f84cd2060167c76dd8e255378beb202fb84e97386`.
Spatial rate-adapted results manifest:
`84418e7db33df939b9609dae312eed19d72b7dc6e8d382d7cbdc91e4c1744943`.

Learn all HMM parameters only from other calibration events. Reset initial
state distribution at each event; never concatenate transitions across events.
All cells can train the calibration-event model, but target held-out cells
cannot affect calibration, initialization, model choice or target inference.
Infer target states from target training cells only. Then score held-out cell
identities, conditioned on each held-out bin's total, by mixing likelihoods
over the frozen training posterior. Sum bin-marginal log scores, not joint
evidence, and do not power the held-out likelihood. Event detection itself
used all cells; this is conditional on frozen candidate selection.

## Learned Comparator

Use hmmlearn 0.3.3's EM and compiled forward-backward kernels. Primary 50 states;
20 and 100 states are prespecified capacity sensitivities, not scored-event
hyperparameter selection. Two seeded initializations, select by calibration
regularized objective only. Learn initial probabilities, all transition rows
including self-transitions, and cell proportions per state. Initial transition
is 0.5 identity plus 0.5 uniform, not fixed during training. Emission starts are
random informative calibration bins. No position, RUN map, geometry or replay
model label enters this fitter.

Use a calibration-only global composition with 100 uniform population
pseudospikes. Each state has ten pseudospikes toward that composition;
initial distribution and each transition row have one total uniform
pseudotransition (Dirichlet parameters 1+1/K). Maximum 500 EM iterations per
restart, objective tolerance 1e-5 times calibration bins. Check the regularized
objective, not raw likelihood, for monotonicity. Save converged status; hitting
the budget does not count as convergence or a passed adequacy check.
Do not select an inferior converged restart instead of the highest-objective fit.

States transition once per consecutive count bin, including the terminal
partial bin. Count conditioning removes total-rate/exposure from emission
likelihoods, not from latent dynamics. This differs from the spatial model's
physical center-time transitions and remains an explicit model difference.

Compare learned HMM, same learned emissions with independent states weighted
by calibration posterior occupancy, and calibration global composition.
Reuse unchanged primary alpha100 real-map spatial IMM/IID/global scores from
the audited parent. The global RUN-informed shrinkage baseline and the new
uniform-shrinkage global baseline are distinct; report both.
Use the parent's K20 whole-population time-bin permutations to compare learned
HMM order sensitivity; this is supportive, not uniquely spatial evidence.

## Adequacy and Outcomes

Before real interpretation, recover known synthetic assembly structure,
verify forward-backward against exhaustive state-path enumeration and test
that perturbing target held-out spikes leaves the training posterior unchanged.
Require complete finite scores, input hashes, exact fold separation, all neural
partitions, convergence reporting and independent predictive reconstruction.

Primary paired spatial IMM minus learned K50 HMM. Separately report HMM minus
global, HMM minus independent assignments, order advantage, capacity and
calibration size. A spatial win against an HMM that cannot beat global is not
proof against a strong nonspatial alternative. If the assembly model wins, this
does not prove activity is nonspatial: learned assemblies can inherit spatial
structure without using coordinates. A tie does not prove equivalence.

Differences within each split, median across five splits per event, equal-event
means within sessions, equal-session within animals, equal-animal within dataset.
Use existing 5,000-draw animal/session/event bootstrap for full experiments;
per-held-out-spike is sensitivity. Pilot checks use no biological gates.
Keep PF and Tanni separate. Do not infer actual replay prevalence, speed,
a novel neural mechanism or completed high-importance discovery from this test.

First runtime/integration pilot: lexicographically first session per dataset,
all its events and all five folds, three state counts. This deliberately retains
the small first Tanni session rather than selecting a decoder-favorable session.
Run all sessions only after synthetic and pilot reconstruction checks pass.
