# Kleinman ripple recruitment and subsequent RUN coding: feasibility protocol

Frozen before the opportunity audit, 2026-09-24. This is a distinct biological
lead, not a rescue or a threshold change for the stopped replay-extent assay.

## Candidate question and novelty boundary

Does VTA suppression alter the association between a cell's post-run ripple
recruitment and subsequent expression of its spatial firing pattern, beyond
changing how frequently ripple events occur?

The narrow candidate contribution is a manipulation-dependent *coupling*, not
the general proposition that dopamine, replay and map plasticity are related.
McNamara et al. already linked dopaminergic stimulation, reactivation and spatial
memory persistence (https://doi.org/10.1038/nn.3843). Bakermans et al. already
analyzed replay-associated remote field formation and strengthening
(https://doi.org/10.1038/s41593-025-01908-3). Roux et al. tested stabilization
through awake ripple disruption (https://pmc.ncbi.nlm.nih.gov/articles/PMC5446786/).
These are close prior work, not discoveries of this project.

The source Kleinman/Foster paper reports replay recruitment/localization and
marginal field-reliability summaries, not the split-temporal cell-level coupling
proposed here (https://elifesciences.org/articles/99678). This targeted reading
does not certify novelty or exclude overlapping literature.

## Population and chronological unit of observation

Inventory all 135 sorted-spike sessions. Preserve the existing 127-session RUN
pass designation; do not use the 14-session extent subset for this different
endpoint. Record metadata/source failures rather than silently repairing them.
Units use (tetrode, cluster) identifiers within a session, never across sessions.

For each direction within each uninterrupted reward epoch, enumerate every
consecutive same-direction RUN pair. The earlier traversal is baseline, the
later traversal is future target. Three earlier same-direction traversals,
strictly before baseline, provide reference-map data. No baseline, intervening
ripple or future spikes determine reference maps or unit inclusion. This needs
five consecutive same-direction traversals, but all earlier pairs remain in
the denominator with explicit insufficient-history status.

The candidate exposure interval is baseline exit to future RUN entry, including
both intervening end visits. It is not a claim that a single arrival caused the
outcome. Other running and the opposite end's consumption occur in that gap.
No opportunity may straddle reward epochs. Repeated observations of a cell or
overlapping reference windows are dependent and cannot count as independent
animals.

## Exposure definition independent of the tested unit

Primary events are the released LFP ripple intervals, not spike-density events
selected with the same spikes whose recruitment will be tested. These remain
ripple detections, not confirmed continuous replay.

Use only the first 10 seconds of each native reward-end visit and tracking
intervals with valid timing and speed <=8 cm/s at both endpoints. Intersect
native ripple intervals with this eligible exposure time. Merge overlapping
intervals before counting time or spikes. Baseline background is the same
eligible time outside native ripples. Count half-open intervals; include
zero-ripple opportunities. Record clipped intervals explicitly. No ripple
power or sharp-wave validation is inferred from interval tables alone.

## Readout feasibility, not a frozen biological estimator yet

Reuse the established 2-cm/4-cm-smoothed directional RUN map helper and its
training-only unit inclusion (>=10 reference RUN spikes and >=1-Hz peak).
The three reference traversals supply movement >8 cm/s; baseline and target
count summaries use valid direction-matched movement >20 cm/s, at least 20 cm
inside the reward-end thresholds. The audit reports counts, occupancy support,
training-selected units, ripple/background duration and recruited-unit counts.

Do not inspect future spatial scores, recruitment/outcome correlations or drug
contrasts in this audit. Low counts or incomplete support should be visible
before choosing an interpretable outcome. A future spatial-predictive endpoint
would condition on a cell's spike count and use only the prior reference map,
with baseline coding quality, local background rate, occupancy, movement,
elapsed time and session time controlled. Silent future cells need explicit
accounting rather than being called stable or removed without a denominator.

Before biological inference, calibrate the chosen endpoint for unchanged maps,
global and cell-specific gain, field shifts, count differences and temporal
autocorrelation. Include a time-direction/control-lag comparison. Do not
interpret an ordinary recruitment-stability correlation as causal plasticity.

## Manipulation limitations and stop rules

Novel-track drug assignments are track-linked; session order is not unrestricted.
Experimental/control groups and within-animal CNO/saline contrasts are required,
but only three spike-recorded animals per group are available. Large cell/event
counts cannot replace those six subjects. Novelty, reward epoch and drug remain
distinct factors; inspect no favorable subgroup as confirmatory.

If there are too few supported repeated observations across animal/condition
cells, stop this endpoint or label a limited feasibility analysis. The present
audit has no fitted effect, p-value, biological pass gate or paper-ready claim.
Do not promote it merely because its code and accounting tests pass.

## Deliverables

- session inventory with original RUN-pass status and exclusion reasons;
- chronological opportunity table, including insufficient-history and no-ripple rows;
- descriptive animal/drug/novelty coverage;
- protocol/input/output hashes, exact commit and durable server job record.

The next decision is whether these inputs can support an independently calibrated,
prospective coding outcome. This protocol does not commit to finding a positive
drug effect and does not reopen the prior unfiltered reward-extent test.
