# hc-11 biological lead: experience-related change in conditional sequence order

Status: prospective assay calibration, not a new biological result. 2026-09-23.

## Biological question and competing explanations

Does PRE-to-POST experience change which neural population state follows which,
beyond changes in firing patterns, state occupancy, and state persistence?
This is distinct from asking whether POST has more bursts, more decoded replay,
or more temporal predictability. All three can increase without changed order.

The broad question is not new. Grosmark and Buzsaki (2016,
https://doi.org/10.1126/science.aad1935) studied rigid/plastic participation in the
same hc-11 design. Farooq et al. (2019,
https://doi.org/10.1016/j.neuron.2019.05.040) reported increased within-assembly
coordination with largely conserved sequential order. A prospective contribution
would require a held-out test that discriminates those explanations, not just a
new algorithm or a significant PRE/POST average. Mahoney et al. (2016,
https://doi.org/10.1371/journal.pone.0147708) already studied short-range temporal
interactions supporting sleep sequences. No novelty claim is established here.

## First experiment: simulation-only falsification

Six latent states and 24 neurons; 20 independent nonoverlapping 20-ms bins/event.
States and trajectories are NOT given to the decoder. Supplied emission maps
are deliberately an easier, oracle calibration, not learned maps from real data.
Each phase has 80 calibration and 40 independent evaluation events. No evaluation
events enter parameter fitting. A fixed disjoint partition uses 18 inference
and six evaluation neurons, with each state represented in both populations.

Fit phase-specific transitions using hmmlearn's multinomial HMM with emissions
fixed, at most 100 iterations. Record actual tolerance convergence, not merely
the iteration-cap flag. Compare own-phase transitions with other-phase
transitions transported to the target phase's equilibrium and state-wise self
transition probabilities. Iterative proportional fitting preserves off-diagonal
cross-product ratios while matching those nuisance quantities. Each transition
model runs its own causal filter on inference neurons. It predicts evaluation
neuron identity 40 ms ahead, across one unobserved 20-ms bin, conditioned on the
evaluation population's future spike count. This tests identity, not burst rate.
The target count is used only for evaluating the multinomial score, not inferring
the latent state. Initial distribution is the same target equilibrium for both.

Per event: (own-phase predictive score - transported-other score)/evaluation
spikes. Average equally over events, then PRE and POST, within simulation seed.
This bidirectional statistic tests phase-specific predictive routing, not a
directional increase in POST organization. It does not identify a learned route,
memory benefit, or a causal effect of experience rather than time/sleep drift.

Frozen generative cases:
- unchanged everything;
- doubled POST spike rate with identical sequence and normalized emission map;
- changed occupancy/dwell with identical off-diagonal routing odds;
- changed state-dependent neuron recruitment with identical transitions;
- reversed transition preference with identical emissions, occupancy and dwell.

Run two emission modes on the SAME generated observations:
- oracle_phase_map: exact phase-specific maps, no true trajectory;
- pooled_map: equally averaged maps, intentionally mismatched when recruitment
  changes. This resembles a naive shared-codebook proposal and must not be
  confused with an estimated-map robustness result.

64 seeds per case, starting 20260924. Even seeds in unchanged case set a 95th
percentile score threshold; odd seeds evaluate each case independently of that
calibration. Null rejection <=15% and changed-order recovery >=80% are only a
coarse engineering screen (32 evaluation seeds/case), not biological alpha or
proof of adequate power. All fits must converge, all rows must be present, and
occupancy/dwell constraints must be met. Do not tune this screen after seeing
results. More seeds can measure error rates better, not rescue a failed method.

If recruitment changes produce apparent order changes, do NOT score real data
with the shared-codebook assay. A different observation model would need a
separate frozen calibration including estimated maps, unmodelled states,
state-dependent gain, missing neurons, and finite-sample nuisance estimation.
Even passing the oracle screen is insufficient to authorize real-data scoring.

## Real-data feasibility, not outcomes

Check all eight hc-11 sessions/four animals, native PRE/MAZE/POST epoch order,
NREM availability in both sleep phases, and same within-recording CA1 unit IDs.
Presence of IDs is NOT proof of electrode/unit stability; inspect waveform or
drift metadata before a biological claim. Existing PRE/POST scores are known, so
this is exploratory, not a preregistered confirmatory replication.

If the assay becomes validated: independently detected NREM events (reserved
neurons), phase-blind paired candidate selection, equal calibration budgets,
duration/count/sleep-time matching, guarded chronological folds, and within-NREM
early/late controls. Report all sessions, equal-session within-animal and
equal-animal effects, and leave-one-animal-out robustness. Four animals are a
limited replication unit. Exposure is not randomized; phase change alone cannot
establish learning causality. A positive result would motivate independent
experience-content validation; a negative result is not proof of fixed order.

No new real replay scoring, IMM claim, or revision of closed dataset conclusions
is authorized by this simulation protocol.
