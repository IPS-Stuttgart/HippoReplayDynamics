# Error-trained temporal-context choice

Frozen before fitting or evaluating this rule. Prior unconditional and
cell-predictive context failures remain failures. PF is development, not a
previously untouched confirmation dataset.

## Target and split

Predict whether the already frozen context posterior improves known position
error and regional Brier loss over independent last-20-ms decoding. This is
supervised loss prediction, not a new latent dynamics model or a tuning sweep.
Reuse the eight-session PF benchmark, six sources, 200 real candidates per
session, fixed clocks, cell partitions and first-half RUN encoders. The complete
predictive-context measurement/audit supplies the independently verified
independent and unconditional-context posteriors; its failed selection rule is
not reused. Split 0 is primary; splits 1/2 are sensitivity only.

For each held-out rat, train only on split-0 known-location rows from the other
three rats, including their fourth-quarter RUN and all four simulation sources.
Never train on real replay, any row/split/session of the held-out rat, or an
agreement measure. Hold out the entire rat, not random endpoint rows. Save all
four leave-one-rat-out models and a separate all-PF model for future external
validation before looking at their real candidate outcomes. The all-PF model
must never evaluate PF. Equal weights apply to rats, sessions within rat,
sources and population sides, then events within each group.

## Frozen predictor

For A and B independently, use only its own spike and active-cell counts,
population size, context duration, independent/context entropy and posterior
width, distance between its two endpoint means, and independent/context
regional concentration. Width and distance are normalized by the valid grid's
diagonal. No absolute location, rat/session identity, opposite-population
feature, replay acceptance or known truth enters prediction.

Two GradientBoostingRegressor models predict independent-minus-context loss:
physical error divided by arena diagonal, and regional Brier score. Parameters:
squared_error loss, 100 trees, depth 2, learning_rate 0.05, min_samples_leaf 40,
subsample 1, random_state 20260914; all other sklearn defaults recorded. No
early stopping, hyperparameter search, probability threshold fitting or replay
labels. Persist every tree, coefficients, library version and training-row IDs.

Choose context only if BOTH predicted gains exceed the 1e-10 numerical guard
and the population's original endpoint has >=3 spikes and >=2 active cells.
Otherwise retain independent decoding. A and B choose separately. Keep every
event and original endpoint. Do not average A/B posteriors or move endpoints.

Compare independent, unconditional_context (prior failure/control),
error_trained_context (PRIMARY), and independent likelihood tempered to match
the primary's entropy. No post-hoc primary substitution.

## Falsification and independent validation

Keep the preceding real-agreement and truth gates: >=10% reductions in both
real separation and regional TV; >=3/4 rats improved and descriptive rat-bootstrap
lower bound >0; no increased entropy; >=5% advantage over entropy matching;
neither side's mean/p90 physical error nor regional Brier may worsen in RUN or
ANY of the four simulations, including late jumps. All endpoint clocks, selected
methods, features and metrics must be independently reconstructed.

The predicted physical-error gain must also rank the actual gain with AUROC
>=0.60 in equal-rat mean and >0.5 in >=3/4 rats, separately for A/B, for both
held-out RUN and late jumps. Only the frozen 3-spike/2-cell supported endpoints
enter this diagnostic metric; report all denominators. Undefined values fail.
Known truth is evaluation only in the held-out rat. Report secondary splits
without turning them into independent replications or replacing split 0.

If PF passes all gates, apply only the frozen all-PF model to the existing
eight-session/four-rat HC-11 benchmark without refit or threshold changes.
Preparation of HC-11 decoder banks does not authorize new selection fitting.
Full independent-data validation and reduction of the original targeted
Home-content contrast, at its original times and without worse regional RUN
recovery, are still necessary before claiming the user's objective achieved.

Run on gpuserver6000 in detached logged services. Commit protocol/code before
fitting. Retain failures and exact input hashes. No raw recordings in exports.
