# Cell-predictive selection of temporal context

Frozen before this experiment's real-data scoring. The preceding unconditional
temporal-context experiment failed abrupt-jump safeguards and remains a failure.
This experiment tests a new selection diagnostic, not new transition parameters.

## Question and exact scope

Can cells not used to infer a path tell us when temporal context is useful,
without consulting the other evaluation population or changing the endpoint?
Reuse all eight PF recordings/four rats, the same200 hash-selected candidates
per recording, three frozen disjoint equal-size population splits, first-half
RUN encoding, fourth-quarter RUN, and all four known-path source generators
from edge-support-content-20260914. Split0 primary;1/2 sensitivity. No reselection,
new timestamps, new maps, cross-population conditioning, or parameter sweep.

## Primary rule

Inside each population A or B separately, freeze three balanced cell folds from
cell IDs and a session/split/side-specific deterministic seed20260914. Folds are
unchanged across real, RUN and simulated source events. Every cell is validation
once. At least six cells per population are required; smaller inputs fail.

For each fold:

1. Infer independent terminal20ms and diffusion+25% reset endpoint posteriors
   using ONLY the other two folds' cells. The temporal model and up-to200ms
   disjoint-bin context are exactly those of the failed preceding experiment.
2. Score the held-out fold's final20ms population count vector under each
   training-cell posterior, integrating its JOINT Poisson likelihood over the
   latent endpoint. Never re-infer either latent posterior with validation
   spikes. Include silence. Do not multiply separately marginalized cell scores.
3. Difference=context predictive log score minus independent predictive log
   score, divided by number of validation cells in this fold.

Use temporal context for that population if its median difference across all
three folds is positive beyond a1e-10 numerical zero guard AND the full population's final20ms contains
at least3 spikes from at least2 active cells. Otherwise use its original
independent endpoint posterior. No event is dropped; A and B may choose different
methods. This is a predictive-selection heuristic, not a calibrated p-value.
After the choice, use the chosen model's all-own-population posterior. Model
selection therefore uses all own cells through cross-validation, but never B
to choose for A or A to choose for B. Claims of prediction pertain to the
fold-held-out scores, not the final refitted posterior's in-sample likelihood.

Methods: independent; unconditional_context (failure/control);
predictive_context (PRIMARY); entropy_matched (independent endpoint likelihood
tempered to the primary's entropy, separately for A/B). No post-hoc method
replacement. Positive diagnostic scores do not establish true replay location.

## Measurements and gates

Store every fold membership, raw predictive score, normalized fold difference,
choice, support count, endpoint posterior, source event ID and original clock.
Report the same location/regional errors, entropy, A/B separation, regional TV,
per-session/per-rat means and p90 truth errors as the temporal benchmark.

Retain the previous advancement gates unchanged: >=10% improvement in BOTH
real agreement measures, >=3/4 rats improved, descriptive rat-bootstrap interval
above zero, no entropy inflation, >=5% advantage over concentration-matched
independent control, no worsening of either side's mean/p90 physical error or
regional Brier in ANY known-truth source, and complete entropy matching.
No sudden-jump gate is removed or weakened. All outputs independently audited.

Additionally, on both held-out RUN and the late-jump source, the fold-predictive
diagnostic must rank known error improvement with AUROC>=0.60 (equal-rat mean)
and AUROC>0.5 in at least3/4 rats, separately for A and B. Labels here are
independent-error minus unconditional-context-error >1e-8cm (numerical zero); ranking predictor is
median fold predictive difference. Compute only on the predeclared3-spike/
2-active-cell eligible endpoints; report excluded counts and choice coverage.
Undefined AUROCs or missing animals fail, not pass. These labels are evaluation
only and never train or select the rule. Report all other source AUROCs too.
Fold scores are correlated; they are not three independent replications.

An independent auditor must reconstruct all predictive scores, choices and
posteriors using direct dense filtering and SciPy Poisson probabilities, verify
the original native-clock count intervals and all input hashes, and recompute
summary statistics. Tiny tests must cover exact latent-state integration,
validation-spike exclusion, A/B independence, fallback and missing-data gates.

## Independent validation and final objective

PF is development: prior failures informed this design. Only if all development
gates pass, apply the unchanged rule to the eight-session/four-rat hc11 benchmark.
hc11 is independent recordings, not a dataset never inspected before. No refit
or threshold changes from hc11 outcomes. Even external passage is not enough:
the original targeted Home-coverage contrast must shrink at the same timestamps
without degrading known regional RUN recovery. Until then the remedy objective
is incomplete. Success on fewer controls is not a substitute.

Run on gpuserver6000 using a detached, logged systemd user service, preserve all
old artifacts, and commit code/protocol before scoring. Export compact reports,
not raw recordings. A failed method is recorded, not promoted by relabeling it.
