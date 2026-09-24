# Kleinman native candidate model adequacy

Frozen 2026-09-24 before inspecting fitted native content or reward contrasts.
This is the last model-adequacy check for the current restricted extent endpoint,
not another choice of decoder or a reward-effect analysis.

## Cohort and observations
Inventory every native spike-density event (SDE) in all 127 RUN-pass sessions.
Preserve the frozen 14 extent-eligible session flag; fit only those 14. Do not
remove the four with weak epoch-transfer support to improve the result. Keep all
source event row IDs, counts and exclusion reasons. Native SDEs are candidates,
not confirmed replay. No new ripple/trajectory selection or reward-label filtering.

Count composite tetrode/cluster identities from the same full-RUN maps as the
calibration. Start at the native onset, take complete disjoint 10-ms bins, and
record the omitted final remainder (<10 ms) and spike count. Do not move the
window to a favorable subevent. Fit durations 40 ms through 2 s (at least four
bins), with spikes in both alternating-bin halves, and a valid timestamp interval.
Everything else remains inventoried with a reason. These are numerical fitting
limits, not evidence of adequate information. Native durations, total/raw and
encoding-cell counts, active cells and truncation remain visible.

A descriptive bank-range flag records 100-400 ms fitted duration, 48-96 observed
encoding spikes and >=5 active encoding cells. The synthetic bank used EXPECTED
48/96 spikes, not these observed-count cuts: this flag is not a calibrated
eligibility rule or a new selection tier. No unannounced reclassification.

## Fitting and checks
Reuse the exact 306-before-support-exclusion static/linear/cosine library,
nine start/end fractions, both directional maps, >=95% map support, 1-ms rate
integration and conditional multinomial likelihood. No true trajectory is given,
no monotonic direction/start near the animal is imposed, and no HMM is added.

1. Report existing fitted extent and alternating-10-ms-bin cross-fitted
   moving-minus-static score. Keep the restricted monotone-path assumption explicit.
2. Fit the best template from ALL families on training bins only and score the
   other bins. Compare with a non-spatial per-cell composition estimated from
   the same training bins using a symmetric Dirichlet(0.5) pseudocount. Swap halves
   and sum. This is temporal-bin cross-fitting, not held-out cells or causal
   future prediction. No test spikes choose a latent path or baseline probabilities.
3. Conditional deviance = 2*(saturated binwise cell-composition log likelihood -
   maximized template log likelihood). Generate 99 parametric replicas from the
   best template, preserving every 10-ms bin's total count, and REFIT each.
   Diagnostic upper-tail probability = (1 + count(replica deviance >= observed))
   /100. A first-index tied MLE supplies replica probabilities deterministically.
   This is a composite-null plug-in diagnostic, not an exact hypothesis-test p,
   not a posterior probability of replay, and not proof of adequacy if >0.05.

Seed 20260924, stable SHA256 session identity plus native event ID. No extra
seeds, threshold tuning or selecting the highest-scoring map. Flag deviance tail
<=0.05 descriptively. Keep gain conditioning limitations: cell-specific changes,
spike dependence, unmodeled paths and within-bin gain changes can all cause misfit.

## Calibration and power limitation
Before native fits, audit 40 controls per frozen session, half static/half moving,
alternating 48/96 total spikes over 200 ms. Randomly choose from the allowed
templates and allocate counts uniformly across bins. For each, generate a matched
composition control and a paired cell-gain-mismatch control: multiply rates of
a randomly chosen half of cells by four and renormalize, holding the path and
bin totals fixed. Refit and run the identical diagnostic. Total 560 matched and
560 mismatched events across 14 sessions. This supplements, not replaces, the
earlier recovery bank and never changes its eligibility.

Report false flags and sensitivity by animal/session and static/moving generator.
A negative-control rate above 10% in any animal blocks interpreting <=0.05 as
a suitably calibrated alarm; retain real fits only as exploratory diagnostics.
Low mismatch sensitivity prevents claiming that non-rejection demonstrates
correct cell tuning. The selected mismatch is only one stress case, not every
biologically relevant alternative.

## Decision boundary
No reward, drug, direction-incidence or across-epoch replay-content contrast.
No new biological effect or confirmation cohort is declared. The six-animal
extent-coverage gate still fails. Readouts will determine whether this endpoint
can be used even exploratorily or should stop. Do not re-tune to obtain an effect.
A future reward contrast would require map-sensitivity analyses and an explicitly
limited design, not promotion of these 14 imbalanced sessions.

Technical gates require all native IDs accounted, all 14 sessions attempted,
control counts complete, no unexpected scoring failures, finite fit scores and
all exclusions explicit. Scores do not select events for a biological result.
Run on gpuserver4090 in tmux, immutable manifests, independent reconstruction
checks, compact paper archive without raw spikes or large per-bin data.
