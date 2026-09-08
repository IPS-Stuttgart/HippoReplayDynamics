# Frozen 2D cross-cell predictive replication

Question: does the conditional temporal-spatial predictive advantage replicate
in Tanni, an independent 2D sorted-spike recording, under the same pipeline as
PF? This is not a new replay mechanism or a retuning of the closed hc-11 result.

## Frozen Cohort and Encoding

All common-eligible detected-core high-MUA windows from the audited
replay-coverage-event-definitions-all33-20260905-v2 artifact: 4,001 PF from eight
sessions/four animals and 5,224 Tanni from 25 sessions/five animals. Include all
arena sizes and all weak/zero-spike windows. No selection on trajectory/model
scores, ripple overlap or decoder quality. No previously reported 226/1264
trajectory denominators are reused. Eligibility requires supported whole-window
immobility under the already frozen speed/duration rules.

This is a retrospective cohort, already inspected for coverage/continuity, not
a prospectively unseen confirmation sample. Candidate detection used the full
population, not training cells alone. Prediction is conditional on that fixed
selection, not an unconditional assay free of every ascertainment effect.

Reuse the hash-pinned common coverage RUN maps (8 cm grids, 1.5-bin smoothing,
speed >10 cm/s, occupancy >=0.05 s) and RUN-only unit QC. Existing maps are not
refitted to replay. Spatial support is the stored valid-bin mask. These maps
have prior held-out RUN decoder QC; that does not validate replay content or
establish calibrated posterior uncertainty. PF's original 160-event/4 ms/6 cm
predictive result is not silently relabeled as this common-pipeline experiment.

Pool successive four native cached 5 ms bins to nonoverlapping 20 ms bins,
retaining each final partial bin and all spikes. No bin-level support filtering
and no temporal smoothing of spike counts. Five deterministic 70/30 neuron
splits per session, with train and held-out cell identities recorded in caches.

## Models and Proper Prediction

Training-only cell identities conditional on their bin totals infer independent
positions, one static position, diffusion, and first-order IMM. Use existing
exact full-support engines. Physical diffusion sigma=60 cm/sqrt(s), stationary
sigma=2 cm, Gaussian support cutoff=3 sigma, IMM switch time constant=60 ms.
These parameters are inherited from the PF primary physical model, not chosen
to make Tanni win. Transition durations use center-to-center elapsed time,
rounded to one nanosecond for cache stability. No likelihood temperature.

Freeze every training posterior before evaluating held-out cell identities,
conditional on their observed totals with proper multinomial probabilities.
Sum marginal predictive log scores; do not call this joint model evidence or
future-time forecasting. Keep zero-held-out bins/events; per-spike ratios for
zero total are undefined, not zero or removed from raw-score endpoints.

Two global, position-free composition baselines: the normalized spatial mean
RUN rates, and a cross-event recalibration. For the latter, divide each session's
events into five chronological chunks; use the other four chunks excluding
calibration windows within one second of any tested window. Aggregate all
cell counts only from these other events, with 100 pseudospikes toward the
RUN global composition. Held-out neurons can supply calibration spikes in
other events, never their scored event. Record weak calibration and exclude
nothing based on it. Fail if a fold has no separated calibration events.

One deterministic shared population-code permutation per session changes
spatial adjacency while preserving the population snapshots. Apply to all
train and held-out rate maps together. Independent and static-location scores
must be invariant; a temporal change tests adjacency, not a cross-context map.
This is not a universal no-replay null or an independent biological replicate.

## Endpoints and Interpretation

Primary Tanni contrasts: IMM minus independent positions, static location,
cross-event global composition; and real minus permuted IMM. Report diffusion
versus independent and IMM versus diffusion rather than selecting a preferred
winner. Report RUN-global, global-recalibration and per-heldout-spike sensitivities.
Train-only nonstationary mass is exploratory metadata, not an inclusion gate.

Pair scores within split, median over five splits per event, mean per session,
equal-session mean per animal, equal-animal dataset mean. Primary uncertainty:
5,000 animal/session/event hierarchical bootstrap draws, seed 20260908. Maps,
cell partitions and calibration fits stay fixed, an important omitted uncertainty
source. Four/five animals limit population inference; pooled events are not
independent biological replications. Do not compare raw dataset magnitudes.

A bounded external result requires all four primary Tanni contrast means and
their lower pointwise intervals to be positive, with all five animal means
positive. If this fails, report which axes fail without selecting new animals,
settings or favorable trajectory subsets. This intersection criterion is not
a guarantee of a unique IMM mechanism; model identity and nonparametric
significance remain separately limited. Successful synthetic fixtures only
establish operating correctness, not power for arbitrary real alternatives.

Before interpretation: exact tiny-grid inference/prediction tests, zero-count
and leakage tests, raw-count and source-hash checks, independent dense-score
reconstruction and aggregate verification. Do not infer no replay from a failure
or claim that awake 2D and sleep 1D differ based on unmatched cohorts. Do not
revive the already unsupported biological speed-uniformity claim.
