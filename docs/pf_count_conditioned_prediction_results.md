# What predicts held-out cells after conditioning on spike totals?

## Result in brief

The PF predictive signal survives proper probability scoring and removal of
the population-total likelihood term. This is a useful positive control, not
yet a novel replay mechanism or independent biological replication.

The switching-specific claim is less robust: IMM exceeds fixed diffusion under
untempered inference, but fixed diffusion exceeds IMM under the historically
used sharpened inference. Both settings use proper, untempered held-out scores.
Do not select whichever setting makes a preferred model win.

## Frozen experiment

- 160 native ripple events within RUN epochs, eight sessions, four rats.
- Each session contributes its first 20 native RUN ripple events; no selection
  on IMM wins. This is a previously inspected, early-session convenience cohort.
- Five fixed 70/30 cell splits. RUN maps learned for each neuron. Replay latent
  inference sees training-cell spikes only; held-out spikes never update it.
- Three training observations: full Poisson, conditional cell identities, or
  population-total counts alone. Two spatial maps, two inference temperatures.
- Primary target: held-out cell identities given their observed total, scored
  with normalized multinomial probabilities. This is cross-cell prediction,
  not forecasting future time bins or computing joint trajectory evidence.
- Event medians across splits, then equal-animal means. No replay hyperparameter
  tuning. A static analytical baseline matches the repository stationary model.

## All five primary contrasts

Units are nats/event, with hierarchical 95% bootstrap intervals. These are
predictive score differences, not Bayes factors or trajectory speeds.

| Conditional held-out contrast | Primary inference T=1 | Positive rats | Sensitivity T=0.3 | Positive rats |
| --- | ---: | ---: | ---: | ---: |
| IMM minus independent positions | +7.723 [3.330, 12.140] | 4/4 | +5.950 [2.709, 9.147] | 4/4 |
| IMM minus static location | +8.599 [2.730, 15.621] | 4/4 | +6.096 [1.945, 11.552] | 4/4 |
| IMM minus fixed diffusion | +1.879 [0.513, 3.806] | 4/4 | -1.367 [-3.314, 0.402] | 0/4 |
| Real minus permuted map, IMM | +0.763 [0.207, 1.341] | 4/4 | +0.783 [0.271, 1.360] | 4/4 |
| Identity-only minus total-rate-only inference, IMM | +16.082 [7.353, 25.394] | 4/4 | +18.813 [8.106, 29.091] | 4/4 |

The T=0.3 IMM-versus-diffusion interval includes zero despite every animal's
point estimate being negative. It establishes sensitivity of ranking, not a
precise population-level proof that diffusion is superior.

With four rats the smallest exact one-sided animal sign-flip p is 0.0625.
Positive bootstrap intervals do not override that limitation. These are
exploratory pointwise intervals, not multiplicity-adjusted confirmation.

Full-Poisson versus identity-only inference gives nearly the same result at
T=1 on this target: identity-minus-full IMM +0.058 [-0.068, 0.189]. Thus the
prediction is not explained merely by retaining the total-rate term. The
total-rate-only baseline still knows the RUN population-total spatial map;
it is not a universal null for every nonspatial neural process.

Descriptive event-level normalization preserves the principal directions.
At T=1, equal-animal means of event-median score differences per held-out spike:
IMM versus independent +0.311, versus static +0.224, versus diffusion +0.057,
real versus wrong +0.024, identity versus total-rate inference +0.568.
Six of 800 event/split combinations have zero held-out spikes and contribute
zero conditional log score; per-spike normalization treats these as undefined.
All 160 events have defined normalized event medians from their remaining splits.

## What this changes

Supported in this PF cohort: training-population temporal inference predicts
the held-out population's cell-identity pattern beyond independent snapshots
and a static position, even without the population-total likelihood term.
Correct spatial adjacency adds a modest positive increment in all four rats.

Not supported as a robust model-identity claim: switching IMM is uniquely the
best dynamical explanation. Its comparison with fixed diffusion reverses with
the inference temperature. Conditioning does not equalize spike information,
remove all neuronal noise correlations, or identify a causal brain mechanism.
Only one shared population-code permutation per session was used; this is not
a cross-context encoding-map replication.

The older held-out analysis supplied temperature-0.3 emissions directly to the
predictive helper. Those are powered, unnormalized test likelihoods. Do not
reuse its absolute scores as proper predictive probabilities. The current
analysis always scores held-out probabilities at T=1; its numerical +7.723
must not be confused with an older, similarly sized headline statistic.
It also does not re-estimate the earlier mode-content/held-out correlation.

## Paper decision

This is concrete progress toward defensible predictive validation. It does not
yet fulfill the goal of discovering a high-importance biological mechanism.
The robust observation is conditional cross-cell temporal coordination; the
identity of the winning dynamic model remains calibration-dependent. Switching
replay models and rate-confound checks have substantial published precedent.

A next mechanistic discriminator, not yet implemented or validated, is whether
RUN neural-code similarity or physical spatial distance better predicts the
held-out population's dynamics. It requires matched flexibility, training-only
geometry, known-generator recovery, and external validation. It must not reuse
the present held-out outcomes to tune a preferred model. This is a research
direction, not a promised positive or a certified novelty claim.

## Verification and provenance

- All 48000 score rows and technical gates pass; runtime 2496.6 seconds on
  gpuserver6000 with 48 single-thread workers.
- Independent verifier checks 48 source MAT hashes, reconstructs 40000 paired
  split contrasts, 8000 event contrasts, and 50 aggregate contrasts.
- Actual training/test cell IDs are disjoint. Frozen posterior hashes remain
  unchanged. Analytical map-invariance error is 1.14e-13.
- 35 focused tests pass, including stationary/moving known-map simulations and
  corruption cases for held-out temperatures, split coverage, and aggregation.
- The verifier does not independently recompute raw posteriors or bootstrap CIs.
- Scoring commit: f63f1dada9209bf589fc56884005b72368d1c878.
- Frozen source CSV SHA256:
  505f8905d3e94cbbc6395c5ba3a35fd9e219368a1b58dca7e4624d936a0be9a8.
- Main artifact:
  /mnt/seagate10tb/florianpfaff/pf-count-conditioned-prediction-all160x5-20260908/.
- Independent audit:
  /mnt/seagate10tb/florianpfaff/pf-count-conditioned-prediction-all160x5-audit-v2-20260908/.

The full manifest records exact input files, parameters, and output hashes.
The scorer and frozen protocol were not changed during the real run.
