# Denovellis surprise-gated retrospective replay test

This analysis asks whether awake W-track replay is recruited after a behaviorally
unexpected choice or reward outcome. It uses the published Denovellis et al.
80%-threshold SWR classification table and does not rescore neural data.

## Behavioral construction

For every run epoch represented in the replay table, center-to-outer-well choices
are reconstructed from the Frank-lab `linpos` `wellExitEnter` field. Arrival is
the beginning of the final visit within 10 cm of the destination well. Replay
opportunity is the following dwell, capped at 10 s. Rates are divided by this
exposure so long rewarded pauses do not automatically create a positive effect.

The primary surprise proxy is a discounted Beta-Bernoulli prediction of whether
the next outbound choice will alternate. It is updated chronologically and uses
only earlier choices. The current replay event never enters the predictor.
Results are reported for memory decays 1.0, 0.95, and 0.8.

Where reward-pump DIO channels can be identified from stable pulse durations and
mapped to destination wells, a secondary outcome-surprise analysis uses observed
reward delivery. The inferred alternation label is checked against these pulses;
this is a validation and sensitivity analysis, not a replacement for the primary
all-animal choice-surprise proxy.

## Neural endpoints

The published analysis restricts these SWR samples to animal speed <=4 cm/s;
the gate table independently audits that condition after event linking. The
analysis keeps rate and content separate:

- all-SWR rate per second of post-arrival dwell;
- trajectory-component SWR rate per second of dwell;
- continuous-state fraction;
- decoded total distance and displacement;
- velocity away from the animal's current location;
- signed velocity toward the center well after an outbound arrival;
- mean decoded distance from the animal.

Content endpoints are reduced to one median per behavioral trial. Extent and
velocity endpoints are evaluated only in events containing a continuous state.
Partial Spearman associations use session-by-outcome fixed intercepts and adjust
for trial progress, exposure, event duration, spike count, active tetrodes,
ripple strength, and animal speed as applicable. Inference uses animal-cluster
bootstrap intervals, within-session/outcome trial-label permutations, and
leave-one-animal-out sensitivity.

An explicitly exploratory, non-gating diagnostic asks whether replay after an
alternation error predicts correction on the next outbound choice. It adjusts
for the causal surprise estimate, trial progress, and replay-opportunity exposure.
Because this endpoint was added after inspecting the primary null result, it is
reported as hypothesis-generating rather than confirmatory.

## Claim boundary

This is a test of **surprise-gated retrospective replay** under an explicit
behavioral surprise proxy. It is not sufficient to identify Bayesian smoothing.
The published event summary does not contain a formal forward-filter versus
backward-smoother revision field, and its scalar path summaries cannot always
distinguish the just-traversed arm from the future arm. Positive rate and content
effects would motivate path-resolved re-decoding; null effects bound this version
of the hypothesis without proving that all smoothing-like replay is absent.

## Example

```bash
python scripts/analyze_denovellis_surprise_gated_replay.py \
  --dataset-root /mnt/lexar4tb/datasets/denovellis-frank-eden-2021/extracted \
  --replay-info /mnt/lexar4tb/datasets/denovellis-frank-eden-2021-legacy-download/replay_trajectory_paper/Processed-Data/replay_info_80.csv \
  --output-dir /mnt/seagate10tb/florianpfaff/denovellis-surprise-gated-replay
```
