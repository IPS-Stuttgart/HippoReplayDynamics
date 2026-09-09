# Occupancy-Matched Forecasts: Verified Results

2026-09-09. Exploratory mechanistic control; not a high-importance discovery.

## Decision

Matching the null's stationary state frequencies does not remove the PF
forecast contrasts. All three real-map/neural models beat their own matched
null at the fixed 40-ms horizon in every PF animal. Tanni is heterogeneous:
spatial IMM retains a positive pooled interval but one negative animal;
neural HMM now has an interval crossing zero; diffusion remains mixed.
No model passes the predeclared full two-dataset criterion. No animal, horizon,
model setting, or event subset was changed to obtain a positive conclusion.
The previous lagged experiment's failed compound verdict remains unchanged.

The result strengthens a limited PF statement: preferences for destinations
improve prediction beyond the specified equilibrium and dwell constraints.
It does not identify a directional sequence-generating mechanism, nor establish
that every MUA candidate is replay. Tanni does not confirm the stronger uniform
cross-dataset claim. Failure of that criterion does not prove absent dynamics.

## Frozen Inputs And Coverage

- Clean scoring commit: `7e7f6a74f2d47349e42bd8733efe2644d929baa7`.
- Run: `/mnt/seagate10tb/florianpfaff/occupancy-matched-forecasts-all9225-v2-20260909`.
- Manifest SHA256:
  `da514a563fc1ac1b9e3704cdf6b8a95b14b88049e14eebaf84901739fd07d980`.
- Source lagged-run manifest SHA256:
  `342d5670f77706ad784af7b1f98368f8eaa66e88247cfe78bb188d2daa79195a`.
- All 9,225 immobile MUA candidates, 33 recordings, nine animals; PF has
  4,001 candidates/eight recordings/four rats; Tanni has 5,224/25/five.
- 691,875 event/split/model/horizon rows. All 642,100 eligible original
  forecasts reconstructed with exact saved hashes; remaining rows retain
  explicit insufficient-full-bin status.
- Five frozen event folds and five frozen neural partitions; source RUN maps,
  cell-rate gains and K50 neural models reused, not refitted.
- 20-ms complete bins. Fixed primary center lag 40 ms leaves a full 20-ms
  unobserved interval. Only training-cell history through the origin enters
  the filter; held-out and intervening spikes never update the forecast.
- Primary temporal eligibility: PF 3,883; Tanni 5,164. Per-spike support:
  PF 3,878; Tanni 5,153. Empty targets score zero and have undefined per-spike
  ratios. Short-event exclusions and discarded partial bins are unchanged.
- Runtime: 149.40 seconds, eight CPU workers on gpuserver6000.

## Primary Results

Animal-balanced nats per held-out target spike. Brackets are conditional
95% animal-bootstrap intervals, not a correction for the wider hypothesis
search. Events use medians across neural splits before session and animal means.

| Model | PF: dynamic minus matched null | Tanni: dynamic minus matched null |
| --- | --- | --- |
| Spatial IMM | +0.042120 [0.038431, 0.045018], 4/4 positive | +0.012105 [0.001062, 0.024066], 4/5 |
| Spatial diffusion | +0.100574 [0.091863, 0.107500], 4/4 | -0.008744 [-0.058890, 0.038818], 2/5 |
| Learned neural HMM | +0.086178 [0.071920, 0.100436], 4/4 | +0.011155 [-0.001853, 0.023326], 4/5 |

These contrasts have different model-specific baselines. Larger increments
must not be read as a ranking of the models' absolute predictive performance.

The noteworthy change is Tanni neural HMM: its old dynamic-minus-dwell result
was +0.015724 [0.003346, 0.026064]; the new lower bound crosses zero.
The directly paired OLD-null minus NEW-null contrast is -0.005478
[-0.007403, -0.003707], negative in all five animals: the occupancy-matched
null predicts better than the old dwell null across Tanni animals. This does
not erase all evidence for dynamics, but demonstrates baseline sensitivity.
Subtraction of separately aggregated median contrasts need not equal the
aggregate of a directly paired difference.

Tanni R2478 remains negative for the learned-HMM increment (-0.011645/spike);
R2482 is only +0.001554. Spatial IMM's negative animal is R2474 (-0.006334).
Diffusion is negative in R2470, R2474 and R2478. No single common exclusion
would explain all these results, and no animal was excluded.

## Spatial And Horizon Sensitivities

Real-minus-permuted difference of dynamic-minus-matched-null contrasts:

- PF IMM: +0.041064 [0.037106, 0.044287], positive in 4/4 rats.
- Tanni IMM: +0.011536 [0.001124, 0.022580], positive in 4/5 animals.
- PF diffusion: +0.083074 [0.073878, 0.089921], positive in 4/4 rats.
- Tanni diffusion: -0.017412 [-0.063796, 0.026194], positive in 1/5 animals.

The full 20/40/80-ms tables are retained. Tanni IMM is positive in all five
animals at 20 ms but only four at the primary 40 ms and two at 80 ms.
Tanni neural HMM is positive in four animals at 80 ms (previous old dwell
comparison: five), with a small pooled increment +0.004859 [0.000254, 0.009816].
PF diffusion is negative at 80 ms in all four rats. These sensitivities cannot
replace the fixed primary endpoint; their eligible event populations differ.

## What Was Preserved And What Remains Unresolved

The new maximum-entropy transition null preserves the full ORIGINAL MODEL
stationary state distribution and state self-transition probabilities.
Spatial models also preserve every source/destination mode probability and
probability of staying at the same position across mode transitions. The null
does not preserve arbitrary pairwise flux, spatial distances, or recurrence
structure. Both forecasts start from the same original forward-filtered state.
This compares propagation rules, not separately optimized alternative filters.

It therefore controls more than a uniform reset, but cannot distinguish
directional ordering from reversible associations or local diffusion. A
positive forecast contrast is not evidence for behavioral planning, Bayesian
smoothing, or a unique IMM mechanism. Learned states can encode space even
when the model has no explicit spatial coordinates.

Prediction of neural sequences and the distinction between assembly
reactivation and ordered sequences are established subjects. See
[Predictive sequence learning in the hippocampal formation, 2024](https://www.sciencedirect.com/science/article/pii/S0896627324003714)
and the experimental dissociation in
[Associative and predictive hippocampal codes support memory-guided behaviors](https://pmc.ncbi.nlm.nih.gov/articles/PMC10894649/).
The positive PF control alone does not supply a new high-importance claim.

## Verification And Failed-Run History

The original run at `occupancy-matched-forecasts-all9225-20260909` stopped on
an original-forecast hash mismatch. It remains unchanged. No null results
from that failed run were interpreted. The correction restored the original
likelihood/permutation operation order and explicit array copies; likelihood
rounding discrepancies were about 1.42e-14. Exact hash checks were not relaxed.
Four new regression cases cover real/permuted IMM and diffusion.

Independent audit:
`/mnt/seagate10tb/florianpfaff/occupancy-matched-forecasts-all9225-v2-20260909-audit/occupancy_matched_audit.json`.

- Passing clean-tree audit at `e460272a262ba152607ced07fc75466a33c242ca`.
- All source/output hashes, all original rows, all 231 null parameter sets
  and their stationary/dwell/mode constraints checked.
- Separate likelihood/filter/transition implementation reconstructed 11,725
  new scores from the first chronological event of each of 165 folds,
  all its neural partitions and eligible horizons/models.
- Maximum score discrepancy 4.97e-14 nats; maximum equilibrium discrepancy
  9.04e-14. All 1,660,500 split contrasts and 332,100 event contrasts,
  session/animal reductions, conditional bootstrap intervals and decisions
  independently reconstructed.
- Native raw files were not reopened and source model fits were not repeated.
- 71 model/forecast/related tests and three non-rescoring reporter tests pass.
- Report figures visually checked; all reported table and figure hashes frozen.

## Research Decision

This closes the occupancy-preservation check without changing a threshold.
Retain it as a bounded positive PF forecasting result and a Tanni sensitivity
result. Do not retrofit fuzzy continuity selection or favorable horizons to
claim replication. The high-importance-paper goal remains unmet. A genuinely
distinct mechanistic claim still needs a discriminator beyond local/reversible
associations and independent confirmation, not another label for successful
HMM prediction. No further experiment is specified or claimed here.
