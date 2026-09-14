# Local RUN calibration predicts error but does not stabilize replay content

## Decision

Neither tested diagnostic passes the frozen PF development advancement gate.
No independent validation or original targeted Home-content remedy is claimed.
Do not advance this fitted method to Alme or relabel improved accuracy as reduced
population-content instability. Prior failed experiments remain unchanged.

## Method and denominators

Producer commit 8a032d08, independent verifier commit 130f1585. Computation ran
on gpuserver6000 in detached services; source hashes were verified unchanged.

All eight PF sessions/four rats; 4,069 fixed candidate endpoints, with 2,036
selected by the predeclared lowest-risk 50% per session (ceil); 7,446 held-out
RUN windows, 3,724 selected. Two matched-map and two drift/gain simulation draws
each retain the original per-population event spike totals and the same 4,069
endpoint identities. Primary split0; splits1/2 sensitivity, not extra animals.
This development cohort is not the original four-session targeted Home-match
cohort. The original +8.58pp Home contrast has not been reduced by this run.

First-half RUN encoding, third-quarter RUN-only risk training, last-quarter RUN
testing. Each side has its own Ridge error predictor. The policy averages those
two predicted log errors. Its inputs include each population's own counts,
entropy, width, location and encoding-map descriptors, never observed A/B replay
agreement, separation or regional TV. Original decoder posteriors and endpoint
times remain untouched. All coefficients were frozen before application.

## Full model results

Equal-weight animals after averaging sessions and draws; values are means, not
pooled event medians. Error is the mean of the two populations' physical errors.

| Outcome | Unselected reference | Selected half | Change |
|---|---:|---:|---:|
| Real endpoint separation (cm) | 45.571 | 45.176 | -0.395 cm, -0.87% |
| Real regional TV | 0.51421 | 0.51705 | +0.00284, worse |
| Held-out RUN truth error (cm) | 61.427 | 48.669 | -20.77% |
| Matched-map simulation truth error (cm) | 45.703 | 33.643 | -26.39% |
| Drift/gain simulation truth error (cm) | 49.163 | 38.058 | -22.59% |
| RUN truth error, original true-tile mixture (cm) | 61.427 | 50.646 | -17.55% |
| Matched simulation error, original true-tile mixture (cm) | 45.703 | 34.079 | -25.43% |
| Drift simulation error, original true-tile mixture (cm) | 49.163 | 38.905 | -20.86% |

Known-error risk correlations are positive in 4/4 rats for native held-out RUN
and both simulations. Both sides become sharper, not more diffuse (real
normalized entropy A 0.7532 -> 0.6802; B 0.7503 -> 0.6761). Known-error benefits
survive the coarse 3x3 true-location reweighting with all original tiles retained.
This is useful error prediction, not a successful stability diagnostic.

Real separation reductions by rat: Rat1 +14.86 cm, Rat2 +1.25 cm,
Rat3 -5.03 cm, Rat4 -9.50 cm (positive means better). Regional TV improves only
in Rat1: reductions +0.12694, -0.01793, -0.04178, -0.07861 respectively.
The descriptive rat-bootstrap reduction CI crosses zero for separation
[-7.26,+9.89] cm and TV [-0.0634,+0.0848]. Do not restrict to Rat1 post hoc.

The simpler spikes/active-units/entropy model also fails: real separation rises
from 45.57 to 49.35 cm and regional TV from 0.5142 to 0.5325, despite improving
known-position accuracy. In held-out RUN itself, full-model selection reduces
physical error but increases A/B separation (38.66 -> 50.16 cm) and TV
(0.4336 -> 0.5515). Hence agreement and accuracy cannot be interchanged.

## Verification and limitations

A separate implementation reconstructed all 144 fitted states via dense
Poisson decoding and closed-form ridge equations, checked 250,119 predictions,
288 selection cases, and all primary aggregate means. Old readout columns were
unchanged; all selections, count denominators and true-tile reweighting checked.

The simulations condition on observed totals, not actual active-cell patterns or
noise correlations. RUN truth is known but replay truth is not. Four rats and
prior inspection make this development, not confirmation. No new biology,
no claim that agreement measures are useless, and no claim of a solved remedy.

## Next distinct mechanism

Repeated single-bin quality selection is not enough. A worthwhile next mechanism
is using preceding observations while retaining the exact endpoint timestamp,
with a switching/jump-capable filter and known-path abrupt-jump controls. This
has NOT been implemented or tested in this run. Any such trial must demonstrate
both better truth recovery and reduced instability without merely enforcing
continuity; it still needs independent data and transfer to the original
targeted Home-content contrast before the persistent goal can be completed.

Results: /mnt/seagate10tb/florianpfaff/local-run-content-risk-20260914/
Subdirectories: frozen/, pf-development/, audit/.
