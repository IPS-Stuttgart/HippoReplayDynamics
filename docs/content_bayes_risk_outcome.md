# Posterior accuracy-cost audit: a constraint, not a remedy

2026-09-15. The independent-data remedy/diagnostic objective remains unmet.

## What this turn established

The original gain-1 Poisson calibration generator uses exactly the early-RUN
rates and 20 ms likelihood of the baseline decoder. Every rate is positive.
All 16,000 position/count vectors across the original four PF pairs were
reconstructed from the frozen generator. The baseline Home posterior is thus
Bayes-optimal under the model and uniform spatial prior, not merely a plausible
decoder. The bank's 50/50 true-class sampling requires explicit importance
weights to recover the uniform-bin prior (Home occupies 3.20-3.53% of bins).

For a Home indicator Y, available spikes N, true posterior q=E[Y|N], and any
fixed adjustment a using the SAME observations:

    expected_Brier(a) - expected_Brier(q) = E[(a-q)^2].

This standard conditional-expectation identity supplies an accuracy constraint.
Both true-class Brier risks cannot be nonworse in expectation while changing q.
Because positive-rate Poisson mixtures have full support on finite count
vectors, exact expected no-harm would require q unchanged at every count vector.
Finite empirical safeguards are weaker: rare harmful changes can go unobserved.
This is not a new theorem, nor an impossibility result for better observations,
better encoding, cell exchanges or biological replay.

## Quantified demonstration on real rate maps, simulated observations

| Original pair | Temperature-2 reduction in RMS Home disagreement | Uniform half-mixture reduction |
| --- | ---: | ---: |
| Rat1/Open1 | 33.64% | 50.00% |
| Rat1/Open2 | 27.40% | 50.00% |
| Rat2/Open1 | 40.82% | 50.00% |
| Rat4/Open2 | 31.32% | 50.00% |

Despite apparently large agreement improvements, ALL 16 nontrivial
population/adjustment combinations had positive model-expected Brier costs.
Observed uniform-prior-weighted Brier also worsened in all 16 combinations.
Every adjustment worsened observed true-Home Brier (16/16), while improving
non-Home Brier (16/16). Agreement was purchased by a harmful regional trade-off.

Rao-Blackwellized expected-cost estimates range 0.000176-0.001265 for temperature
2 and 0.000191-0.002554 for the half-mixture. These are Brier units, NOT physical
position error or percentage points of replay prevalence. Conditional costs are
exact for each count vector; integration over the count distribution remains
Monte Carlo on previously inspected calibration data.

The paired L2 bound was verified for all 12 original pair/adjustment rows:
D_new >= max(0, D-sqrt(C_high)-sqrt(C_low)). A reduction r in RMS disagreement
requires summed expected Brier cost at least r^2*D^2/2, under the same-information,
correct-model assumptions. This is NOT a bound on the original replay mean Home
gap, class-conditional J, full regional TV or trajectory classifications.

## Positive control and limits

An enumerated two-neuron/four-location Poisson example verified both sides of
the information distinction. Mixing halfway toward the prior halves disagreement
but adds 0.002438 Brier per population. Giving both populations the full pair of
observations removes disagreement and improves Brier by 0.013393 per population.
The second control changes the information and is deliberately NOT proposed as
an equal-observation remedy. Count truncation was bounded below 1e-12 probability.

The earlier whole-cell exchange changes information and therefore is NOT ruled
out by this argument. Neither its finite-search failures nor posterior-only
failures prove that no valid remedy exists. The actual replay latent state is
unknown, so model-correct optimality cannot be asserted for biological replay.

No accuracy threshold was relaxed. No event, population or dataset was dropped.
No method was fitted/selected from these results. No native Q3/Q4, held-out test
bank, replay or independent recording was decoded in this turn. Existing source
hashes were checked without interpreting those other arrays as new validation.

The next useful branch must earn a gain through demonstrably better encoding
or genuinely better information, or produce a diagnostic that predicts risk
out of sample. More agreement from posterior flattening is not sufficient.
That independent predictive/reduction result has not been achieved.

## Audit and artifacts

- Server: gpuserver6000; detached systemd services, single-threaded BLAS.
- Producer commit: 6aa4c3d4b17157376293abdc7eb93e9ca42f37b2.
- Verifier commit: c85345e92d742f4678d1d0238ea6acd5972a47f1.
- Root: /mnt/seagate10tb/florianpfaff/content-bayes-risk-20260915.
- Producer service started 04:14:09 UTC; journal reports completion at 04:14:15.
- Verifier service completed 04:19:59 UTC, MainPID=0, ExecMainStatus=0.
- Manifest SHA256:
  d7f90654ba196d23b83a2594b978c3f2c9742bbbc16fe92962cf77fc2eff168e.
- Independent verifier used full scipy Poisson likelihoods without importing
  the producer, reconstructing 96,000 saved probabilities, 24 accuracy rows and
  12 paired bounds. Maximum probability discrepancy: 2.332e-15.
- 66 focused tests passed; Ruff and staged whitespace checks passed. Tests cover
  prior correction, source corruption, nonzero finite-sample cross terms,
  rare-count costs, information augmentation, model misspecification, and
  tampered outputs even when their manifest hashes are updated.
- No unrelated user-owned regional-bound files were changed.

Primary outputs: measurement/accuracy_costs.csv,
measurement/agreement_cost_bounds.csv, measurement/exact_toy.csv,
measurement/report.md, audit/reconstruction.csv and audit/independent_audit.json.
The protocol records the proof, assumptions and literature references.
