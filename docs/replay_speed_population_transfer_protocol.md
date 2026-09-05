# Frozen Retrospective Held-Out-Population Calibration Test

This specification precedes the new leave-one-animal-out fits and outputs.
Earlier within-session panel outcomes have already been inspected. This is
retrospective external-population transfer within the available datasets, not
prospective validation on newly collected animals or a biological replay test.

## Question and Scope

Does the existing scalar inverse speed-gradient calibration transfer to a
recording population excluded from fitting and residual calibration? Compare
against the existing within-session calibration and raw-bootstrap baselines
on exactly the same test panels. No spike simulation, decoding, event selection,
threshold tuning or new biological scoring is performed here.

Source: audited `replay-speed-identifiability-all33-20260905`, eight PF sessions
from four animals and 25 Tanni sessions from five animals. All frozen panel
files and schedules are required, including missing decoded statistics. The
original reconstruction audit must match the source manifest, and hashes of
every consumed source table must match that manifest. Do not use a subset
chosen for successful decoding.

## Exclusion and Fitting

- Leave one animal out within each dataset. All its sessions and all its
  fit/calibration/test panels are excluded from the new inverse fit and
  calibration radius. Do not pool PF and Tanni for fitting.
- For each of the eight existing estimator/support/selection readouts, fit
  the unchanged inverse OLS baseline using only the other animals' `fit`
  panels. Use only A-map/Poisson panels, with the existing minimum of 20
  finite statistics and variance rule. Missing training statistics remain
  explicitly counted; missing gradients are an input error.
- Compute the unchanged residual order-statistic radius on the other animals'
  `calibration` panels. Missing statistics contribute infinite residuals.
  Neither test outcomes nor B-map/shared-gain observations enter either step.
- Training/calibration rows are pooled with the existing baseline's equal
  panel weight. This gives sessions equal scheduled weight, not animals;
  animals with more sessions and sessions with more finite fits can influence
  fitting more. Record animal/session membership and finite counts. This is
  an explicit standard pooled-transfer baseline, not an optimized population-
  adaptive calibration method or an animal-exchangeable conformal method.
- Apply inverse Gaussian and pooled-conformal intervals to every test panel
  of the held-out animal. Invalid test statistics or failed fits produce
  unbounded intervals, not dropped observations. Do not clip intervals to the
  generating range. No new animal-specific inverse slope/radius is learned.
- The held-out animal still supplies its own RUN-fitted encoding maps for
  decoding, as in the source experiment. Exclusion concerns learning the
  speed-statistic-to-truth calibration, not an implausible transfer of cell
  identities or rate maps across animals.

The same source within-session inverse Gaussian/conformal and raw-bootstrap
summaries provide the comparison. Those use local fit/calibration panels, so
they are a less demanding reference, not an independently held-out-population
method. New and reference summaries must have identical test denominators.

## Frozen Readouts and Decisions

Primary: posterior mean, >=2 cells/3 spikes per bin, selected continuous run.
The all-data readout is a diagnostic, not a replacement chosen after failure.
Retain MAP, unfiltered bins and all source fixed-gradient strata as sensitivities.

Truth is `v(x)=1000*(1+g*q(x))` cm/s with horizontal q in [-1,1], not actual
physical wall distance. The source fit/calibration/test panel schedule remains
40/99/100 uniform-g draws plus 20 each at g=-0.5,-0.25,0,+0.25,+0.5. Each panel
uses up to 30 fixed source duration profiles. Evaluate all A/B-map and
Poisson/shared-gain test conditions, without refitting to the stress conditions.

Report empirical interval coverage, finite availability, finite-only coverage,
finite width, nonzero decisions, strict +/-0.25 equivalence and false
equivalence, with +/-0.10/0.50 sensitivity. Unbounded intervals count as covered
but must be shown as abstention, never as successful identifiability.

Summarize panels within sessions, sessions within animals, and animals equally.
Report the paired transfer-minus-local contrasts on the same test cohorts.
Animal bootstrap intervals use 5,000 resamples, seed 20260916; only four/five
animals support each dataset. Leave-one-out fits share source animals, so
bootstrap intervals are descriptive conditional summaries, not an independent
new-animal sampling guarantee and not uncertainty from refitting the procedure.

Pooled panels are clustered by animal/session and held-out populations can
shift the data distribution. The pooled conformal radius therefore has no
claimed finite-sample 95% coverage guarantee for an unseen animal. Evaluate it
empirically. This distinction is part of the experiment, not a hidden excuse
for failed transfer.

## Outputs and Verification

- `speed_population_transfer_fits.csv`: exclusions, fitting/calibration counts,
  membership hashes, source animals/sessions, fitted coefficients and radii.
- `speed_population_transfer_decisions.csv.gz`: every held-out test panel and
  both methods, including failed/unbounded outputs.
- `speed_population_transfer_session_summary.csv`,
  `speed_population_transfer_animal_summary.csv`,
  `speed_population_transfer_summary.csv`,
  `speed_population_transfer_paired_comparison.csv`.
- `speed_population_transfer_gate_summary.csv`, manifest, reconstruction audit,
  non-rescoring report and figures.

Technical gates require source completeness/hash linkage, no excluded-animal
or test/B/gain leakage, complete expected rows and reference denominator
matches. Failure to achieve useful calibration is a result, not a technical
failure or a reason to change settings.

Tests must cover exclusion even when held-out fit/calibration values change;
test-value changes cannot affect fitted parameters; B/gain contamination,
duplicate/missing panels and changed source hashes fail; missing calibration
statistics preserve abstention; interval endpoints/decisions match the frozen
scalar implementation; animal weighting and reference pairing are checked.

The auditor must independently reconstruct pooled regressions and residual
ranks, all interval endpoints/decisions and group denominators from frozen
panel inputs. This is not a fresh audit of raw spikes or every simulation;
those are covered by the explicitly linked earlier reconstruction artifact.

## Decision Boundary

This closes the held-out-population evaluation of these declared calibration
baselines, whether transfer succeeds or fails. It does not establish a universal
decoder correction, prove biological uniformity, or show that no other
population-adaptive method could work. No parameter is selected to make a
positive result. Afterward, integrate the observed limits into the manuscript
and perform the full claim-by-evidence completion audit.
