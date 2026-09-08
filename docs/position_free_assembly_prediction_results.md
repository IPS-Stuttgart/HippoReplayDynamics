# Spatial versus position-free prediction: frozen diagnostic

## Provenance

- Producer: `565de51c31b063912beb7acbc9f9a1fa17cf520b`.
- Audit: `4aebf73f1a7a5705db26dacd07bb341a826e97ed`.
- Reporter: `893ddd2a`.
- Run: `/mnt/seagate10tb/florianpfaff/position-free-assembly-prediction-pf-hc11-20260908`.
- Manifest SHA256: `6aa537e46de992c4e9e7a9ef438d201874d2b5e75e1b02fc21df5da4b0befd8a`.
- Audit/report directories: same parent with suffixes
  `position-free-assembly-prediction-pf-hc11-audit-20260908` and
  `position-free-assembly-prediction-pf-hc11-report-20260908`.

## Outcome

PF retains a bounded predictive advantage against the tested assembly models.
However, the assemblies themselves lose to their global composition baseline.
This control does not establish a new spatial mechanism or exclude all
nonspatial explanations. Do not increase capacity or tune persistence based
on these scored events to obtain a desired answer.

Equal-animal mean paired score differences, median across five cell splits
per event and equal-session means within animal. Intervals resample only four
animals, keeping all maps, calibration fits and within-animal data fixed.

| Cohort | Contrast | Mean nats [95% CI] | Positive rats |
|---|---|---:|---:|
| PF, 160 RUN ripple events | spatial IMM - global composition | +11.715 [4.049, 19.382] | 4/4 |
| PF | spatial IMM - persistent assembly K=3 | +19.313 [7.927, 30.699] | 4/4 |
| PF | spatial IMM - persistent assembly K=8 | +18.372 [7.570, 29.175] | 4/4 |
| PF | persistent K=3 - global composition | -7.573 [-11.124, -4.022] | 0/4 |
| PF | persistent K=8 - global composition | -6.822 [-10.012, -3.633] | 0/4 |
| hc-11 POST, 160 events, direction mixture | spatial IMM - global composition | -0.836 [-1.105, -0.567] | 0/4 |
| hc-11 POST | spatial IMM - persistent K=3 | +1.246 [-0.496, 4.468] | 1/4 |
| hc-11 POST | persistent K=3 - global composition | -2.186 [-5.187, -0.432] | 0/4 |
| hc-11 PRE, 160 events, direction mixture | spatial IMM - global composition | -1.676 [-3.378, -0.676] | 0/4 |

These are proper held-out cell-identity log scores conditional on the bin's
held-out spike total, summed over posterior marginals, not joint log evidence.
Positive values mean better prediction relative to the stated comparator.
Differences of aggregated rows need not add because split medians are nonlinear.
Do not compare raw magnitudes between datasets: PF uses 4 ms native bins and
RUN-period ripples, hc-11 uses 20 ms PRE/POST bins, and cell populations differ.

## Comparator Adequacy

The global comparator estimates average cell proportions from separate
candidate events. Assemblies add three/eight co-firing components; only
training cells infer the assignment within scored events. Component fitting
uses no spatial coordinates or firing-rate maps, but can inherit spatial
correlations from spikes and is not a guaranteed no-replay null.

Calibration uses at most ten short candidate events per fold. PF's RUN spatial
maps have different and generally much greater training exposure. The known-
assembly operating check succeeds, but does not establish adequate real-data
learning at this calibration budget. The global baseline being better than
the fitted assemblies is therefore a limitation to foreground, not conceal.
The original spatial maps/priors and event subsets were never retuned.

## Verification

All 480 event count matrices were independently rebuilt from raw timestamps;
all 4,800 population partitions were checked. All 96 cross-event calibration
fits were checked for guarded separation, pooled count conservation, objective
and convergence, with deterministic refits using the shared fitter. A separate
dense solver reproduced all 19,200 predictive scores, maximum absolute error
6.09e-12. All 152,000 split contrasts, 30,400 event medians and 190 summary/CI
panels were rebuilt. Parent spatial score tables were hash-checked and joined,
not independently spatially rescored by this audit. 46 focused tests passed.

## Claim Boundary

Retain PF's proper predictive result; external IMM replication remains
unsupported in the frozen hc-11 cohort. This does not establish no replay in
hc-11, a PF-versus-hc-11 biological difference, or a uniquely spatial mechanism.
The comparator is not Maboudi et al.'s fully learned HMM and should not be
presented as a defeat of that published method. Co-firing versus sequence
structure and cross-validated HMM fit already have direct precedent:
[Maboudi et al., 2018](https://elifesciences.org/articles/34467).

This is a completed diagnostic, not satisfaction of the high-importance-paper
goal. No new dataset scaling, prior sweep or favorable-event subset follows
from this result.
