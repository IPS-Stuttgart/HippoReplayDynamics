# Post-diagnostic zero-mass control

2026-09-16. Added AFTER inspecting the frozen diagnostic, before this control's
results. This is not a predeclared confirmatory method or a change to the frozen
primary analysis. Original outputs/source remain unchanged.

Motivation: the 5-ms arm has about 61% silent windows. Continuous KDE prevalence
error (~20-22 pp) is worse than ternary (~7 pp). A continuous density estimator
smooths a large point mass at neutral BF=0, potentially producing a density-fit
artifact in addition to actual spike loss.

Control: retain the continuous-neutral score (silent windows set to zero), but
model its exact-zero mass separately in each class using a Beta(.5,.5) estimate.
Fit the same frozen arcsinh Gaussian KDE only to nonzero calibration scores.
Evaluate class likelihoods relative to the mixed reference measure (a point
mass at zero plus continuous density elsewhere). No validation labels enter
the density fit. No bandwidth changes. Apply to full populations at both 20 ms
and 5 ms using existing archives, the same validation panels and pooled/oracle
conditions. Report all earlier variants beside it, not just the best method.

This control can diagnose KDE misspecification at a point mass, but does not
validate real endpoint prevalence, generator knowledge, nuisance robustness,
or external transfer. The five-point/90% criterion remains unchanged.
