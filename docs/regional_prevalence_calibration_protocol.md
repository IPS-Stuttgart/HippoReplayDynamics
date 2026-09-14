# Regional prevalence calibration: fixed matched-population test

Frozen before calibration/target readouts, 2026-09-14. This tests the original
PF Home-content contrast, not an easier endpoint agreement surrogate. It is a
development screen; success also requires an unchanged independent-data test.

## Estimator and assumptions

For each original population separately, s is the independent uniform-prior
Poisson decoder's posterior mass within 20 cm of inferred Home. Estimate
mu0 = E[s | outside Home], mu1 = E[s | inside Home] on calibration observations.
Adjusted prevalence = (mean(s_target) - mu0)/(mu1 - mu0). This is probabilistic
adjusted classify-and-count (PACC), not a new estimation principle. See the
primary experimental study https://doi.org/10.1007/s10618-024-01014-1.

Identification requires transfer of class-conditional score distributions.
Changes in firing gain, within-region location distribution, correlated activity
or maps can violate this even when population averages appear plausible. No
coupling, pooling or agreement target is fitted between high/low populations.
No individual posterior, endpoint or event classification is modified.

Retain unbounded estimates. No clipping, common prior imposed on targets, silent
denominator regularization or dropping difficult target observations. An estimate
is unavailable when either calibration class has fewer than 100 observations,
or its response gap is below 0.05 (inverse amplification exceeds 20). Native
calibration must additionally contain at least 10 original 250-ms RUN blocks in
each class. The number of blocks, not the number of 20-ms bins, measures temporal
replication. Out-of-[0,1] estimates are flagged incompatible, not silently fixed.

## Frozen inputs

Original pf-matched-population-content-20260913 confirmed targeted pairs only:
Rat1/Open1, Rat1/Open2, Rat2/Open1, Rat4/Open2. Preserve all four, selected cell
IDs (which may overlap), inferred Home, grid, endpoint indices and both existing
candidate/accepted-segment cohorts. Verify every original early/full-RUN Home
mass from raw cached endpoint counts before interpreting a correction.

Primary correction uses early-RUN maps; full-RUN encoding is a descriptive
sensitivity and cannot be described as held-out RUN validation. Original unit
eligibility and spatial support were full-RUN-derived; original matching also
used third-quarter RUN and confirmation used fourth-quarter RUN. This is a
conditional test on that pre-existing selected cohort, not a pristine study.

Native calibration uses the 12 non-overlapping 20-ms windows at each frozen
third-quarter 250-ms RUN window start. Native validation uses the equivalent
fourth-quarter windows. The last 10 ms of each parent is unused. Use raw source
spikes and interpolated known midpoint position. No spike-support selection.

## Known-truth controls

All simulated observations are drawn once for the entire original cell universe
and then subset; shared cells receive identical spikes. The scoring maps remain
fixed. Calibration and test draws have separate deterministic seeds. Each bank
has 2,000 uniformly sampled supported states per class (4,000 observations).

Calibration banks: native third-quarter RUN (primary), matched Poisson at gain1,
and conditional-multinomial counts with total spike counts sampled from original
candidate endpoints. The latter assumes spike total independent of true region;
it is a sensitivity, not a claim about biological counts.

Test banks: native fourth-quarter RUN, independent matched Poisson gain1,
Poisson gain4, conditional endpoint-total counts, conditional counts generated
from second-half RUN maps, and a nonspatial shared-assembly stress. In the last,
80% of spikes come from a fixed near-Home assembly profile independently of true
position; remaining 20% follow the real location's early map. All such conditions
are declared, not selected for improvement. Synthetic uniform-within-class
sampling does not reproduce native occupancy or biological replay truth.

Known-prevalence panels use 0.05, 0.15, 0.30, 0.50, 0.75 by weighting separately
observed class means; these panels share observations and are NOT independent
replicates. Native natural-prevalence estimates are also reported. This evaluates
finite-bank systematic recovery; it is not a finite-event variance guarantee.
No synthetic true labels enter real target fitting or population selection.

## Screen and interpretation

Technical requirements: all four sessions complete, hashes unchanged, frozen
baseline exactly reconstructed (1e-9 tolerance), dense independent readout and
calibration reconstruction, count/time/cell alignment verified and all failure
rows retained.

For moving onward, primary native calibration must be available for both
populations in all four sessions. On native held-out known-location panels,
equal-rat mean absolute prevalence error must improve by at least 20% over
uncalibrated mean posterior mass, be <=0.05, and not worsen in any retained rat.
Original candidate high/low absolute discrepancy must decrease by >=20% without
incompatible estimates. Accepted-endpoint sensitivity cannot worsen. Wrong-map,
gain and shared-assembly recovery are explicit robustness limits, not evidence
that apparent agreement identifies true replay content.

Even passing this screen does not achieve the user goal: freeze identical
calibration settings and test on independent recordings before calling this a
validated remedy. A fail does not justify threshold tuning or overwriting this
run. Never interpret global agreement or a correct simulated prevalence alone
as known biological replay content.
