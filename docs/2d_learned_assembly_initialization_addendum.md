# Limited-Calibration Initialization Addendum

2026-09-08, after integration-pilot verification and a terminal failure of the
all-session run, before reviewing its aggregate comparisons.

One Tanni session, R2481 / 2019-04-29_16-26-33, has only five selected events.
Its five calibration folds contain four events and 27-33 informative count
bins. The code required selecting distinct informative bins for initial
emissions, making the declared K50/K100 initializations impossible. This is
an initialization restriction, not mathematical nonexistence of the regularized
HMM. The original run is retained as failed, including all completed outputs.

For this low-support case ONLY, sample informative calibration bins with
replacement when initializing the already-declared number of states. Preserve
the seed, all priors, model capacities, EM iterations, objective, scoring,
event/cell folds and candidate cohort. No data are excluded; the primary remains
K50. Record the replacement flag and effective occupied-state count. This does
not make four calibration events adequate for learning 50 distinct assemblies.

Reuse the 485 completed fold fits and scores with verified file hashes and
explicit producer provenance; compute the ten previously missing K50/K100 fits.
The non-replacement path is unchanged. This is a disclosed implementation
correction, not an independent confirmation or improved biological selection.
Convergence and finite predictions do not establish comparator adequacy: retain
global/independent-state comparisons and low-calibration warnings in reports.
