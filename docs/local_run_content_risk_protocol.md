# Recording-local, two-population RUN-risk diagnostic

Frozen before this diagnostic's PF outcome evaluation. PF is development data,
not an independent confirmation. Prior failed global PF-to-Tanni predictors and
all their gates remain unchanged. No success here completes the external goal.

The previous predictor trained on PF to predict B error from A features and
extrapolated across arena sizes. This experiment instead calibrates each
population's own error within its recording, using only third-quarter RUN
20-ms windows and first-half RUN encoding. A and B each get a separate model.
It is a diagnostic requiring both populations' observations, not a forecast of
unobserved cells. It must never use their decoded agreement, separation,
regional TV, replay truth, or candidate acceptance as an input or training label.

Reuse all eight PF sessions and all three previously frozen disjoint splits.
Fit Ridge(alpha=10) to log1p(own known RUN position error / arena diagonal).
Fit/scaling/imputation use only that recording/population's calibration RUN.
Models: constant mean; spikes+active-cell count+entropy; full additionally uses
relative posterior width, relative x/y, relative encoding coverage and local
encoding-code gradient at that population's own posterior mean. No temporal
inference, posterior averaging between populations, or decoder changes.

Freeze all coefficients and input hashes before applying them to real endpoint
readouts, last-quarter RUN, matched-map known-location simulations, and
drift/per-cell-gain known-location simulations. The diagnostic risk is the mean
of the two predicted log errors. Select its lowest 50% (ceil) separately in each
session/source/split/draw, deterministically breaking ties with event index.
The actual replay endpoint and its posterior remain unchanged. Split0 is primary;
splits1/2 are sensitivity, never substitute for a failed primary result.

For each nonconstant model report risk prediction of known A/B mean error and
of real A/B separation/TV. The retention reference is expected equal-count random
retention, namely the full-case mean. Report all outcomes and denominators:
- regional TV and endpoint separation, independently, by session and rat;
- each side's entropy (avoid apparent improvement through diffuse posteriors);
- each side and mean known-location error, in each generator and native RUN;
- true-location 3x3 tile-reweighted error, so selective location retention cannot
  masquerade as improved recovery; zero selected support in any originally
  occupied true tile makes this diagnostic incomplete, not zero error.

PF advancement (not validation) requires: >=10% reduction of both real TV and
separation, positive reductions in >=3/4 rats with descriptive rat-bootstrap
lower CI >0; no increase in either side's real entropy; positive known-error
predictor correlation in >=3/4 rats; no increased mean known error or either
side's error in RUN and both simulation families; no increase after true-tile
reweighting, with all original tiles retained. At least 100 calibration windows
and nonzero selected/test events are required. Choose no post-hoc retention or
threshold. Do not scale or claim a remedy if these fail.

If a method passes, freeze it unchanged for an independent cohort. Candidate:
Alme familiar-room recordings (existing data, seven animals, four sessions each),
whose raw timestamp/units were inspected but no replay outcomes have been
computed here. Independent validation still requires source/encoding QC, known
truth and real fixed-endpoint stability, all before returning to the original
targeted matched-population content contrast. No independent success is inferred
from the PF development gate or from readiness metadata alone.
