# Session-calibrated, region-protected content screening

Frozen before fitting, 2026-09-15. Development only. The active goal still
requires an unchanged method to succeed on independent recordings.

## Why this is a new experiment

The leave-rat-out learned screen reduced pooled disagreement but selected
observations with worse known-Home recovery in Rat4. Two limitations were
exposed: calibration pooled spatial losses, and calibration from other rats
did not describe the target session. Merely tightening the old LP cannot
improve its sub-unit training optimum. This experiment instead uses a local
RUN-only calibration, like fitting the encoding model itself, and separately
protects both true spatial classes. It does not retune the previous rule or
erase its failure.

Use the same four original matched populations, all 1,836 original candidate
endpoints and 513 accepted-segment endpoints. Counts, maps, original 20-ms
endpoints, flat spatial prior, likelihood and Home region remain unchanged.
No extra cells, retiming, trajectory priors or target replay labels.

## Frozen rule

For EACH session, fit using only that session's early native RUN Q3 (first
20-ms bin per 250-ms parent), cal_poisson_gain1 and cal_conditional. Reuse the
independently reconstructed calibration readouts from the previous experiment.
Other sessions, native Q4, test simulations and replay outcomes cannot affect
fitting. Inherit the previously frozen full-RUN cell eligibility; this is not
new blinded raw-recording validation. All evaluation banks have been examined
in development before, so a pass would still require external confirmation.

Observable features are exactly the previous ten paired-population features:
Home probabilities, normalized entropies, log1p spikes/active cells, normalized
position separation and regional TV. This uses BOTH populations. It is not an
A-only prediction of unseen B or a test independent of observed disagreement.

Tree settings remain depth8, <=128 leaves, >=64 rows per leaf, squared error,
seed20260915. Training weights equal the three sources and the two true classes
within each source. Targets are the previous nine costs, true_home, and each
side's normalized physical error/Home Brier multiplied by each class indicator.
Standardize all targets by weighted SD with floor0.01. These truth quantities
are TRAINING TARGETS ONLY; deployment uses only observable features.

One leaf retention probability is shared across the three local sources.
Maximize t in [0,1] subject to the previous per-source requirements: exact 50%
retention within each true class, 20%*t Home-gap reduction, 10%*t separation/TV
reduction, no entropy or class-balanced accuracy worsening. Additionally,
neither side's physical error nor Home Brier can worsen WITHIN EITHER TRUE
CLASS in ANY training source. Constant p=0.5,t=0 remains feasible.
The second LP holds t within1e-8 of optimal and minimizes weighted L1 distance
from p=0.5. Export both primal/dual certificates and the observable trees.

Freeze all four models before accessing evaluation readouts. Retain exactly
ceil(N/2) observations per session/source/map by predicted probability; ties
use the first8 little-endian SHA256 bytes of
`20260915|local-content-screen|SESSION|SOURCE|OBSERVATION_INDEX`.
Fractional training feasibility does not certify the realized hard subset.
Also report spike-count/entropy half-screen comparators, not alternative winners.
No settings or thresholds change after fitting or evaluation.

## Evaluation and stop rule

All previous numerical gates remain: complete fixed denominators; half retained;
candidate Home-gap reduction>=20% in early/full maps with no rat worsened;
accepted-segment gap not worsened; primary separation/TV>=10% reduction with
improvement in>=75% rats; neither pooled entropy increases; both populations'
class-balanced physical error/Brier do not worsen in any rat/control source;
both true classes retain>=20% in every session/control source.

Add separate non-worsening physical error and Home Brier gates for EACH
session, true class, side and native Q4/five simulated truth sources. Report
all classwise values, not just pooled error or the worst example. Do not
interpret selection agreement as accuracy or describe unknown replay truth
as measured. Training progress<1, fractional-vs-hard discrepancies and any
failed gates remain visible; no session is dropped to rescue a result.

Independent audit must verify the prior source audit/hash chain, exact source
rows, per-session training restriction, target construction, independent tree
refits, eight original-scale LP certificates, frozen timing, deterministic
selection, all numerical summaries and classwise gates. A technical pass is
not a scientific pass. Any failed scientific gate prevents external promotion.
