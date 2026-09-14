# Independent Population Content: Frozen Pilot Protocol

Directions 2 and 3 follow the matched-population intervention. This is a
method-development experiment, not a test guaranteed to find multiplexed replay.

## Sources and Scope

Use the eight frozen PF session caches and existing all-MUA event windows. Keep
all fixed candidate endpoints; previously full-cell accepted endpoints are a
secondary cohort. The catalog and accepted endpoints used pooled cells, so A-only
prediction is conditional on that catalog, not independent event detection.
Original unit eligibility and spatial support used full RUN. New population
selection and decoding maps use first-half RUN. Report that conditional boundary.

## RUN-Only Disjoint Splits

Match nearby place-field peaks and normalized rate-map profiles into cell pairs;
assign one cell of each pair to A and one to B. Each candidate is disjoint and
equal sized. An odd last cell is unused. Evaluate 128 seeded assignments per
session; choose up to three using matching RUN only. Repeat 0 is primary;
others measure split sensitivity and never multiply biological sample size.

Require the existing mean-rate (10%), stability (0.05), field-area (20%) and
global RUN gates (median difference <=2 cm, p75 <=5 cm, each median <=20 cm,
each p75 <=40 cm). Also require normalized regional firing profiles across a
fixed 3x3 spatial tiling to differ by <=20% in every region. Choose qualifying
partitions by smallest regional profile discrepancy, then candidate ID.
Freeze choices before confirmation RUN; no replacement after failure.
Regional confirmation errors are reported, not silently treated as equal.
Local matching gate: >=10 confirmation windows in a region, median-error gap
<=5 cm; unobserved regions are unsupported. Local failure blocks a strong
biological conflict claim, but not a labeled technical readout.

## Decoding and Disagreement

Use independent flat-prior Poisson decoding, 20 ms windows advanced by 5 ms,
the existing 8 cm grid, first-half RUN maps, no dynamics or smoothing prior.
Primary endpoint discrepancy is distance between posterior means. Also report
posterior overlap, fine/coarse cross-support, within-event median separation and
the longest run with means >=40 cm apart while each RMS width <=20 cm.
Overlapping windows are never counted as independent observations.

An A claim is the 20 cm or 40 cm neighborhood around A's posterior mean.
B support means at least 50% of B's posterior is in that neighborhood.
An additional resolved-disagreement flag preserves the distinction between
conflict and insufficient information. It does not prove neural multiplexing.

## Conditional Single-Path Simulations

For each fixed candidate and confirmed split, generate two paths for each of:
1. first-half-map conditional multinomial spiking;
2. second-half-map plus per-cell persistent lognormal gain (sigma=0.3).

Both groups observe the same latent path. One path is calibration, the other is
an independent null test. Simulate on 5 ms base bins, preserving EXACT observed
per-group total counts in every base bin, then aggregate into overlapping 20 ms
frames. Thus duration, base support totals and overlapping-window structure
are preserved. Active-cell support is measured, not assumed matched. This
conditional generator is a model-check, not an exact unconditional Poisson null.

Paths start uniformly on supported spatial bins, with speed drawn from
0,100,300,600,1200 cm/s, random direction, reflected at the rectangular boundary;
projection onto supported bin centers is recorded. No real posterior selects
the simulated truth. For the injected-conflict control, B follows a reflected
spatial alternative to A; evaluate only endpoint separations >=40 cm.

Check real/simulated active-cell distributions and median error against known
truth. Retain all low-support rows. A strict subset requires each population
>=3 endpoint spikes and >=2 active cells. Null calibration uses endpoint spike
strata (minimum across groups: 0-2,3-5,6+) and has separate test draws. At a
nominal 5% tail: independent-test FPR <=7.5%, known-path median error <=20 cm,
active-support KS <=0.1 and injected-conflict sensitivity >=80% are continuation
gates, not assumed outcomes. Reweight null rows by joint spike/active-support
strata for sensitivity; report unmatchable real rows explicitly.

Do not infer structured biological conflict unless BOTH generator checks and
regional RUN matching pass. Low recovery or low power yields unresolved, not
evidence for a common path. Simulations do not recreate candidate detection;
their inference is conditional on the frozen observed population bursts.

## A-Only Reliability Model

Fixed predictors: log endpoint spikes, active cells, normalized entropy,
posterior RMS width, peak posterior probability, A-only local matching-RUN
error, within-A endpoint stability, population coverage at A's mean and cell
count. No B feature, event outcome or accepted flag enters prediction.

Compare prevalence, spikes-only, entropy-only, spikes+entropy, and full models.
Use logistic regression with fixed C=1; training-only imputation/scaling. Outer
leave-one-rat-out groups all sessions, splits and simulations from a rat.
Balance training rows by rat, session, event and split. No threshold tuning.

Predict fine and coarse support separately, enforce coarse probability >=fine.
At 0.8 predicted support: destination supported; otherwise at 0.8 coarse support:
only coarse region supported; otherwise content unresolved. Keep B truth labels
separate. Summarize coverage, Brier score, log loss and false agreement at equal
rat weight. Full model must improve log loss >=5% vs spikes+entropy and improve
in every rat to justify further development. This is agreement, not true content.

Apply real-trained, rat-held-out models to independent null and conflict draws;
evaluate BOTH B agreement and correctness against A's known endpoint. False
agreement <=10% at coverage >=20% is a known-content continuation gate. Common
incorrect agreements count as errors. Evaluate fixed simulation thresholds on
unseen draws only. No calibration fitted on the test rat or test simulations.

## Artifacts and Audit

Write frozen populations, per-region/global RUN QC, real and simulation event
readouts, support matching/calibration summaries, LOAO predictions and metrics,
per-rat/session summaries, gates, figures and a bounded report. Hash inputs,
code and protocol. Preserve all failures. Independent audit must test disjoint
IDs, fixed selection hashes, posterior reconstruction, simulator count/overlap
invariants, absence of B in prediction features and held-out-rat separation.
No raw dataset is copied into the paper package.
