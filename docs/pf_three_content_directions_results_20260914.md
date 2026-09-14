# Three Replay-Content Directions: Results and Decision

All three requested directions have now been tried with real PF data. This is
an experimental-development result, not three positive biological findings.
All computation ran on gpuserver6000 in detached services. Original data and
previous frozen results were preserved.

## 1. Same Cell Count, Different Content Readout

Source: pf-matched-population-content-20260913, producer 05ebb0f3; audit/results
c27cedf3. Independent audit passed: 84 RUN blocks, 528 replay endpoint checks.

Equally sized populations were matched on early RUN firing rate, field stability,
field area and global held-out RUN error, then independently confirmed. Targeted
Home-coverage matches passed in 4/8 sessions, spanning three rats. Whole-tetrode
matches passed in three sessions/two rats. Failures were not replaced.

Home-rich versus Home-poor populations assigned 11.38% versus 2.80% posterior
probability to within 20 cm of inferred Home at unchanged candidate endpoints:
+8.58 percentage points, positive in all three retained rats. For previously
accepted trajectory-segment endpoints the difference was +6.87 points. The
whole-tetrode candidate-endpoint contrast was +4.08 points, limited to two rats.

Important mechanism/caveat: local Home RUN accuracy and actual replay spike
support were not matched. Rat4/Open2 had 68 cells in each group and global RUN
errors 12.37/11.91 cm, but local Home errors 10.74/18.43 cm. Thus global QC can
hide regional sensitivity. This does not show that the published planning
result is an artifact. The coverage contrast was deliberately maximized, not
sampled to estimate typical recording bias; Home coordinates remain inferred.

## 2. Do Disjoint Populations Support Conflicting Trajectories?

Source: pf-independent-population-content-20260914. Initial producer 09ae1545;
summary gates were corrected in a documented non-rescoring review.

Pairs of similar RUN place-field profiles were split into disjoint equal-sized
A/B populations. A fixed 3x3 spatial coverage profile had to match within 20%,
alongside the previous global descriptor/error gates. Up to three splits were
frozen per session before confirmation. Split 0 was primary; later splits never
replaced a primary confirmation failure.

Primary: 1,881 unchanged high-MUA candidates in Rat1/Open1, Rat1/Open2,
Rat2/Open2 and Rat4/Open2: four sessions, three rats. This is NOT the same cohort
as direction 1. Rat3 had no qualifying balanced partition; Rat4/Open1 failed
confirmation. Rat2/Open1 passed only secondary splits.

Observed pooled median A/B endpoint-mean separation was 34.96 cm. For the same
event-count profiles generated from ONE known path, median separations were
38.60 cm with matched maps and 40.88 cm with map drift/per-cell gains.
Those pooled medians are descriptive; the formal sensitivity averages sessions
within rats after reweighting minimum-spike/active-support strata.

| Rat | Real-minus-null mean separation, matched map | Map-drift/gain null |
|---|---:|---:|
| Rat1 | -1.11 cm | -8.07 cm |
| Rat2 | -10.22 cm | -20.17 cm |
| Rat4 | +1.43 cm | -12.00 cm |

These comparisons do not establish the cause of real disagreement. Conditional
simulations preserve exact 5-ms per-population spike totals and overlapping
20-ms windows, but not exact active-cell support or real noise correlations.
Even the improved spike-supported null retains sizeable known-location errors.

Primary null-tail calibration used spike-only strata, with joint-support
reweighting separate. Corrected independent-null false-positive rates were
3.46-10.61% (matched) and 5.05-9.42% (drift/gain), versus a nominal 5%. Injected
conflicts were detected only 54.69-67.04% and 46.24-69.90%, below the frozen 80%
power gate. Active-support and per-population recovery checks also failed.

No primary split matched local RUN accuracy in all nine regions: the four
sessions matched 7/9, 8/9, 6/9 and 8/9. Therefore both null-calibration and local
RUN gates block biological interpretation.

**Decision:** no evidence established for multiplexed/conflicting replay.
This is unresolved, not proof that every event contains one shared trajectory.
Ordinary single-path decoding error can produce disagreements of this size.

## 3. Can A Predict Whether Previously Unused Cells B Support Its Content?

Seven predeclared models/policies were compared: prevalence, spike features,
entropy, spikes+entropy, a fuller A-only model, and simple spike/entropy
threshold policies. The full model also included posterior width/peak, local
RUN error, within-A temporal stability, local coverage and cell count.

Leave-one-rat-out training excluded ALL sessions and splits of the test rat.
Only real primary-split candidate endpoints trained the models. All imputation,
scaling and entropy thresholds used training rats only. Secondary splits were
prediction-only sensitivity, not extra independent animals. B features never
entered prediction. Simulation calibration rows never trained the model.

Target: whether B places >=50% posterior mass within 20/40 cm of A's mean.
This is independent-population support, NOT known spatial truth.

| Target | Spikes+entropy log loss | Full A-only log loss | Relative reduction |
|---|---:|---:|---:|
| Within 20 cm | 0.3271 | 0.2765 | 15.46% |
| Within 40 cm | 0.5390 | 0.4885 | 9.38% |

Both reductions were positive in each of the three held-out rats. The fuller
feature model improves probability forecasts in this small development pilot;
the improvement cannot be attributed specifically to local coverage without
further ablation. Three rats do not provide broad external validation.

At the frozen 0.8 predicted-support threshold:

- Destination supported (20 cm): 0/1,881.
- Only coarse region supported (40 cm): 16/1,881; B supported 10 of those 16.
- Content unresolved: 1,865/1,881.

Equal-rat claim coverage was 1.19% (raw event fraction 0.85%). No claims occurred
in Rat1. A global error rate that silently omits that undefined rat denominator
must not be treated as a three-rat validation.

Known-path simulations received claims on only 0.75% of events after equal-rat
weighting, far below the frozen 20% usable-coverage gate. Some rats had zero
claims, so correctness rates were undefined, not perfect. The paired injected
conflict control changes only B while keeping A identical; A-only forecasts
necessarily stay identical, illustrating their dependence on population
agreement assumptions, not an ability to see unobserved interventions.

**Decision:** promising improvement in probabilistic prediction, but NO validated
high-confidence destination-certification rule. Do not lower thresholds after
seeing these results and describe the resulting labels as confirmed.

## Engineering and Reproducibility

- 76 relevant tests pass; Ruff passes for all new producer/auditor/reporter/tests.
- Independent new-run audit reconstructed 68 RUN blocks and 546 real/simulation
  endpoints, including conditional counts, overlap, grid/cell alignment and
  posterior probabilities through a separate dense implementation.
- Confirmation failures contributed no replay rows; original hashes verified.
- Regression tests cover zero calibration coverage, per-population failures,
  cross-generator gates, held-out-rat leakage, B-feature perturbation, constant
  labels and repeated-row weights.
- Matplotlib optional Axes3D/backend warnings and sklearn/SciPy deprecation
  warnings were benign; no test or scoring failure remained.

## Recommended Paper Role

The strongest current contribution remains direction 1: comparable global
decoder quality does not guarantee comparable regional content estimates.
Directions 2 and 3 supply limits and validation attempts: disagreement alone
does not establish multiplexing; improved probability prediction does not yet
justify certifying destinations. The next methodological step would be improved
local recovery and calibration on an independently frozen evaluation, not a
new biological claim from these pilot outputs.

## Result Locations

- Server code: /home/florianpfaff/HippoReplayDynamics-independent-content-20260914
- Direction 1: /mnt/seagate10tb/florianpfaff/pf-matched-population-content-20260913
- Directions 2/3: /mnt/seagate10tb/florianpfaff/pf-independent-population-content-20260914
- Reliability tables: the latter directory's reliability/ subdirectory.

Raw spikes, position and source arrays remain on gpuserver6000. Compact result
tables, figures, frozen IDs, protocols and provenance can be shared separately.
