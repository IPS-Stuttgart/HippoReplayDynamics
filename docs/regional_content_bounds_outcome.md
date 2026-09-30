# Regional content bounds: informative only under strong assumptions

2026-09-14, gpuserver6000. The requested feasibility experiment is complete.
It did NOT establish a useful calibrated-content remedy at this operating point.
This is a negative answer to the tested design, not proof that no improved
regional-content method can work. Do not conflate completion of this experiment
with completion of the broader search for a publishable neuroscience result.

## Experiment

- Eight PF recordings/four rats, 1600 unchanged fixed MUA endpoints.
- 1600 previously frozen known-position RUN-Q4 windows; each session's first
  100 windows calibrate and remaining 100 evaluate without refitting maps.
- Four disjoint, RUN-coverage-ordered cell groups; Poisson, uniform valid-grid
  prior, 20-ms windows, inferred Home radius20cm, BF threshold3 and1/3.
- 196608 fresh synthetic target windows: 512 per panel,12 panels per session,
  four conditions. Additional 64000 synthetic calibration windows (2000 per
  class, native and8x count settings, eight source recordings).
- Latent-class EM is a comparison. The primary LP does NOT assume independent
  view errors. It requires conditional calibration-transfer bounds explicitly.
- Whole-population counts are sampled from real endpoints and allocated by a
  conditional multinomial generator; no claim to reproduce all replay biology.

## Real compatibility intervals

RUN-derived sensitivity intervals give [0,1] for both session prevalence and
every real event's conditional Home probability. There are only2-11 Home examples
in the100-window calibration half per recording, so this is not a well-powered
regional calibration set. It is not evidence of absent Home content.

Under exact synthetic-calibration transfer, session Home prevalence bounds are:

| Session | Compatible Home prevalence |
| --- | --- |
| Rat1/Open1 |0-.420|
| Rat1/Open2 |0-.667|
| Rat2/Open1 |0-.580|
| Rat2/Open2 |0-.546|
| Rat3/Open1 |0-.556|
| Rat3/Open2 |0-.598|
| Rat4/Open1 |0-.492|
| Rat4/Open2 |0-.334|

None meets the frozen width<=.20 utility convention. Allowing .10 absolute
change in each conditional readout probability broadens mean prevalence width
to .921; all real event-level conditional intervals become [0,1]. No-assumption
bounds are also [0,1], as required by the mathematical nonidentification case.
Real temporal dependence and calibration transfer remain unverified: these are
assumption-indexed compatibility sets, not certified95% biological confidence
intervals. We have not estimated true Home-replay prevalence.

## Known-truth control and the false-agreement problem

True target Home prevalence is .30 in all synthetic conditions. Mean blind EM
estimates are .408(native), .301(8x), .333(8x quarter-map drift), .554(8x shared
assembly). It can therefore look successful under matching high-information
conditions but mistakes misleading common activity for content under stress.
Shared-assembly Brier loss is .419, versus .210 for a constant prediction using
the known true prevalence. Those target labels never fit the EM or bounds.

The dependence-robust native simulation intervals cover .30 in every target
panel, but their mean width is .572: wide coverage is not useful recovery.
With 8x matching counts width is .642. Under the shared-assembly stress, exact
transfer bounds exclude .30 in21/96 panels despite remaining wide. Their
assumptions have been violated; robust-to-view-dependence is NOT robust-to-any
calibration misspecification. Widening slack restores inclusion but largely by
losing informativeness, and does not validate that chosen slack on real data.
Calibration is reused across12 target replicates: these are conditional stress
frequencies, not independently repeated calibration-coverage experiments.

The81-pattern Bonferroni construction is conservative. It is not a minimax
impossibility result. More informative calibration, efficient simultaneous
intervals, different group construction or another fixed readout would be a
separate frozen experiment, not a retrospective rescue of this one.

## Correction to the proposal's interpretation

The separate original matched-Home audit has four confirmed sessions/three rats
and1836 fixed candidates (also513 previously accepted segment endpoints).
The populations overlap and differ from the new disjoint four-view cohort.
Their equal-rat posterior masses reproduce11.38% versus2.80%.

For the Home-poor population, however,56.2% of fixed endpoint readouts have
BF<1/3,35.4% lie between1/3 and3, and8.5% have BF>3. A mean near the spatial prior
is NOT equivalent to all events being uninformative. These are model-evidence
categories, not verified negative/positive biological content labels.

## Verification and artifacts

18 focused numerical, independent-audit and aggregation tests passed. Ruff
checks passed. The extended independent audit reconstructs source timestamp
counts, all new population posteriors, all simulation draws, calibration bounds,
all prevalence and observed-pattern conditional extrema, EM probabilities and
proper losses. It also checks all Stage0 Bayes-factor rows and128 independently
decoded matched-population readouts. The audit does not validate raw spike
sorting, biological replay truth or EM global optimality.

Production:
`/mnt/seagate10tb/florianpfaff/regional-content-bounds-pf-20260914`

Extended audit:
`/mnt/seagate10tb/florianpfaff/regional-content-bounds-pf-audit-20260914/reconstruction-v2.json`

Report:
`/mnt/seagate10tb/florianpfaff/regional-content-bounds-pf-report-v2-20260914`

All source/script/kernel hashes are in manifests. The producer was run with
new files uncommitted, so its exact file hashes, not the base commit alone,
identify the code. Other work on the shared branch continued independently.
No existing result was overwritten and no new commit/push was performed here.

## Literature and claim boundary

The model-class identifiability assumptions are substantive, not supplied merely
by splitting cells. Allman, Matias & Rhodes (2009):
https://jarhodesuaf.github.io/papers/Latent.pdf

Regional sampling concerns have direct precedent. van der Meer, Kemere & Diba
(2020): https://pmc.ncbi.nlm.nih.gov/articles/PMC7209917/

This bounded test does not establish first-in-literature novelty, multiplexed
content, a speed-uniformity result, or an external replication. Its useful
outcome is an executable calibration-sensitivity diagnostic and a concrete
counterexample to treating population agreement as calibrated spatial truth.

