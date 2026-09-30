# Bounded dopamine-dependent ripple-to-RUN updating

Version 1, 2026-09-30. This protocol is fixed before the intervention contrast.
Success is an identifiable, calibrated answer or a documented early stop.
No endpoint substitution, subgroup rescue, decoder development, speed analysis,
IMM expansion, new dataset search, or automatic author email is authorized.

## Biological endpoint and limits

Use the existing conditional spatial-firing score and chronological six-traversal
packets. R0-R2 define maps and inclusion; R3-R4 define preceding change; native
ripple/background recruitment strictly between R4 and R5 supplies the predictor;
R4-R5 define prospective change. R4 counts are shared, not regenerated for the
two comparisons. Use common spatial support in all three readouts, session-local
unit IDs and reference-only cell inclusion. Keep zero-information observations
as missing, never zero effects. No replacement of unavailable packets.

Within each physical track and drug, sum each lag's conditional association
score U and information V across eligible packets, then compute
`A = U_prospective/V_prospective - U_preceding/V_preceding`.
Pair CNO and saline within the same tracks. Average track effects equally within
animal. The primary statistic is
`D = mean_experimental(A_CNO - A_saline) - mean_control(A_CNO - A_saline)`.
All three experimental (Exp_1, Exp_3, Exp_4) and three control (Con_1, Con_2,
Con_3) animals are required. Missing an entire drug/track after information
filtering is an unsupported frozen comparison, not permission to drop that track.

The hypothesis is D < 0; the test is two-sided. Use an animal-level Welch working
interval/test, with small-sample degrees of freedom, calibrated on the complete
design. No cell bootstrap. Treatment/group permutation is not asserted to be
valid without allocation information. Even with exchangeability, six group
labels permit only 20 three-versus-three allocations: they cannot supply an
ordinary two-sided exact p < 0.05 test. Simulation cannot create more animals.

The endpoint is **reference-aligned spatial redistribution**. It does not measure
all field changes, synaptic plasticity or confirmed replay. Recruitment and its
availability are post-treatment variables, so the drug contrast is an
intervention-modified association, not causal mediation by ripples.

## Metadata and freeze

Reuse the paper repository's 135-row `kleinman_track_metadata_request.csv`.
Preserve `animal,session,released_drug,released_novel`. The author supplement
adds/finishes:

```text
physical_track_id,pretraining_track,experience_number_on_track,cno_dose_mg_kg,
injection_to_recording_minutes,same_track_used_under_both_drugs,
confirmed_group,treatment_allocation,author_confirmed,source_reference
```

Booleans must be explicit; group is `experimental` or `control`. Source reference
identifies the author response or attachment, not an inferred identity. The
importer rejects conflicts, unknown sessions, duplicate identities, impossible
doses and contradictory within-track metadata. Blank fields remain unknown.
Track experience numbers count individual recording exposures in chronological
order; ask for clarification if the author's unit of experience differs.

Primary eligibility requires familiar recordings, author-confirmed shared tracks,
both drugs and overlapping experience-order ranges for each track. Report dose,
delay, session and recording-day support; do not silently adjust these away.
An explicit design review is required before freezing, including whether
remaining dose/order/allocation confounding makes interpretation unacceptable.
This conservative overlap rule is predeclared, not chosen from the outcomes.

Supply the existing nonoverlap `packets.csv` with its hash-verified manifest.
Supply a previous-scoring registry JSON with `history_complete_confirmed_by`
and `sources`, each containing `path` and `sha256` of a prior selected-packet
table. Include both prior native pilots and any subsequent inspected readouts.
The driver extracts every past/baseline/target traversal. Any primary packet
whose R3, R4 or R5 was previously scored is excluded and labeled exploratory;
reference-only reuse is allowed. This history still does not make the overall
research program an independent replication.

Design review JSON must name `reviewer`, provide an `assessment`, bind
`metadata_sha256` and `packet_inventory_sha256`, and explicitly set
`design_approved,dose_timing_reviewed,allocation_reviewed,recording_days_reviewed,
experience_overlap_reviewed` to true. Do not approve unsupported comparisons.
The freeze records exact cohort, code, protocol, inputs and raw-file hashes.

Allow five working days of feasibility work, logged in the request state, not
including author waiting time. A draft is not a sent request. After sending,
one clarification round is allowed; after 14 calendar days without essential
metadata, record `blocked_metadata`. Do not fabricate data to unlock analysis.

## Cohort-specific calibration

Reuse the existing bank's observed occupancy, recruitment counts, rates and
elapsed traversal timings for the metadata-confirmed frozen cohort. Generating
maps come from the existing full-RUN simulation utility, but simulated reference
estimation and cell inclusion use R0-R2 only. Readouts are simulated, not reused
native outcome scores. Reference fitting is regenerated independently in every
replicate. No cell identity is linked across sessions.

Development: 64 deterministic replicates per scenario. Validation: a separate
frozen seed namespace with exactly 1,000 replicates per scenario. Development
completeness is required; the 64-replicate null estimate is descriptive, not the
final false-positive gate. One exclusive lock per bank and frozen cohort prevents
rerunning a failed bank with a different output directory. A crash leaves the
lock and logs intact, requiring explicit investigation, not fresh draws.

Nulls: unchanged fields; drug-dependent recruitment and exposure; scalar gain;
heterogeneous cell expression; compound-Poisson bursting; elapsed-time spatial
drift; shared-baseline coupling. Nuisance strength may differ by drug and group.
Elapsed drift is indexed by actual minutes between readouts, not traversal index.
The scalar-gain and cell-expression nulls retain spatial profiles, and shared-R4
coupling uses the same baseline in both contrasts.

Planted effects: +/-0.175, +/-0.35, +/-0.7 in experimental-CNO prospective spatial
alignment only. Report both signs, recovery curves and the smallest tested effect
with >=80% correct-direction power; if none passes, leave MDE missing. These
generator coefficients are not unbiased estimates on the U/V scale.
The 0.35 value is an **engineering benchmark**, not biological relevance.

Every declared null requires complete six-animal inference and the upper endpoint
of an exact two-sided 95% binomial Monte Carlo interval <=0.075. A failed validation
stops this project; do not tune the estimator or choose another endpoint.
Before native scoring, a separate signed adequacy review must bind the validation
manifest and discuss precision/power. It cannot waive a failed null gate. Use
`adequacy_approved,reviewer,precision_and_power_assessment,calibration_manifest_sha256`.

## Native analysis and report

Run once under an exclusive native lock. Preserve all packet/cell scores and
missingness, track effects, six animal effects, D and two-sided interval,
leave-one-animal-out and leave-one-session/day-out summaries. A primary directional
discovery requires a negative interval and all LOO-animal contrasts negative.
The automated discovery flag also requires all session/day deletions to remain
estimable and negative. Session/day instabilities and exposure imbalances must
remain visible and can prevent interpretation; do not use a subgroup instead. No equivalence/absence
claim is automated because no biological absence bound has been specified.

## Novelty assessment

[Kleinman and Foster](https://doi.org/10.7554/eLife.99678) already investigate VTA
dependence of hippocampal ripple/replay localization, reward and novelty. A
generic dopamine/replay association would not be new. The present candidate
extension is a within-track intervention modification of recruitment's
prospective-minus-preceding association with spatial firing.

[Bakermans et al.](https://doi.org/10.1038/s41593-025-01908-3) already relate replay
to subsequent spatial firing and landmark-related map changes. We therefore do
not claim novelty for replay/place-field coupling in general. An interpretable
intervention-by-temporal-direction contrast could be a narrower contribution,
but novelty and publication value depend on the completed evidence, not a claim
of priority established by these two references alone.

## Commands and durable execution

Use an isolated committed checkout on gpuserver4090. The launcher refuses another
host, launches a detached session and retains `job.json`, `job.log`, `launch.json`
and an atomic `status.json` with exit status. Existing result directories and
locks are never overwritten. A successful job exit can mean a documented stop:
read `decision.json`, not just the process exit code.

```bash
python scripts/launch_kleinman_vta_job.py --job-dir /absolute/new/job-dir -- \
  run_kleinman_vta_intervention.py prepare \
  --dataset-root /verified/dataset/root \
  --request-table /existing/135-session-request.csv \
  --request-state docs/kleinman_vta_request_state.json \
  --output-dir /absolute/new/feasibility-dir
```

When metadata arrive, add `--verified-metadata`, `--packet-inventory`,
`--previous-selections`, and `--design-review`. A passing prepare emits
`freeze.json` and `frozen_cohort.csv`. Then run the calibration script with
`--dataset-root --frozen-cohort --bank development --output-dir`, followed by
`--bank validation` through the same launcher. Only a validated bank and approved
adequacy review unlock `run_kleinman_vta_intervention.py score` with
`--dataset-root --frozen-cohort --calibration-dir --validation-review --output-dir`.

The native driver independently rebuilds each packet's histograms, reference
fit and conditional moments before reporting its scores. A mismatch aborts the
run and is retained in the durable error log; there is no automatic rerun.

Archive compact tables, protocol, decisions and provenance in
`FlorianPfaff/2026-09-HippoReplayDynamics`; do not copy raw MAT files or alter
concurrent research work. Before calling results verified, independently
recompute the conditional scores and aggregation, and retain any failures.
