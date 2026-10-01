# Do hippocampal ripples prevent relapse to obsolete routes?

## Target and decision rule

Test selective old-memory interference, not another demonstration of trajectory
instability. The original study already established that closed-loop ripple
disruption destabilizes routes after first optimized-route use. The proposed
extension asks whether **the next trial after successful optimized-new-route use
selectively reinstates the obsolete route rather than another deviation**.

The first implemented stage is a fail-closed public-release audit. Route examples
and audit labels are descriptive. No new treatment contrast is opened until
chronology, contingency, cohort, and route-label identifiability are verified.
Stopping at this stage is explicitly allowed by the experiment's decision rule.
It is not evidence against the biological hypothesis.

Sources:

- Igata, Ikegaya and Sasaki (2021), [article](https://doi.org/10.1073/pnas.2011266118).
- Full [SI package](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC7817193/supplementaryFiles).
- Public release v1, [Mendeley Data](https://doi.org/10.17632/4xk5w69yr5.1).

## Bounded novelty audit

Main Fig5 and SI p12 / FigS15 quantify modified-Levenshtein changes between
successive routes after first start-S-C2 use. The source analysis does not
report a conditional obsolete-versus-other multinomial dissociation. The source
also already describes perseveration on the old route during acquisition.
Therefore neither old-route visits nor a disruption-instability effect is new.
Absence of this particular contrast in reviewed sources is only candidate
novelty, not proof of novelty across the entire literature.

## Source-defined geometry and labels

SI p6 names start u, goal e, old checkpoint s, new checkpoint g. The release
geometry script uses uppercase **U/E/S/G** for those lattices; its lowercase
labels instead refer to external boxes and return alleyway. Each field lattice
is 200 mm wide; the field is 1,000 x 1,000 mm. Boundary points use the first
polygon in the author's ordered geometry. Preserve loops/backtracking: only
consecutive repeats represent dwell and may be collapsed.

The source counts a segment as a named path when string length <8 and calls
the optimized S-C2-G route length <12. Frozen audit labels are:

- New: short start-new-goal, no early old-checkpoint/goal visit; arrival flag
  precedes goal-port contact, no timeout.
- Obsolete: short start-old-goal attempt before visiting the new checkpoint and
  correcting. Old-first without the early goal is not automatically obsolete.
- Other: observable completed nonmatching/exploratory or mixed routes.
- Unclassifiable: missing tracking/task evidence, sensor/string disagreement,
  malformed timestamps, incomplete field passage, or unverified contingency.

Success is a checkpoint/goal sensor proxy, not direct milk-delivery measurement.
The maximum tracking gap is provisionally 250 ms; report 125/250/500 ms and
common lattice shifts of -20/-10/0/+10/+20 mm as diagnostic sensitivities.
These cutoffs are frozen before group contrasts, not claimed as author criteria.
No native result is interpretable merely because these audit labels exist.
The published processed `locus_string_list` is the primary audit representation.
Exact agreement with an instantaneous raw-grid string is reported, not required:
the raw conversion can add short boundary flickers/backtracks. The undocumented
details of the source's string processing remain a limitation. Dwell-frame totals,
documented task sensors, tracking gaps and source-defined endpoint order are
checked separately. A raw-grid disagreement alone is not a tracking failure.
Boundary shifts compare raw-grid labels with their own zero-offset baseline;
native-versus-raw label agreement is a different reported field. Neither this
raw-grid diagnostic nor unpublished string processing proves a validated route
classifier under boundary uncertainty.

## Cohort and chronology

The archive supplies Disrupted rat6-rat11 and Delayed rat12-rat17. Treat these as
**released group labels** linked to the paper's manipulation descriptions; do
not claim independently measured ripple-trigger latency or randomized assignment.
Infer observed active-checkpoint status only from documented arrival flags plus
coordinates, never the S/G filename suffix or obsolete log columns 7/11.
Columns 14/15 are undocumented and cannot resolve the task design by guesswork.

Keep trial_data and return_data distinct. Files encode local trial indices,
not globally ordered recording clocks. Never bridge missing trial numbers,
overlapping clocks, recording blocks, or animals. Hash all NPZ inputs and audit
identical log payloads separately from byte-identical compressed files. Do not
drop inconvenient animals to repair unresolved chronology.

## Planned primary model, conditional on a go

Fit a hierarchical multinomial model over new/obsolete/other next-trial outcomes,
with other as reference, animal-level intercept variation, group, chronological
trial number and prior successful-new-route experience. Freeze priors, covariate
scaling, missing-data strategy and inference diagnostics before calibration.
Primary contrast: immediate-minus-delayed log odds obsolete versus other.
Report all three absolute probabilities and increased obsolete probability as
well as increased obsolete-versus-other odds before a selective-relapse claim.
Successful new-route use is post-treatment: this is conditional behavior, not a
total causal treatment effect. Acquisition and unconditional post-relocation
behavior must be reported separately. Trials are not independent animal replicas.

This downstream model is **not implemented/executed yet**: the current release
audit must first establish a genuine actual-design matrix. A generic multinomial
fit to invented metadata would not implement the scientific plan safely.

## Validation required before a native test

Separate development and frozen validation seeds (431001 / 431002). Use the
verified animal/trial structure, including missingness, for 1,000 validation
replicates per generator: no effect, generic destabilization, selective relapse,
and impaired discovery without selective relapse. Simulate the full acquisition
sequence before conditioning, not only fixed already-successful transitions.
Use a predeclared effect grid and report power rather than selecting a favorable
effect size. Require false-positive calibration under both null and generic
destabilization, interval coverage, route-boundary sensitivity, unequal counts,
sparsity and leave-one-animal-out results. No passing assertion may be based on
zero simulations or a synthetic design substituted for missing public metadata.

## Execution and provenance

Use an isolated committed checkout on gpuserver6000. The detached supervisor
uses a new session, closed SSH descriptors and durable log/status/exit files.
The audit writes atomic progress/checkpoints every 50 files; a resume verifies
the same commit, protocol, sources and input hashes. Final artifacts contain
actual HEAD, dirty flag, command, host, source/input/output SHA256 values.

```bash
python3 scripts/launch_igata_route_audit.py --job-dir <new-job-dir> -- \
  audit_igata_obsolete_route_relapse.py \
  --dataset-root /mnt/lexar4tb/datasets/igata-2021/extracted \
  --archive /mnt/lexar4tb/datasets/igata-2021/archive/igata2021.zip \
  --source-main-xml <saved-public-full-text.xml> \
  --source-si-pdf <saved-public-supplement.pdf> \
  --source-si-text <pdftotext-layout-output.txt> \
  --output-dir <new-result-dir>
python3 scripts/verify_igata_route_audit.py \
  --dataset-root /mnt/lexar4tb/datasets/igata-2021/extracted \
  --output-dir <completed-result-dir>
```

The independent verifier rehashes all inputs and outputs, rechecks trial and
transition accounting, and recomputes the published edit-distance algorithm
using a separate full-matrix recurrence. It never upgrades a feasibility stop
into a biological result. A completed process and a scientific go are distinct.

Archive compact tables, route figures, source review, protocol and provenance in
FlorianPfaff/2026-09-HippoReplayDynamics under a new research folder, without raw
NPZ/LFP/spike files or edits to frozen manuscript claims. Ensemble data are
inadequate for a multi-animal replay-content mediation test. No author request
or new recording is part of this experiment.
