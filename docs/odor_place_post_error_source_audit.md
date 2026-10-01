# Odor-place source feasibility audit

Protocol: `odor_place_post_error_feasibility_20261001_v1`. Feasibility only.
No replay-behavior association is fitted. Existing manuscript and Denovellis
claims are unchanged.

## Version reconciliation

The frozen contract requests Figshare article 19620783 version 1. The actual
[version-1 API record](https://api.figshare.com/v2/articles/19620783/versions/1)
lists `ClaireDataShort-20-Apr-2022.mat` (2,805,269,858 bytes; published MD5
`e54980a8ef70cb062dd8700e71b88b2a`). It contains `SuperRat` records for
26 recording days and six animals. Version 3, not version 1, supplies the
eight-animal `Figure1-6.zip`. Versions were inspected for reconciliation;
the version-3 files were not substituted or downloaded.

The pinned [DANDI release](https://dandiarchive.org/dandiset/001539/0.250815.1203)
contains 38 NWB assets and eight animals. Its session dates describe conversion,
not the source recording chronology. Source `daynum`, timestamp comparison and
epoch information are kept separate from those dates.

## Author code and semantics

Audit the author repository at the paper's archived revision
`88d5f8d2bb39796b8656dc42bd42969a7bbe8697`:
[Jadhav-Lab-Codes-JHB](https://github.com/JadhavLab/Jadhav-Lab-Codes-JHB/tree/88d5f8d2bb39796b8656dc42bd42969a7bbe8697/BetaOdorProject).

- `figures1-6Functions/Behavior/cs_getOdorTriggers.m` constructs left/right
  triggers from odor-solenoid digital outputs, not from the subsequent outcome.
  Its cue and outcome records are therefore different observables.
- `figure7Functions/ParseClaireBehavior.m` gives precedence to run-trajectory
  and run-trial bounds; otherwise it can populate `leftright10` from odor
  triggers **or reward records**. The single compiled column is not by itself
  a verified original stimulus log.
- `figure7Functions/ClaireDataAggregator.m` documents unknown odor identities
  for some unrewarded CS39 trials and reconstructs correct-trial side labels
  from rewards. It also documents manual metadata substitutions and removed
  recording blocks. These are provenance issues to audit, not reasons to
  silently discard errors or unfavorable animals.
- `figure7Functions/A_Figure7Wrapper.m` calls CS39 a partial long track.
  A compiled `longTrack=1` flag is not sufficient to include it as the full
  apparatus. Its status remains source-declared but not independently verified.

The version-1 file records historical `files.taskfile` paths, including run
bounds, odor triggers and rewards, but does not ship those original per-epoch
task/trigger arrays. A historical path is not the missing record. No independent
cue verification is claimed for this release, and no cue is recovered from
correctness plus chosen arm.

## Bounded prior-art comparison

The exact proposed contrast is post-error route content interacting with whether
the next independent cue repeats, predicting the arm correct for the previous
error cue. This is narrower than replay predicting correct versus incorrect
choices. The following audit does not establish exhaustiveness or novelty across
the entire literature.

| Study | Relevant established result | Distinction from proposed contrast |
|---|---|---|
| [Symanski and Bladon et al. 2022](https://doi.org/10.7554/eLife.79545) | Odor-place decisions, beta/respiration coordination, choice-responsive activity and spatial coding | Available article, supplement descriptions and archived odor-project code do not establish a post-error ripple-content by repeated-cue behavioral test. The paper's minimum odor sample is 0.5 s; aborted samples are excluded. |
| [Shin et al. 2019](https://doi.org/10.1016/j.neuron.2019.09.012) | Awake CA1/PFC replay, learning and actual versus alternative trajectories around upcoming correct/error choices | W-track alternation lacks the independently repeated odor cue defining this interaction. Generic replay-choice prediction is already covered. |
| [Tang et al. 2021](https://doi.org/10.7554/eLife.66227) | Theta sequences and CA1/PFC replay coordination around correct/error choices | The proposed cue-gated post-error contrast is not the published actual/not-taken comparison. Generic coordination predicting success is not a new endpoint. |
| [Gillespie et al. 2021](https://doi.org/10.1016/j.neuron.2021.07.029) | Trial-level replay content versus subsequent choices, reward history and recency | This is a close competing prior, but not the same post-error repeated-odor contrast. A positive new result would still need a careful comparison to its null future-choice finding. |

Primary full texts were read via Europe PMC where publisher rendering failed.
Supplement descriptions were checked; not every external analysis-code artifact
was independently executed. Exact-contrast novelty therefore remains provisional,
and the novelty prerequisite is **not promoted to pass** in this delivery.

## Reproduction

The CLI provides `acquire`, `inventory`, `run-qc`, `verify` and `report`.
`run-qc` explicitly records not-run tables after a failed source gate; it cannot
download neural recordings or manufacture decoder metrics. This delivery is the
pinned-source stopgate, not a complete neural adapter for a future amended input.

```bash
python scripts/audit_odor_place_post_error_feasibility.py acquire \
  --dataset-root /home/florianpfaff/datasets/dandi001539-odor-place \
  --source-archive /home/florianpfaff/datasets/dandi001539-odor-place/source/ClaireDataShort-20-Apr-2022.mat \
  --protocol docs/odor_place_post_error_protocol.json \
  --output-dir /home/florianpfaff/odor-place-results/20261001-feasibility-v1
```

Use the same arguments for subsequent stages. Production stages run through
`launch_odor_place_feasibility.py` in a committed, isolated gpuserver6000 checkout.
The download retains partial files, verifies the published MD5 and records SHA256.
Each header has a hard 32 MiB transfer cap, recorded budget use and an
input/protocol/commit-bound checkpoint. No full NWB file is downloaded in this
source-blocked run.

Do not start sequence/statistical calibration under this failed source gate.
A reviewed protocol amendment is required before version 3 can be tested.
That amendment would not guarantee sufficient cohort or RUN-decoder coverage.
