# Regional prevalence calibration: no validated correction

Completed on gpuserver6000, 2026-09-14. Producer protocol committed before
readouts. Results were inspected only after independent reconstruction passed.
All original four targeted matched sessions / three rats retained. No thresholds
were relaxed and no missing population was replaced.

## Why the primary correction fails

The native RUN calibration response gap (mean Home posterior when physically
inside minus outside Home) was:

| Session | Home-rich | Home-poor |
|---|---:|---:|
| Rat1/Open1 | 0.3336 | 0.0579 |
| Rat1/Open2 | 0.2901 | 0.0448 |
| Rat2/Open1 | 0.1180 | 0.0228 |
| Rat4/Open2 | 0.3015 | 0.0304 |

Three Home-poor populations fail the frozen 0.05 response-gap requirement.
Their inverse corrections would amplify response errors by factors >20. The
one remaining Home-poor population in Rat1/Open1 gives an unclipped replay
prevalence of -17.77%, versus +34.59% for its Home-rich counterpart. The negative
estimate is a calibration-transfer incompatibility, not a physical prevalence.
Its real mean score lies below the calibration outside-Home mean. Broadly,
limited regional discrimination and RUN-to-candidate transfer both matter.

For native known-location validation, third-quarter RUN was calibration and
fourth-quarter RUN was evaluation at fixed 20-ms resolution. Do not use the
full-RUN-map sensitivity as held-out validation. The legacy full-RUN-derived
unit/grid eligibility and already selected matched cohort remain limitations.

## Agreement without recovery is not a remedy

The count-profile-matched synthetic calibration narrows the candidate high/low
prevalence discrepancy in Rat1/Open1 and Rat1/Open2 to 2.03 and 2.32 percentage
points. This is tempting but insufficient: applied to natural held-out RUN in
those sessions, mean absolute prevalence error worsens from 4.07 to 22.36 pp and
from 3.97 to 20.10 pp. Rat2/Open1 and Rat4/Open2 remain unavailable under the
frozen response-gap requirement. This sensitivity cannot be promoted as success.

The native validation stress-panel baseline error of 27.59 pp refers to averages
over imposed class proportions 5%,15%,30%,50%,75%, NOT natural RUN occupancy.
All such panels share observations and are not extra independent replicates.

## Audit and provenance

- Producer v1 and its failed audit retained. An exact raw spike at 28579.1 s was
  double-counted because independently added floating-point endpoints overlapped
  by 3.64e-12 s. No scientific outcome had been inspected when this was fixed.
- Producer v2 uses a shared 13-edge array per RUN parent window. No replay
  endpoint or simulated draw changes. A real-timestamp regression test covers it.
- Independent v2 audit reconstructs 374,228 regional scores, 1,048,097 spike
  counts, 32 calibration rows and 464 prevalence rows across all four sessions.
- Raw native spike assignment, original frozen endpoints, cell IDs, rates, Home
  mask and all simulated count draws are checked. Original early/full-RUN
  endpoint probabilities match to numerical tolerance.
- 56 focused tests pass; all new scripts/tests pass Ruff. Benign optional
  Matplotlib Axes3D/backend warnings do not affect the 2D report.
- Producer commit: 603322ef; auditor/reporter core: 58fc99fc. Plot/wording changes
  after outcomes are non-rescoring and do not change any estimate or gate.
- Unrelated untracked regional-bounds work was left untouched. Manifests record
  dirty worktree state and exact input/code hashes rather than calling it clean.

## Decision and remaining goal

PF development screen fails; independent validation not attempted for this
failed remedy. No corrected replay-Home fraction is justified. Calibration is
not information creation, and matching a candidate count distribution does not
guarantee transfer of its regional likelihoods. A future remedy must demonstrate
known-content recovery and population stability on independent recordings; that
user goal remains unachieved.

Server artifacts:
- /mnt/seagate10tb/florianpfaff/regional-prevalence-calibration-v2-20260914
- /mnt/seagate10tb/florianpfaff/regional-prevalence-calibration-audit-v2-20260914
- /mnt/seagate10tb/florianpfaff/regional-prevalence-calibration-report-v3-20260914

Local compact copy:
/mnt/c/Users/emper/Documents/codex/2026-09-14/regional-prevalence-calibration
