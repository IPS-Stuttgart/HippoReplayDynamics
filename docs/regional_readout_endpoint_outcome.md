# Regional readout / terminal-window diagnostic outcome

2026-09-16. Eight PF sessions / four rats. Frozen primary diagnostic plus
explicitly post-hoc zero-mass control; no real content estimation or hc-11.

## Decision

The tested three-category compression is not the main 20-ms pooled-calibration
limitation. Knowing each simulation's generator has a much larger effect:

| Full-population mixed validation, 20 ms | Absolute error | Within 5 pp |
|---|---:|---:|
| Ternary / pooled | 8.72 pp | 40.0% |
| Continuous-neutral / pooled | 8.62 pp | 39.2% |
| Ternary / generator oracle | 2.65 pp | 87.3% |
| Continuous-neutral / generator oracle | 2.46 pp | 88.9% |

These are equal-rat descriptive means, not independent-animal confidence
intervals. Even the oracle does not meet the 90% target overall, and all
session/population/prevalence conditions must pass separately.

## Important interpretation

Raw BF endpoint AUC is .938 for stationary and .108 for late-jump simulations.
In the latter generator, earlier spikes come from a deliberately opposite-region
origin. The oracle can invert that cue (.892 calibrated AUC); it is not evidence
that the last five milliseconds themselves contain equally strong endpoint
information. This dependence was imposed by the simulation and must not be
attributed to real replay without evidence.

The final 5 ms have about .51 QC-population spikes on average and 61% silent
windows, compared with about 4.6 spikes and 2.5% silence for the 20-ms readout.
No short windows were excluded. On known late jumps, ternary calibration error
increases from 3.00 to 7.21 pp. The frozen continuous KDE increases from 2.50 to
21.64 pp. A separately declared post-hoc model treating neutral zero scores as
a point mass reduces the latter to 6.99 pp (47.2% within 5 pp). Thus the 21-pp
error combines a density-estimation problem with a sparsity limitation. Do not
attribute all of it to lack of neural information. Stationary controls show
similar short-window deterioration.

## Boundary and next question

No validated real-content remedy emerged. These results motivate bounds that
allow uncertainty about within-window dynamics, or an explicitly different
window-averaged estimand. They do not prove every instantaneous endpoint is
unidentifiable. Do not silently relabel window content as terminal content,
use oracle generator labels on real events, relax the error budget, or proceed
to hc-11 as if validation passed.

## Artifacts

Primary: /mnt/seagate10tb/florianpfaff/regional-readout-endpoint-pf-20260916
Audit: /mnt/seagate10tb/florianpfaff/regional-readout-endpoint-pf-20260916-audit.json
Post-hoc: /mnt/seagate10tb/florianpfaff/regional-readout-zero-mass-pf-20260916
Synthesis: /mnt/seagate10tb/florianpfaff/regional-readout-endpoint-pf-report-20260916
Code: /home/florianpfaff/HippoReplayDynamics-content-stability-20260914

Primary independent audit passed: 46,080 estimates, 320,000 short-window
likelihood/call reconstructions, 128 second-optimizer refits, all exact total
counts/truth/acceptance and source/output hashes. The primary run reconstructed
1,600 complete panels with identical original seeds/counts/targets/active gates.
The zero-mass follow-up contains 6,400 additional estimates and separate tests.
Code is uncommitted; existing work was not reverted.
