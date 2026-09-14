# Fixed-time encoding-uncertainty outcome: remedy screen failed

2026-09-14. Server6000, detached services. Do not overwrite earlier failures,
change the frozen decision rule or call technical completion a validated remedy.

## Frozen test and execution

Producer/protocol commit `1d8bc26f`; independent audit `c441ffcf`;
non-rescoring report `7d47dd2b`. Eight sessions/four rats in each of PF and hc11,
1600 fixed high-MUA candidate endpoints/dataset, three fixed half-population
partitions; split0 primary. All mean rates and unit QC were training-RUN-only.
Every method used the SAME final20ms, cells, rates' means, grid and spatial prior.
No temporal prior, trimming, new events, changed population or abstention.

Methods were plug-in Poisson, Gamma-Poisson with training sampling uncertainty,
Gamma-Poisson also incorporating training-quarter variance (primary), and
Poisson likelihood tempering matched to primary posterior entropy.
The Gamma approximation is not an exact posterior for smoothed fields.

Both measurement services, audit and reporter finished with status0. Independent
audit reconstructed all16 recordings: 229164 event/method/split readout rows,
687492 population posterior rows and 6297 real/RUN endpoint spike recounts.
No entropy-control population window was unavailable. A separate non-rescoring
check reconstructed all3240 aggregate before/after/paired-difference rows directly
from source event tables with equal-rat weighting; all matched at1e-10 tolerance.
Forty-four focused producer/auditor/reporter/edge-support tests passed; Ruff passed.
The later auditor edit only renames a local variable for lint; its completed-run
implementation and input hash remain pinned to `c441ffcf` in the audit manifest.

## Primary real-event results

Events average within session, sessions within rat, rats equally. Intervals are
descriptive four-rat bootstraps, not a precise population-level guarantee.

| Dataset | Quantity | Poisson | Primary uncertainty | Relative change |
| --- | --- | ---: | ---: | ---: |
| hc11 | A/B posterior-mean separation cm |28.794627|28.584541| -0.7296% |
| hc11 | 3x3 regional posterior TV |0.319761|0.317414| -0.7338% |
| PF | A/B posterior-mean separation cm |45.449571|45.127887| -0.7078% |
| PF | 3x3 regional posterior TV |0.509727|0.507276| -0.4808% |

hc11 separation change=-0.210086cm, descriptive95%CI[-0.296451,-0.062568];
TV change=-0.002347, CI[-0.003540,-0.001342]. Separation decreases in3/4rats;
TV decreases in4/4. These are reproducible SMALL changes, not a practical remedy:
both miss the prospectively frozen10% reduction requirement by a large margin.
The two alternative partitions have hc11 reductions0.66-0.69%, not a rescue.

Entropy-matched Poisson already gives hc11 separation28.641241cm andTV0.318048.
The primary method's additional separation improvement is only0.056700cm,
CI[-0.125006,+0.029218]; additionalTV improvement0.000634,
CI[-0.001398,-0.000082]. Thus most of the tiny agreement improvement is also
obtained by entropy matching, without a specific rate-uncertainty mechanism.
Sampling-uncertainty-only changes real disagreement by merely0.05-0.08%.

## True-position controls

hc11 primary mean A/B held-out RUN error changes -0.0509cm; regional Brier
-0.00115. Simulated mean error changes +0.0064cm(moving), +0.0133cm(gain),
+0.0341cm(stationary), +0.0293cm(late jump). All per-population error upper95
bounds satisfy the predeclared2cm noninferiority tolerance. This avoids a large
accuracy penalty but does not demonstrate useful error reduction.

Mean regional Brier worsens slightly for hc11 stationary/gain/late-jump controls
and PF stationary/gain controls, failing the frozen proper-score condition.
These tiny penalties should not be exaggerated: the decisive failure is the
absence of substantial stability improvement, not severe localization harm.

## Training-map sensitivity diagnostic

A-only q1-versus-q2 posterior regionalTV is positively associated with real
A/BregionalTV: mean within-session/rat rho+.326hc11 and+.092PF, positive4/4rats
in each dataset. This association does NOT establish accuracy prediction.
Against independent B true-position error in held-outRUN, rho is -.045hc11
and-.053PF, negative4/4rats in both. Simulated true-error associations are weak
and inconsistent. No diagnostic threshold was fitted; no selector was validated.

## Decision and remaining claim

FAIL the primary external promising-remedy screen. Do not amplify uncertainty,
tune temperatures or replace splits after seeing hc11 to manufacture a pass.
No disjoint-event confirmation is warranted for this particular method.

This does not show uncertainty-aware decoding cannot work. It shows that this
frozen, independently audited moment-based uncertainty estimator does not
materially reduce the tested instability. No biological replay truth is known;
hc11 POST MUA is not automatically ripple-positive/NREM/replay. Occupied-grid
simulations do not reproduce native noise correlations or verified maze topology.

The original matched-Home/accepted-endpoint result remains untouched. Random-half
raw candidate endpoints are a different test population. The broader goal of a
validated remedy or diagnostic remains OPEN, not completed by this failed test.

## Artifacts

Server root: `/mnt/seagate10tb/florianpfaff/encoding-uncertainty-content-20260914/`.
Subdirectories `pf`, `hc11`, `audit`, `report`. All large posterior/count arrays
stay on server6000. Compact local copy:
`/mnt/c/Users/emper/Documents/codex/2026-09-14/encoding-uncertainty-content-benchmark/`.
