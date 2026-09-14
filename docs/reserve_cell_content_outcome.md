# Reserve-cell acquisition: accuracy improved, instability not remedied

Completed on gpuserver6000, 2026-09-15 (local date). Frozen design `d1df761b`,
producer/auditor `b891053c`, reporting `faaae8fa`, exploratory decomposition
`265d3f05`. The detached `reserve-cell-content-20260915` service exited 0.
Scientific development screen: FAIL. No external validation was launched.

## Original target preserved

The original four high/low Home-tuning pairs, three rats, 1,836 fixed candidate
endpoints and 513 previously accepted segment endpoints all remain. No endpoint
shift, event filtering, new temporal prior or posterior rescaling. Poisson
decoding remains independent per 20-ms window on the unchanged grid.

Both original cell sets were retained. The unused-cell reserve had 34/36/30/21
cells; each population received 17/18/15/10 additional cells, respectively. The
two added sets were disjoint and contained no original cell from either side.
Thus shared-cell overlap was EXACTLY unchanged. This is not an intervention
that can be applied to a recording if those extra neurons were never recorded.

Selection used only early-RUN maps and the first 20-ms bin of each native Q3
250-ms parent. Greedy allocation maximized the reduction in class-balanced Home
Brier loss. All assignments were saved before inspecting Q4 or target arrays.
Twenty seeded random allocations used the identical extra-cell budget. Their
metrics, not their posteriors, were averaged within session.

## Result on original regional content

Equal sessions within rat, then equal rats; gaps are absolute differences of
session-mean posterior Home mass, not true Home replay proportions.

| Measure | Original | Random additions | RUN-guided additions |
| --- | ---: | ---: | ---: |
| Early-map candidate Home gap (pp) | 9.0003 | 8.9266 | 8.9652 |
| Full-map candidate Home gap (pp) | 8.5829 | 8.5258 | 8.6173 |
| Early-map accepted-segment Home gap (pp) | 7.3047 | 7.1443 | 7.1846 |
| Full-map accepted-segment Home gap (pp) | 6.8679 | 6.7529 | 6.8425 |
| Early candidate mean-position separation (cm) | 42.1505 | 43.2104 | 42.7890 |
| Early candidate regional posterior TV | 0.49713 | 0.50916 | 0.50609 |

The main discrepancy is essentially unchanged. Early-map candidate separation
increases 1.51%, and regional TV increases 1.80%. Rat1 improves on these two
agreement metrics but Rat2 and Rat4 worsen. Selected additions beat random
additions on separation/TV, but not on the primary Home-content gap. Neither
comparison reaches the frozen improvement requirements. Some rats worsen even
where the accepted-segment pooled gap decreases, so its no-rat-worsened gate
also fails. Neither posterior becomes more diffuse: high/low mean entropy falls
from 0.6963/0.7328 to 0.6679/0.7031.

## Known-position and known-region accuracy

Q3 training Brier decreases in both populations of every session. More
importantly, Q4 physical error AND regional Brier improve in both populations
of every retained rat. Equal-class/equal-rat averages, with population sides
averaged for this overview:

| Known truth | Physical error original -> selected (cm) | Brier original -> selected |
| --- | ---: | ---: |
| Native RUN-Q4 | 47.4156 -> 45.5236 | 0.38061 -> 0.37579 |
| Matched Poisson | 44.8312 -> 42.7181 | 0.37613 -> 0.36963 |
| Gain x4 | 28.8909 -> 26.4368 | 0.28140 -> 0.28144 |
| Endpoint-total counts | 36.7686 -> 34.2332 | 0.33071 -> 0.32595 |
| Late-map drift | 40.6094 -> 38.3447 | 0.34069 -> 0.33742 |
| Shared assembly | 54.6466 -> 53.9746 | 0.37193 -> 0.36833 |

For RUN, random addition already gives 45.8662 cm and Brier 0.37682, so the
incremental benefit of targeting is small. The 20-ms errors are NOT the 250-ms
RUN median errors used in the original global-QC matching. Both truth classes
are equally weighted; no difficult region or silent window is excluded.

Despite these pooled improvements, the stricter per-rat high-population Brier
checks fail for gain4, map drift and shared assembly, and shared-assembly high-
population physical error also fails. These are failures of the frozen no-harm
screen, not a claim of statistically certain harm. Passing on native RUN does
not cancel failures under declared mismatch conditions.

## Exploratory explanation, not a new success criterion

Using only audited event errors (no rescoring), decompose known-position errors:

    separation^2 = error_A^2 + error_B^2 - 2 error_A dot error_B

On native Q4, with equal truth-class/session/rat weighting:

- Summed A+B mean-squared localization error: 6430.50 -> 6067.23 cm^2.
- Twice the mean error dot product: 4415.41 -> 3909.33 cm^2.
- Mean squared between-decoder separation: 2015.09 -> 2157.91 cm^2.

The reduction in shared error alignment exceeds the reduction in individual
squared errors. Therefore disagreement can increase while localization improves.
The dot product includes systematic bias; it is NOT centered error covariance
and does not identify a neuronal noise mechanism. Real replay lacks positional
ground truth, so this exact truth-error decomposition cannot certify its content.

This post-outcome explanation does not replace any frozen gate. It reinforces
the need to evaluate agreement, probability accuracy and localization separately.

## Verification and decision

Independent reconstruction covered every greedy calibration choice, all random
allocations, 2,404,028 event/method rows and 4,808,056 population posteriors, along
with original baselines, full primary event tables, equal-rat summaries and gates.
These large totals include reused truth banks and draws, not independent replays.
The native-audited source snapshot was unchanged. 35 related tests and Ruff pass,
including rehashed-table tamper rejection and the non-rescoring report.

Source root: `/mnt/seagate10tb/florianpfaff/reserve-cell-content-20260915`.
Authoritative measurement: `measurement/`; verification: `independent_audit.json`.
Final non-rescoring tables/figure: `report-final/`. Earlier `report/` and `report-v2/`
are preserved; the latter introduced the explanatory decomposition. The final
report explicitly reads event IDs as strings, avoiding a mixed-type warning.
Manifest SHA256:
`4bc500f65d77de92fd45f1e7a4b0bf1d48c73e2199fe66e397593ff2bd951d6d`.

This is a useful negative intervention test, not a validated remedy. Additional
RUN-selected neurons modestly improved known-position accuracy without resolving
the original regional-content instability. The independent-data objective remains
open; no external cohort was scored and no thresholds were changed after results.
