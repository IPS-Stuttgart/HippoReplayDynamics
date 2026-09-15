# Session-local region-protected screening: development failure

## Decision

Do not promote this rule to independent recordings. Candidate agreement improves
and the previous Rat4-only Home-gap improvement is no longer the whole story,
but accepted-segment and known-position safeguards fail. This is concrete
progress in identifying a failed remedy, not completion of the active goal.

Protocol frozen at c9c4e3cc; producer/auditor/tests at 6960f81e. No thresholds,
tree settings, sources, candidate windows or cell populations were changed after
fitting. A non-rescoring transport diagnostic was added at 59ff6bb5 afterward.
Earlier failed methods remain preserved. This development series has already
examined the evaluation banks and is not blinded independent confirmation.

## Design

Fit four session-specific models using only that session's earlier RUN Q3 and
two separate calibration simulation banks. Each source and true Home/non-Home
class receives equal calibration weight. Observable paired-population features
define a fixed regression tree; its leaf retention probabilities are optimized
with separate error and Home Brier constraints for each true class. This uses
both populations, not a blind forecast of unseen B. Truth is a training target
and validation reference only; it is not a replay selection feature.

Retain exactly half, with seeded ties: 919/1,836 original candidate endpoints,
257/513 previously accepted-segment endpoints. Each readout retains its original
20-ms window, flat prior, Poisson model, grid and cell set. No rescoring or
retiming was introduced by the diagnostic. Selection changes the analyzed
subset, not the data quality of every original event.

## Real-data results

Equal rats after equal sessions within rat, early maps primary:

| Original candidate endpoints | All | Local half |
| --- | ---: | ---: |
| Home posterior gap (percentage points) | 9.0003 | 6.6674 |
| Mean position separation (cm) | 42.1505 | 34.1585 |
| Regional posterior TV | 0.497130 | 0.408748 |
| High-population normalized entropy | 0.696284 | 0.662460 |
| Low-population normalized entropy | 0.732813 | 0.716281 |

The changes are about 25.9%, 19.0% and 17.8% reductions in Home gap, separation
and TV, respectively. Candidate Home gap improves in every rat:
Rat1 12.0779->10.5926 pp; Rat2 6.6206->4.0309 pp;
Rat4 8.3024->5.3788 pp. Both entropy guards pass.

Full-map candidate Home gap is 8.5829->5.8083 pp and passes its rat safeguard.
Accepted-segment early-map gap improves pooled 7.3047->4.8185 pp, but Rat2
worsens 6.6943->6.9731 pp, so the accepted-segment safeguard FAILS.
Full-map accepted-segment gap improves 6.8679->2.2712 pp and passes.
These probabilities concern inferred Home, not known replay destinations.

## Known-position accuracy is still not protected

On native Q4, the high population in Rat4 worsens in class-balanced physical
error (47.8047->47.8959 cm) and Home Brier (0.326260->0.331020).
Within true Home positions it worsens more clearly in physical error:
38.3078->40.9469 cm, Home Brier 0.649945->0.6599. The selected true-Home
subset is 222/460 observations. Rat2 high-population Home error also rises,
34.0510->35.6447 cm. Neither class is discarded: every source/session retains
at least 20% of both true classes.

Several simulated source/side losses also worsen. All per-session, per-class
numbers are reported in truth_class_changes.csv. Some failed differences are
small, and these point-estimate guards are not significance tests. Their
failure means the predeclared remedy criterion was not met, not proof that
all local-calibration methods are harmful or that true replay is absent.

## Distinguish hard selection from transfer

Post-hoc diagnostic only: keep the SAME frozen models, compare their fractional
retention probabilities with the original hard top-half decision. No new rule,
new score, parameter search or replacement primary result is introduced.

| Source | Fractional regional losses worsened | Hard-half losses worsened |
| --- | ---: | ---: |
| Calibration, all three banks | 0/96 | 7/96 |
| Native held-out Q4 | 11/32 | 6/32 |
| Matched Poisson test | 6/32 | 6/32 |
| Gain4 test | 8/32 | 6/32 |
| Conditional test | 6/32 | 5/32 |
| Map-drift test | 5/32 | 4/32 |
| Shared-assembly test | 12/32 | 13/32 |

Each 32 is four sessions x two true classes x two populations x two losses,
NOT 32 independent biological experiments. The diagnostic uses tolerance1e-8
to disregard solver rounding; the original numerical science gates did not
change. Fractional calibration retains exactly half of each true class as
certified; hard calibration does not (native Q3 range38.2%-55.6%).

Thus the empirical training protection can be lost through hardening, but
hardening is not the only failure: even unrounded probabilities lose regional
protection on held-out observations. Sampling noise, fitting and distribution
changes are not separated here. These results do not validate fractional
weighting as an alternative remedy or establish an impossibility theorem.

All four training progress optima are below the full t=1 target:
0.58456, 0.43846, 0.64567, 0.40534. Feasible partial training progress was
never a guarantee of full numerical recovery on held-out data.

## Verification and artifacts

Server gpuserver6000, detached user service local-content-screen-20260915,
terminal exit0 in12.09s; completed2026-09-15T00:55:46.869682UTC.
Root: /mnt/seagate10tb/florianpfaff/local-content-screen-20260915.

Independent audit refit all four local trees and checked eight primal/dual
certificates, source audit/hash chain, unchanged evaluation rows, exact
selection, classwise accuracy and all gates. It verifies 33,910 calibration
rows and 109,274 evaluation rows, which include simulated truth banks and map
sensitivities, NOT that many independent replay events. Raw likelihoods reuse
the previous independent reconstruction, rather than a new decoder implementation.

Largest primal residual2.69e-14, stationarity1.28e-12, duality gap1.02e-12.
Measurement manifest SHA256:
41ef42d44c874ed695655a3e9b903b210f3b53b34ff05a7bc2a2b315bbf67592.
79 related tests passed; the six focused tests passed again with the transport
reporter; Ruff passed. Source manifests and frozen models were not changed.

Authoritative compact report is report-final/, with all classwise tables,
transport diagnostics, figure and report manifest. It is copied locally to
/mnt/c/Users/emper/Documents/codex/2026-09-15/local-content-screen/.
The report and figure hashes match the server and the figure was inspected.

External validation was NOT run. The validated-remedy/diagnostic goal remains
OPEN. No failed condition was removed or relabeled as a pass.
