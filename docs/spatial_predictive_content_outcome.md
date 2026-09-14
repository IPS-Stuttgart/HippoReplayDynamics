# Within-bin spatial predictive diagnostic: development failure

Completed on gpuserver6000, 2026-09-15. Frozen protocol `4feda190`, numerical
producer/auditor `72ee02c8`, non-rescoring reporter `ba531c15`.
The detached service `spatial-predictive-content-20260915` finished with exit 0.

## Question and retained scope

Do spike identities from internal held-out cell folds support the same location
as other cells in that population, and does this screen reduce the ORIGINAL
regional-content instability without worsening known-position recovery?

The conditional-on-count compatibility score uses three folds independently
within each original matched population. The minimum of their median scores
ranks candidate windows. It uses BOTH populations offline, not an A-only
prediction of an unused B. No temporal model, future bin, posterior modification
or endpoint shift was used. The three folds are not independent animals.

The four originally matched PF pairs, three rats, 1,836 candidates and 513
previously accepted fixed-segment endpoints are unchanged. Exactly half per
session/source is retained: 919 candidate endpoints or 257 accepted endpoints.
Selection changes the analyzed cohort; it does not correct full-set prevalence.

## Modest predictive association, incomplete instability reduction

The score correlated negatively with endpoint separation in each retained rat:
Rat1 -0.213, Rat2 -0.201, Rat4 -0.166 (within-session Spearman ranks followed by
equal-session averaging within rat). This passes the frozen directional
association gate, but is not an externally validated or quality-adjusted
incremental prediction claim.

| Original content readout | All events, gap (pp) | Predictive half, gap (pp) |
| --- | ---: | ---: |
| Early map, fixed candidates | 9.00 | 8.57 |
| Full map, fixed candidates | 8.58 | 7.71 |
| Early map, accepted segment endpoints | 7.30 | 4.59 |
| Full map, accepted segment endpoints | 6.87 | 4.66 |

The candidate reductions do not reach the required 20%. The fixed accepted-
segment sensitivity improves without a worsened rat, but cannot replace the
primary candidate-set gate after inspection.

Primary early-map candidate mean-position separation falls 42.15 -> 38.87 cm
(7.78%); nine-tile posterior TV falls 0.49713 -> 0.46893 (5.67%). Neither reaches
the predeclared 10% reduction. Low-Home population mean entropy increases
slightly, 0.73281 -> 0.73585, so the no-flattening safeguard also fails.
Approximately 58.47% of candidate pair scores are exactly zero, reflecting
sparse internal-fold observations; tie handling was frozen and all rows remain.

Simple fixed-half baselines are competitive or better. For early-map candidates,
the Home gap is 6.34 pp after entropy ranking and 6.97 pp after spike-count
ranking, versus 8.57 pp with the predictive score. Their regional TV values
are 0.43667 and 0.45105, versus 0.46893. Therefore the new score is not established
as an improvement over ordinary information-quality screening.

## Truth safeguards prevent promotion

Metrics below give equal weight to true Home/non-Home classes within session,
then equal sessions within rat, then equal rats; populations averaged here.

| Known truth | Physical error all -> retained (cm) | Home Brier all -> retained |
| --- | ---: | ---: |
| Native RUN-Q4 | 47.416 -> 46.940 | 0.38061 -> 0.38263 |
| Matched Poisson | 44.831 -> 44.604 | 0.37613 -> 0.37316 |
| Gain x4 | 28.891 -> 25.669 | 0.28140 -> 0.28047 |
| Event-total counts | 36.769 -> 34.556 | 0.33071 -> 0.32382 |
| Late-map drift | 40.609 -> 38.562 | 0.34069 -> 0.33380 |
| Shared assembly | 54.647 -> 54.430 | 0.37193 -> 0.37090 |

Small pooled improvements conceal individual-rat/side failures. For example,
RUN-Q4 high-population error worsens in Rat2 (45.638 -> 45.935 cm) and Rat4
(47.805 -> 48.423 cm); both populations' regional Brier errors worsen in Rat2
and Rat4. The no-rat-worsened requirements fail on RUN, matched Poisson, gain4
Brier and shared-assembly checks. Event-total and map-drift safeguards pass.
No claim of a statistically certain harm is needed: the predeclared uniform
no-harm screen was not met.

Both true classes retain at least 33.45% in every tested source/session, so the
failed screen is NOT caused by complete removal of a hard region. Class-balanced
truth testing was retained, not replaced with a favorable natural-prevalence
summary. No post-hoc seed, coverage or cutoff was substituted.

## Decision

There is a modest developmental association with reproducibility and a useful
accepted-segment sensitivity, but no validated remedy. The diagnostic does not
meet its original-content, disagreement and known-truth safeguards. Independent
recording validation was NOT launched. The broader goal remains open.

## Verification and artifacts

Server root: `/mnt/seagate10tb/florianpfaff/spatial-predictive-content-20260915`.

- `completion.json` verifies unchanged source files against the snapshot bound
  to the fresh native-count/simulation audit from the preceding mixture test.
- `measurement/` retains all event rows, internal cell assignments, summaries,
  correlations, gates and code/input hashes.
- `independent_audit.json` separately reconstructs all 109,274 event/source/map
  rows and 655,644 internal fold-predictive scores from cached counts/rates,
  posteriors, deterministic selection, truth metrics, aggregation and gates.
- `report/` is a non-rescoring report, five compact tables, figure and hash manifest.
- Source manifest SHA256:
  `1ae807592b9430b9ed60d3a7f0acaba2b0c837d48161128d998510664023e056`.
- 21 related tests passed, including synthetic shared versus incompatible
  locations, silence neutrality, gain invariance, fold/selection determinism,
  truth-label perturbation, hard-region removal, single-rat harm, full artifacts,
  reporting and rejection of rehashed but scientifically tampered summaries.
- Ruff and whitespace checks passed. The figure was visually inspected.

The 109,274 rows include known-truth banks and map sensitivities, NOT that many
independent real replay events. Technical audit success does not change the
failed scientific decision.
