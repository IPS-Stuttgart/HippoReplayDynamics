# Three-population diagnostic: PF development and AutoPI external failure

## Decision

**Not a validated remedy or positive-support diagnostic. The goal remains open.**

The rule was frozen in commit 29891d63 before three-population measurements or
external decoding. Two observed groups A/B supplied features; a disjoint, equal
sized group C supplied the outcome. All predictors were fit on PF primary real
endpoints, with PF leave-one-animal-out development. AutoPI outcomes never tuned
features, coefficients, population assignments, thresholds or retention.

This follows the failed A-only PF-to-Tanni diagnostic and the failed RUN-balanced
PF-to-Blackstad sampling remedy. Neither earlier failure is overwritten.

## Independent-data feasibility

Native AutoPI extraction ran on gpuserver4090; common encoding, development,
external measurement and auditing ran on gpuserver6000. All jobs used detached
systemd user services and completed with terminal exit status zero after the
documented native-reader fixes. Earlier failed launches are retained in logs.

- 39 native recording directories attempted; 29 yielded native catalogs.
- Eight animals contributed 14,239 known native-rest high-MUA candidates.
- Following common RUN cell QC, only three recordings in two animals had at
  least five cells in each of three groups: 1,623 primary endpoints, 11.40% of
  known candidates. Source failures with unknown counts prevent claiming an
  exhaustive denominator; even the known-denominator fraction is far below 80%.
- The three recordings have 19, 22 and 37 RUN-qualified units. The primary groups
  contain 6, 7 and 12 units, respectively.
- The expected external minimum was four animals and 80% of source endpoints.
  Neither is achieved. Failed recordings are never silently removed.
- These are native-rest population bursts, not independently verified sleep,
  ripple, immobility or biological replay events.

The detailed cluster_info author-good labels and explicit shank_neuron electrode
assignments were used. Missing legacy group entries are reported; direct
curation conflicts and unmapped electrode regions are excluded. The input
resolution is documented in autopi_native_metadata_resolution.md.

## Primary external result

Equal session means within animals, then equal animals; no draw or split is
counted as a new animal.

| Metric | All / random-retention expectation | Predicted-best half |
|---|---:|---:|
| C-support fraction | 0.0000 | 0.0000 |
| A/C regional posterior TV | 0.199490 | 0.242124 |
| A/C mean-position separation, cm | 9.163830 | 11.060670 |
| A normalized posterior entropy | 0.969582 | 0.950464 |
| C normalized posterior entropy | 0.963888 | 0.963988 |

Regional disagreement increased 21.37%; mean-position separation increased
20.70%. Both animals had increased disagreement. Secondary population splits
also showed increases; none replaces the primary split.

Full-model support log loss was 0.027226 versus 0.028889 for the fair pooled-A+B
baseline, a 5.75% decrease. The training-prevalence baseline had loss 0.155876.
This must NOT be advertised as validated content prediction: primary external
labels were all negative (0/1,623 positive C-support examples). Lower loss here
only means lower erroneous predicted support, not demonstrated sensitivity to
genuine supported content. A constant-zero predictor would exploit this
all-negative realization; it was not used to retune the frozen rule.

C-support requires >=50% of C's posterior within 0.15 times the valid-grid
diagonal of A's posterior mean. It is a coarse, population-support criterion,
not known memory content or known replay truth.

## Why the subset is uninformative

At fixed last-complete 20-ms endpoints, mean A spikes per event were
0.89, 0.64 and 0.98 in the three recordings; mean C spikes were
0.89, 0.41 and 0.82. Posterior entropies were consequently close to one.

Known-position mean errors remained large, despite modest selection improvements:

| Known truth | A all -> selected, cm | C all -> selected, cm |
|---|---:|---:|
| Held-out RUN | 30.86 -> 29.10 | 30.58 -> 29.77 |
| Matched-map simulation | 30.17 -> 27.88 | 29.62 -> 28.42 |
| Map-drift/gain simulation | 29.93 -> 28.24 | 28.69 -> 27.63 |

Agreement, support and truth error are not interchangeable. Diffuse posteriors
can have similar means with little actual spatial information. Making one
population's readout sharper can increase its disagreement with another
uninformative population. This experiment neither rescues the remedy nor proves
the absence of replay in AutoPI.

## Verification and traceability

- 45 relevant tests passed; all new files and the shared adapter change pass Ruff.
- Independent dense implementation reconstructed 114,663 real/RUN/simulation
  readout rows, 5,692 fixed endpoints and 11 recordings across PF and AutoPI.
- Raw cached timestamp counts, first-half maps, held-out RUN truth coordinates,
  partition identities, whole-universe count-matched simulations, posterior
  probabilities and feature/outcome columns were checked.
- All 31,290 external prediction rows, losses and fixed-half decisions were
  independently checked. C perturbation tests leave predictions and selections
  unchanged; the pooled baseline observes the same A+B neurons as the full model.
- A separate reporter reconstructed session/draw/animal aggregates and each
  numeric gate; stale or corrupted tables fail its tests.
- Encoding maps are shared source caches. This does not independently validate
  raw spike sorting or biological replay ground truth.

Frozen PF model SHA256:
`51e83bb18d8785bff7979ed1299fa9da0b9980d76bf51cc5befeb6376083052e`.
Producer/validator: d41d2950; AutoPI cache: df866ab7; dense auditor: 040fe522;
aggregate/gate reporter: 771c847a. Exact commands, input hashes and clocks are
stored in each artifact manifest.

Server artifacts:
`/mnt/seagate10tb/florianpfaff/three-population-content-20260914/`

Native extraction and original failure logs:
`/home/florianpfaff/autopi-three-population-20260914/` on gpuserver4090.

Compact local report, figures, audit and frozen model:
`/mnt/c/Users/emper/Documents/codex/2026-09-14/three-population-content-validation/`

## Next scientific constraint

A further independent test must establish adequate per-population spatial
observability before relying on content agreement as validation. Do not turn
all-negative support predictions, sparse endpoint availability or a favorable
secondary split into an apparent solution. A different dataset or observation
design would be a new frozen experiment, not a successful reinterpretation of
this failed one.
