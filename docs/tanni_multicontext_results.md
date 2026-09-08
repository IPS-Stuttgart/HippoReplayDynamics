# Tanni cross-context prediction: completed diagnostic

2026-09-08. Technical validation passes. The prespecified all-candidate
cross-context spatial-content rule does not pass. No new high-importance
biological result, remote-replay prevalence, or temporal replay claim.

## Why this was tested

The current-arena Tanni model did not reliably beat a nonspatial other-event
activity baseline. An alternative explanation was that candidate events
reactivate another familiar arena. The five recordings per animal share
jointly sorted unit IDs, making a matched cross-arena held-out-neuron test
possible without introducing an IMM or momentum prior.

Remote replay is already established; this was an exploratory dissociation of
current-map mismatch, contextual rate composition and spatial information,
not an attempt to claim remote replay as a new phenomenon.

## Frozen design and safeguards

- All 5,224 previously selected high-MUA windows, 25 recordings/five animals.
- Four familiar physical contexts A/B/C/D; A occurs twice. Equal physical
  context priors, with each A template assigned half of A's mass.
- Common neuron IDs and first-half RUN-only map-derived unit QC. Retained
  cells: R2470 100, R2474 158, R2478 163, R2481 196, R2482 198.
- Maps use only first-half RUN and its occupancy support. Second-half RUN
  validates decoding on 100 nonoverlapping 200 ms windows per recording.
- Five fixed 70/30 neural splits per animal, shared across recordings.
- Infer context and independent-bin location using training cells only.
  Score held-out identities conditional on each bin's held-out total with
  proper multinomial probabilities. No powered target likelihood or
  held-out update of context/location.
- The current-context comparator retains both A templates. The other-event
  baseline uses five chronological candidate folds and a one-second guard.
- Candidate detection/ascertainment is inherited from the all-cell parent.
  This is predictive validation conditional on that frozen candidate set,
  not independently detected held-out events. All arena templates are used
  retrospectively; this is not prospective prediction before arena exposure.
- This shared first-half-trained population differs from the earlier
  current-session/full-RUN cohort; absolute scores must not be compared across
  those experiments as if the encoding population were unchanged.

## Results

| Animal | Held-out RUN context accuracy | RUN spatial-minus-global score |
|---|---:|---:|
| R2470 | 67.30% | +0.942 |
| R2474 | 64.75% | +0.894 |
| R2478 | 65.95% | +1.419 |
| R2481 | 78.00% | +2.227 |
| R2482 | 71.70% | +1.450 |

Chance for four contexts is 25%; the frozen RUN gate required accuracy above
50% and positive current-context spatial prediction in every animal. It
passes. This does not make individual MUA context assignments ground truth.

| Primary MUA contrast | Mean nats/event | Hierarchical 95% CI | Positive animals |
|---|---:|---:|---:|
| Multiple-context spatial minus current-context spatial | +0.561 | [+0.147, +0.990] | 4/5 |
| Multiple-context spatial minus context-specific RUN rates | +3.460 | [+2.498, +4.438] | 5/5 |
| Multiple-context spatial minus other-event rates | -1.392 | [-5.003, +1.695] | 3/5 |

Paired split differences are median-aggregated within event. Events are then
averaged within recording, the two A visits within context, four contexts
within animal, and five animals equally. Intervals resample animals and
events within the fixed recording/context design, 5,000 draws. Five animals
limit generality; maps and cell partitions remain fixed.

The per-spike sensitivity does not rescue the last comparison: -0.026
[-0.203, +0.156], 3/5 positive. Independent animal-only exact-bootstrap
sensitivities preserve the same qualitative interval pattern.

Other-event baseline contrasts by animal are +0.126, -8.094, -2.977, +3.503,
and +0.482 nats/event for R2470/R2474/R2478/R2481/R2482, respectively.
Do not remove the weak animals to obtain a positive aggregate.

## Interpretation and next-action boundary

Allowing another arena modestly improves the current-map prediction, and
spatial maps outperform context-specific RUN-rate composition. However,
allowing other arenas does not establish the required improvement beyond
other MUA activity in the recording. The context-mixture explanation is
therefore insufficient under this frozen all-candidate test.

This is not proof that remote reactivation is absent. State-dependent firing,
encoding transfer, spatial observation-model mismatch and genuine remote
content remain distinguishable possibilities. A high training-only remote
posterior is not itself independent evidence of remote replay. No temporal
order or continuous-trajectory test was performed here.

Do not scale, select a favorable subset, change priors or loosen the gate to
turn this into a biological confirmation. The broader paper search remains
open; this experiment closes one plausible explanation without treating a
software pass as a novel finding. It does not displace the existing
recording-coverage/kinematic-inference methods-paper candidate.

## Verification and provenance

- Server: gpuserver6000. Producer commit:
  `699deb0d6e4ce921e32d6369e82f982d70f61849`, clean at launch.
- Scoring artifact:
  `/mnt/seagate10tb/florianpfaff/tanni-multicontext-prediction-20260908`.
- Manifest SHA256:
  `2338477f0444ea44f70105aaa9dcbd06084848845cef3be8291a777ec63b018b`.
- Auditor/reporter commit: `29dbefa0`.
- All 7,724 candidate/RUN windows and 77,508 bins recounted from cached native
  timestamps; all 38,620 event/split rows, held-out support counts, partitions,
  first-half map-bank construction, calibration exclusions and group means
  checked. Independently written predictive equations check 625 rows, maximum
  absolute difference 2.49e-13.
- Original NWB reopening and RUN-map fitting were not repeated: this audit
  reuses the parent's native/map validation. Hierarchical interval draws are
  not independently reconstructed; exact animal-only bootstrap is a separate
  sensitivity rather than a replacement.
- The first audit stopped on a CSV-parsing cutoff rounding discrepancy of
  4.55e-13 seconds in R2481. The selection used the correct JSON cutoff.
  Audit v2 reads that exact authoritative value; no data, selection,
  parameters or scores changed. The first audit directory is retained.
- 42 focused tests pass; Ruff passes. The non-rescoring report includes the
  failed baseline comparison and the RUN context confusion table.
- Report:
  `/mnt/seagate10tb/florianpfaff/tanni-multicontext-prediction-20260908-report/tanni_multicontext_report.md`.
