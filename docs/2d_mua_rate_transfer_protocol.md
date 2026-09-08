# Frozen 2D MUA observation-transfer diagnostic

2026-09-08. Exploratory follow-up to a failed stronger baseline, not a new
confirmatory cohort or a procedure chosen to make IMM positive.

## Question and novelty boundary

Does unequal observation-model adaptation explain why the RUN-trained Tanni
spatial model fails to beat a nonspatial predictor trained on other MUA events?
Test both Pfeiffer/Foster and Tanni in the same frozen all-candidate cohort.
Improved likelihood alone is not replay: require additional independent-position,
nonspatial, map and time-order comparisons. Do not change thresholds, dynamics,
candidate selection, map support, neural splits or bin widths.

State-dependent firing and replay rate codes are not new. Relevant precedent:
[Tirole et al. 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC9489210/) and
[Sartorato et al. 2026](https://www.nature.com/articles/s41467-026-75492-w).
The same simple rate recalibration failed to establish temporal replication
in our hc-11 diagnostic. This test is specifically the unresolved unequal
training-exposure comparison in the common 2D cohort, not a new gain mechanism.

## Frozen sources

Common parent prediction manifest SHA256:
`d16542d0bf935929a05a335f84cd2060167c76dd8e255378beb202fb84e97386`.
Order-map parent manifest SHA256:
`40f4b123142ddd47b22e7db1311602b849a4d996d943076d26c07e0485c524ae`.
Use all 9,225 original eligible MUA events: PF 4,001/eight sessions/four animals;
Tanni 5,224/25 sessions/five animals. These are not trajectory-selected events.
Use the parent's audited caches, 20 ms nonoverlapping counts including partial
last bins, 8 cm RUN maps, five fixed 70/30 splits and shared map permutation.

## Matched cross-event calibration

Reuse the parent's five chronological candidate folds and one-second exclusion
guard. Let c_i be pooled calibration-event counts, and p0_i the normalized
spatial-mean RUN rate. Define p_i=(c_i+alpha*p0_i)/(sum(c)+alpha), g_i=p_i/p0_i.
Scale every position of cell i's RUN rate map by g_i. This preserves each cell's
rate-map shape; cross-cell likelihoods and decoded locations can change.
The spatial-mean raw-rate composition then matches p exactly. This is not a
claim that a mixture of count-conditioned likelihoods has identical marginals.

Primary alpha=100 population pseudospikes, identical to the existing event-global
baseline. Alpha=1000 is a declared stronger-shrinkage sensitivity, with its own
matched global baseline; it cannot replace a failed primary. Reuse the existing
hc-11 calibration helper, without fitting alpha to target scores.

Held-out cells may supply spikes in other calibration events, never the scored
fold or its guarded neighbors. No target held-out spike updates the gains,
latent path or mode weights. This is retrospective cross-validation, not future
prediction; candidate detection used all cells in the fixed parent. Gains can
reflect represented-location composition as well as physiology: do not call
them measured biological gain or remapping.

## Models and factorial

For both alpha values score independent positions, static location, diffusion
and first-order IMM, with real and jointly position-permuted maps. Use unchanged
parent dynamics (diffusion sigma 60 cm/sqrt(s), stationary sigma 2 cm, IMM mode
time constant 60 ms). Scores are proper frozen-training-posterior marginal
held-out multinomial log scores conditioned on each held-out bin's count.
No powered likelihood or held-out path inference.

For alpha=100, use all K=20 whole-population-bin permutations from the frozen
parent seed. The same permutation acts on training/held-out cells and both
maps; partial-bin duration travels with its snapshot. Identity/repeated orders
remain present. Independent-position and static scores must be order-invariant;
the shared map permutation must preserve these scores and global predictions.
Reuse hash-verified unadapted scores and factorial contrasts. Do not conflate
adaptation gain, order dependence and map correctness.

## Primary outcomes and weighting

Within each event/split compute:
1. Adapted-real IMM minus unadapted-real IMM.
2. Adapted-real IMM minus independent positions.
3. Adapted-real IMM minus matched event-global.
4. Adapted-real IMM minus permuted-map IMM.
5. Adapted-real IMM original minus mean K shuffled scores.
6. Order-by-map interaction: (real original-real shuffled) minus
   (wrong original-wrong shuffled).

Report diffusion analogues, static and IID comparisons, changes from the
unadapted order/map controls, and alpha=1000 sensitivities separately. A positive
recalibration result requires all six primary means and CI lower bounds >0,
with every animal positive, within the dataset being evaluated. No dataset-wide
or replication claim from a pooled PF+Tanni statistic. A failure remains visible.

Differences within split first, then median across five splits per event.
Average events within session, sessions within animal, animals equally within
dataset. Reuse the parent's 5,000-draw animal/session/event bootstrap. Per-spike
normalization is sensitivity; zero-held-out-spike ratios remain missing. Four
or five animals limit generality, and these conditional intervals do not absorb
all model/experiment-selection uncertainty. No favorable subset selection.

## Decision boundary

If calibration improves global fit but not the full spatial/order set, close
the simple rate-transfer explanation. If it passes, treat it as an exploratory
observation-transfer result requiring new-event confirmation and stronger
nonspatial temporal controls, not a novel circuit mechanism. This test alone
cannot complete the high-importance-paper objective or validate physical speed.
