# Fixed-time count-conditioned content remedy: frozen test

2026-09-14, following the failed fixed-time rate-uncertainty test. This is a
prospective test of an EXISTING likelihood, not a claim to invent multinomial
decoding or to establish a new biological replay mechanism.

## Hypothesis and primary intervention

RUN-to-replay population-rate changes may make the total-count component of
plug-in Poisson decoding misleading. Decompose the likelihood into cell identity
given total count and the probability of that total count. Remove only the
latter. At each independent20ms endpoint and each frozen population:

- Baseline: logL=sum(n_i log(lambda_i)) - .02 sum(lambda_i).
- Primary `conditional_multinomial`: logL=sum(n_i log(lambda_i/sum(lambda))).
- `poisson_entropy_matched`: adjust the baseline likelihood temperature to
  match the primary posterior entropy, per event/population, using only that
  population's counts/rates. Frozen search logT[-20,20],70bisections,tolerance1e-7.

All priors stay spatially uniform. No temporal model, path smoothing, shifted
endpoint, removed event or changed rate map/cell mask. The zero-count conditional
posterior is uniform and contains NO spatial observation information. Lower
disagreement caused solely by this flattening cannot count as success.

Conditioning discards potentially useful rate information. It may improve
robustness to global activity gain while making matched-Poisson decoding worse.
The simulations below deliberately test both possibilities. Per-cell gains or
different latent content across cells are not repaired by common-gain invariance.

## Inputs and controls

Reuse the independently audited edge-support inputs: PF/hc11 each8sessions,
4rats,1600frozen candidates,3equal half partitions; split0 primary. Original
first-half RUN mean maps/unit masks and original final20ms endpoints unchanged.
hc11 has informed earlier methodological failures; this is a SCREEN, not a
pristine independent confirmation cohort.

Keep six unchanged sources: real,run_q4,sim_stationary,sim_moving,
sim_moving_gain,sim_late_jump. The four existing simulators condition on observed
whole-population totals. A conditional decoder matches an aspect of their
observation model, so they cannot alone validate the remedy.

Add two unconditional, known-location sources, using the SAME frozen stationary
latent endpoints and full eligible-cell universe from sim_stationary:
`sim_poisson_stationary` and `sim_poisson_stationary_gain20`.
The true position is the frozen occupied grid bin; mean20ms counts are
.02*lambda_i(x) times1 or20 respectively. Independent Poisson draws once per
whole population, then apply all frozen partitions; never redraw per decoder.
Seed SHA256 is frozen from(20260914,count_conditioned_poisson,dataset,session,
source,event_id). Both zero-spike draws and all other draws remain in evaluation.
These validate likelihood consequences, not biological SWR gain or replay truth.

## Readouts and a prospective diagnostic

Keep A/B means, regional3x3 probabilities/TV, entropy, RMS width, spike/active
counts, known-position errors, regional Brier and NLL. Same truth region rule and
float64 geometry arithmetic as the audited previous test. Store posterior arrays
and the exact same original clocks. Native counts are independently recounted.

`A_total_count_reliance_tv` is the regionalTV between A's Poisson and conditional
posteriors. It uses no B observations. Report within-session Spearman association
with independent B true-position error (known controls) and real A/B disagreement.
This is only an unadjusted diagnostic screen, not a trained selector, not a
validated certificate, and not a post-hoc threshold for event deletion.

## Frozen screen gates

Use the previous primary remedy requirements unchanged except adding both
unconditional Poisson controls to the mandatory known-position sources:

1. Technical/source audits complete, all1600real events and entropy controls.
2. hc11 real separation AND regionalTV reduced at least10% versus Poisson,
   each direction positive in at least3/4rats.
3. Better than entropy-matched Poisson on BOTH real disagreement metrics.
4. q4 RUN mean A/B error and Brier not increased.
5. EACH known-position source and EACH population: upper one-sided descriptive
   95%rat-bootstrap bound on error increase<=2cm.
6. Mean A/B regional Brier not worsened in ANY simulated source, including
   matched-Poisson/no-gain and gain20.

Average events within session, sessions within rat, rats equally. Four-rat
bootstrap intervals are descriptive; two extra splits cannot rescue a primary
failure. Paired before/after rows always use the same event and population.

Passing this screen is not enough to finish the goal. Freeze the estimator and
test disjoint next200 SHA-ranked hc11 candidates/session, and the ORIGINAL
matched-population PF Home/accepted-endpoint contrasts with unchanged populations,
including nonuniform known-content controls. The goal is not to reduce raw
random-half disagreement by hiding genuine regional content. If the screen
fails, preserve it; do not tune rate gains, entropy or support on hc11.

## Execution

Commit before outcome inspection; scientific jobs detached on gpuserver6000.
Pin all source and imported likelihood/metric/protocol hashes. Independent audit
rebuilds native counts, unconditional expected counts/draws, likelihoods through
SciPy PMFs, entropy matching and posterior metrics. Non-rescoring report includes
all sources, per-session/rat results, gates and plots. No raw data in paper repo.
