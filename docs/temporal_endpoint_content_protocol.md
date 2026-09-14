# Fixed-endpoint temporal-context remedy: development protocol

Frozen before this experiment's decoding. Earlier RUN-risk, uncertainty,
count-conditioning and edge-trimming failures remain failures. This is a new
mechanism, not a refit of a failed selector or a claim that HMM filtering is new.

## Inputs and question

Use the independently audited edge-support benchmark: eight PF recordings/four
rats, 200 hash-selected candidates per recording (1,600 of 4,069 source events),
first-half-only RUN encoder and unit QC, three frozen equal disjoint populations.
Split0 is primary, split1/2 sensitivity. No new event selection. Fixed 8cm
supported-state grid. Reuse its held-out fourth-quarter RUN and four known-path
simulations (stationary, moving, moving with per-cell gains, late distant jump).
Simulations preserve original whole-population 5ms totals, not actual active
cells/noise correlations; moving paths use 1000cm/s on an occupied-grid graph.
They are falsification tools, not replay truth.

Evaluate whether prior observations reduce A/B endpoint-content instability
WITHOUT corrupting known physical location or hiding real late jumps. Each
population uses only its own spikes. No A/B cross-conditioning or consensus.
Do not infer unknown biological truth from their agreement.

## Fixed methods

Keep the last complete 20ms window and its time-averaged truth exactly unchanged.
Group the preceding 5ms bins into disjoint 20ms windows, aligned from the right,
up to 200ms total context. Ignore only leftover leading bins. Never count the
same spike twice as independent evidence. Do not use future observations or
move the endpoint to a more active part of the event.

Observation likelihood is the existing Poisson rate model with 20ms exposure,
including silence; no gain fitting. Prior at context start is uniform.

- independent: flat-prior decoding of the terminal20ms only.
- diffusion_reset (PRIMARY): each step's transition is 75% spatial Gaussian
  plus 25% uniform reset over all supported states. Gaussian sigma20cm/20ms,
  truncated at4sigma and column-normalized using the existing state-space API.
  A reset permits a distant jump; it does not guarantee that sparse jumps can
  be identified. This is not a claim that replay speed is20cm/20ms.
- diffusion_no_reset: same Gaussian without resets, an over-smoothing control.
- pooled_static: all context spikes under one fixed location, a static control.
- entropy_matched: temper the terminal independent likelihood to match the
  PRIMARY endpoint entropy separately in each population. Temperature bracket
  exp(-20)..exp(20), existing70-step bisection. Missing match is visible and
  blocks the primary entropy-control gate, not silently discarded.

Use existing exact first-order recursion. Its final smoothed state equals its
final filtered state, so no future evidence enters this endpoint readout.
No parameter tuning, replay evidence comparison or label selection in this run.
No alternative method can replace the frozen primary if it fails.

## Measurements and advancement gates

Every method retains every input endpoint, even zero-spike bins. Report A/B
entropy, width, means, 3x3 regional probabilities, separation and regional TV.
Known-truth sources also report each side's physical error, regional Brier score
and regional negative log score; event truth refers to the unchanged last20ms.
Explicit late-jump readouts retain the before-jump and terminal known locations.

Compute paired changes within event, then session and animal with equal animal
weight. Report means, p90 truth-error tails, each rat, primary versus secondary
splits, and descriptive5,000 rat-bootstrap intervals. Four rats do not establish
population generality. Never choose a rat/session after seeing outcomes.

PRIMARY advances only if complete/audited, real separation AND regional TV each
improve >=10%, each improves in >=3/4 rats and its descriptive bootstrap interval
excludes0. Neither side's real entropy may increase. It must outperform the
entropy-matched control by >=5% on both real agreement metrics. Each side's mean
AND p90 original-time physical error and mean regional Brier must not worsen
in RUN or ANY simulation (numerical tolerance1e-8). All entropy matches required.
The strong jump gate is intentional: a gain bought by erasing late content is
not a remedy for fixed-endpoint content.

If PF fails, record it and do NOT transfer this fitted/frozen method as a success.
If it passes, apply the unchanged protocol to the existing eight-session hc11
benchmark, before using hc11 outcomes to change any parameter. hc11 has been
used for other controls; it is independent recordings, not a pristine blind
dataset. External passage is necessary, but this benchmark still does not
establish a remedy for the original targeted Home-coverage contrast: apply it
there too without changing timestamps and report whether its +8.58pp contrast
shrinks without worsening local known RUN recovery. No biological claims.

## Reproducibility

All calculations run on gpuserver6000 in a detached systemd user service.
Freeze code/protocol/input hashes before run; preserve input files. Store every
endpoint posterior, source identity and metadata. Independent verifier uses
direct dense transitions/log-domain forward filtering, native spike recounting
and recomputed metrics/aggregations. Code tests include tiny-grid path enumeration,
reset/independence limits, a late jump, spike multiplicity and endpoint invariance.
The goal is not complete unless reduction and truth safeguards survive independent
data and the original content question, not merely a green execution manifest.
