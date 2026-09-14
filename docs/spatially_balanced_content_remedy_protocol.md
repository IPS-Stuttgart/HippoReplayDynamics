# Follow-up: spatially balanced neural sampling

Status: frozen design before new partition outcomes. This follows a FAILED
PF-to-Tanni agreement-selection experiment; no Tanni refit is confirmatory.

## Design

Primary development data: eight PF recordings. Independent evaluation: the five
downloaded Blackstad--Moser open-field recordings, native position and MClust
units, retaining the 741 previously frozen immobile high-MUA candidates. Existing
continuity labels do not select candidates or optimize the remedy. These events
are not biological replay ground truth. At least four animals and 80% of source
candidate endpoints must remain technically measurable for external evaluation;
all excluded recordings/events and reasons must remain visible.

The remedy acts on neuron sampling, not on decoded replay outcomes. Within a
recording retain all eligible neurons as the sampling universe, split into two
equal disjoint halves and drop at most one seeded odd neuron. Both random and
balanced conditions use the same universe and exactly the same count per half.
At least five cells per half are required; sparse recordings are kept visible.
All events are decoded: no event-abstention improvement is possible.

Encoding uses first-half RUN maps and the common grid. Cell eligibility is frozen
with the existing common RUN pipeline, explicitly conditional on full-RUN QC.
The third temporal quarter is development for population selection; the fourth
is held-out known-position validation. Use the existing 20 ms RUN-window rules
from content_stability_protocol.md, with >=100 windows in each block. No
replacement population or threshold change after fourth-quarter results.

For each of three seeds construct one random complementary half-pair. The
balanced condition pairs neurons using first-half tuning-curve shape and peak
position, then assigns one neuron from each pair to each half. Generate 128 seeded
assignments and rank by the squared difference between normalized summed rate
profiles over the same nine fixed spatial tiles. Evaluate the best eight on the
third-quarter RUN windows; choose the pair minimizing the worse population's
mean true-position error, plus the mean absolute difference of their per-tile
mean errors over jointly supported tiles (>=10 windows per tile). No replay
values enter either ranking. Primary seed0, other seeds sensitivity only.

## Readouts

Reuse fixed candidate endpoints and unchanged 20 ms flat-prior Poisson decoding.
Compare balanced with random samples on regional posterior TV, posterior-mean
endpoint separation, each population's width and entropy, MAP tile agreement,
and true-position error on fourth-quarter RUN. Known-content simulations generate
ONE whole-universe spike-count vector at a common true endpoint and repartition
the SAME observations under random/balanced cell assignments. Include matched
first-half maps and second-half-map/lognormal-gain mismatch, two independent draws
per event. All spike counts, rates and timestamps must be auditable.

Average A/B when reporting localization error and entropy: population naming is
arbitrary here, unlike the previous A-only diagnostic. Also report each side so a
large deterioration in one half cannot be hidden. Average events within session,
sessions within animal, then equally weight animals. Never count seeds or draws
as additional animals.

## External pass criteria

- Input, count, disjointness, frozen-selection and reconstruction checks pass.
- Primary regional posterior TV reduces by >=10% relative to random sampling.
- Primary mean endpoint separation reduces by >=10%.
- Both improvements occur in >=4 external animals; descriptive animal-bootstrap
  intervals for both mean reductions have lower bounds above zero.
- Mean A/B entropy does not increase; neither side's entropy increases by >0.01.
- Mean A/B true-position error does not worsen on observed held-out RUN and on
  either conditional generator. Neither side worsens by >2 cm.
- At least four external animals/80% source endpoints are represented. No event
  filtering, posterior flattening, uniform replacement or trajectory prior.

PF development is not external confirmation. A failed external criterion is a
failure of the complete remedy, not a reason to optimize against Blackstad data.
Passing would support robustness of a sampling-aware measurement procedure, not
correct goal inference, biological replay prevalence or novelty over all prior
work on stratified sampling. The existing Tanni failure must remain in the report.
