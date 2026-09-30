# Shared region-blind regional-content simulation bank

Frozen before new-bank generation, 2026-09-16. Development bank, not a biological
result or completed prevalence/false-flag validation. Foundation for the terminal
estimand, held-out-cell participation and inferred-dynamics experiments.

## Cohort and encoding

Use the unchanged selection-matched-regional-pf-v3-20260915 source: eight PF
sessions, four rats, first-half RUN maps, exact original candidate spike times,
original native MUA detection and complete-window endpoint convention. All 1600
candidate identities are retained in a cohort manifest. The primary common
cohort has 1139 candidates with at least 100 ms inside the original event;
461 shorter events are explicitly unavailable, not padded or relabeled.

All raw cells participate in spike generation and original active-cell gating.
Preserve total counts and times exactly. Store each generated spike's cell
identity, latent path and all region labels, so later analyses can use disjoint
cells/tetrodes without regenerating data. Confirmed population definitions,
including their overlap and legacy eligibility limitation, stay unchanged.

## Empirical anchors, not biological jump truth

Pin and hash the existing full-population first_order_imm_switch_location_full
transition table and shard manifests. It covers 108 selected clean-IMM events,
not all MUA candidates. Its selection and its smoothed 4-ms decoder resolution
are limitations of the anchor, not claims about true replay dynamics.
The last transition timestamp can be shortened by the source's partial final
bin. Recover displacement using the configured 4-ms divisor used in that
source's speed column; use actual transition timestamps for episode intervals.
No irregular internal time steps are accepted.

Per session, multiply posterior-mean step speeds by each event's fitted bin
interval. Define a resolved large-step episode as a contiguous run of decoded
steps >=20 cm. Record raw large-step amplitudes/inter-step times AND episode
statistics. Primary jump amplitude pool: maximum decoded step per episode.
Primary renewal interval pool: differences between episode onset times within
the same event; never cross event boundaries. Record counts, medians and full
empirical pools. The episode convention avoids treating every consecutive
4-ms transition as a separately observed neural jump. It is still a heuristic.

Moving speed anchor: session median speed for steps with displacement >=0.1 cm
and <20 cm (exclude numerical-zero and large-step bins). Freeze scale factors
0.5, 1 and 2. Jump amplitude uses the same three scale factors. Report requested
and realized distances and boundary-induced truncation; intervals remain the
session's empirical episode distribution. Do not call these empirical latent
jump measurements. Missing within-session intervals must fail, not silently
borrow a different rat's dynamics.

## Geometry-blind dynamics

Generator functions accept geometry, empirical parameter pools and RNG only;
they have no Home center, region mask, desired label or cell tuning as input.
Generate backwards from a uniformly sampled valid-grid terminal location. This
is a defined, region-blind trajectory law, not a fitted model of replay starts.

Stationary: fixed location. Moving: occupied-grid geodesic motion at the fixed
scaled speed, selecting successive destinations without reference to Home.
Jumping: piecewise-constant locations; candidate next locations chosen by radial
distance from the empirical scaled amplitude, with uniform choice among equally
supported candidates. If the requested radius has no grid support, use nearest
attainable radii and record mismatch. No forced region crossing.

Renewal jump timing uses a length-biased sampled interval and uniform phase
at the terminal point, then independent empirical intervals backwards. Jumps
can occur anywhere within the segment or before it, not always at -5 ms.
Seven strata: stationary; moving at 0.5/1/2; jumping at 0.5/1/2.

## Estimand and sampling

For each Delta=20/40/60/100 ms, label after generation by total represented time
inside the geometric 20-cm Home disc, using exact segment/circle intersections
for moving paths and exact interval lengths for jumps. Primary positive label:
at least 20 ms cumulative represented time in Home (not necessarily contiguous).
Also save occupancy fraction, longest contiguous Home visit, endpoint label,
origin label, number/type of crossings and a late-crossing-within-5-ms indicator.

Accept/reject geometry within each dynamics stratum to realize the desired
segment label, with no special region-dependent generator branches. Endpoint
prevalence is not constrained and is not substituted for segment prevalence.
Generate the earlier portion of the same region-blind path only after its
terminal 100-ms geometry passes the label condition; this is an efficiency
device for a backwards Markov/renewal generator, not a different prefix model.

Choose exact rounded label quotas at prevalence .05/.15/.30/.50; calibration
uses .50. If original active-cell support fails after allocating spikes, reject
the joint path/spike sample and draw again, preserving the desired label. Thus
the retained bank is conditional on both label and original support. Log all
rejection counts and impose a finite attempt cap; fail rather than weakening
the detector. The native amplitude/duration/boundary detector remains unchanged
because timestamps and the surrounding whole-population count trace are fixed.

Delta-conditioned banks are separate: one path may carry labels for all four
lengths, but its sampling prevalence applies ONLY to its conditioning Delta.
Calibration, null and validation seeds are disjoint and recorded. Initial
development size: four independent replicas per phase/target prevalence.
This is sufficient for bank sanity checks, NOT to establish 90% coverage or
a 5% false-flag rate. Increase replicas later with frozen code/parameters.

## Sanity gates before downstream use

Check geometry-blindness in tests, both labels within every stratum/Delta,
exact quotas, exact count/time preservation, native active support, native
detector equivalence on representative reconstructed sessions, truth by an
independent numerical occupancy audit, and independent phase seeds.

Report occupancy histograms conditional on each label, BEFORE and after spike
support selection, by generator/scale/Delta. Stationary positive occupancy=1
is structurally required. At Delta=20, positive occupancy=1 is also required
by the 20-ms threshold for every generator. These are explicit exceptions, not
defects to tune away. Flag near-degenerate conditional occupancy in other cells;
do not add Home-aware path rules to manufacture a wider distribution.

Report preconditioning region-crossing frequencies and proposal efficiency;
conditioning on rare labels will deliberately alter those frequencies in the
retained bank. A uniform-terminal-location reference is generator-specific,
not a universal biological chance level. Store the rejected-proposal geometry
summary separately from the selected label-balanced distributions.

No real endpoint calibration, Delta* selection, absolute nonspatial-Hz claim,
IMM validation or hc-11 transfer follows from bank QC alone. At fixed population
totals, common gain and additive Hz are identifiable only up to common scale.
