# Post-error replay content balance: separately registered v2

V1's decoder-qualified coverage stop remains unchanged. V2 changes the encoding
policy and the primary endpoint as explicitly approved, before joining any replay
content to outcomes. It is discovery on previously inspected data, not independent
replication. Seed 20261001; all numerical choices are in the adjacent v2 JSON.

## First milestone and necessary upper bound

Reuse the complete verified v1 behavioral inventory, not only its nine
decoder-qualified trials. Independently reload recorded position/metadata and
reconstruct visits. Split each metadata-qualified RUN epoch at exactly one and
two thirds of its recorded elapsed time (first/last stored position timestamps).
No compression of tracking gaps, performance-dependent cutoffs, or boundary moves.

The entire preceding incorrect outbound traversal, first subsequent center pause,
and following outbound choice must lie in the final third. In particular, test
the departure from center before the error, not just arrival at the mistaken well.
History is reconstructed across the full supported epoch; gaps reset history.
Use strict speed <4 cm/s and the original first center pause, capped at 10 s.
Missing neural information is never encoded as zero replay.

Before loading marks or fitting encoders, count these trials optimistically as if
every future decoder and sequence prerequisite passed. Require >=5 animals,
>=100 transitions, >=20 corrections and >=20 repetitions; each retained animal
must have both outcomes and >=2 recording days. Failed animals stay in ledgers.
These are engineering floors, not adequate-power guarantees. If this necessary
bound fails, stop without RUN fits, replay extraction, calibration or association.

## Causal readout if the upper bound passes

Only complete same-history well-to-well traversals wholly inside the first third
can supply moving RUN samples and marks for fitting. The middle third supplies
only complete held-out traversals; boundary-crossing traversals are excluded.
No interpolation through missing tracking and no exposure from outside a block.
Freeze the first-third encoder; do not refit on validation or later data. Use
all metadata-verified CA1/CA2/CA3 tetrodes with available four-channel marks,
native amplitudes, 6 cm graph KDE, 24-unit mark KDE, 3 cm bins, 20 ms independent
flat-prior decoding and >4 cm/s RUN samples. Keep silent validation bins.
Middle-third balanced arm accuracy must reach .80 and each arm recall .75.
Shared-stem observations are not assigned to either arm. Repeat the same cohort
floor on actual QC-qualified observations, without outcome-selected replacements.

## Endpoint and inference

Native ripples wholly within valid exposure are candidates. Sequence validation
requires five supported bins, two tetrodes and p<=.05 from 1,000 whole-bin
shuffles, remaximizing over routes each time. Average each validated event's
posterior mass on each unique arm; shared mass remains unassigned. Sum contributions
per pause and divide by valid exposure: C, M; primary balance B=C-M, amount S=C+M.
Zero-event trials remain, with S=B=0; missing inputs and unresolved candidates
have explicit statuses. The baseline includes S and the frozen history/quality
covariates. The alternative adds B. L2=1; training-only scaling; animal-balanced
weights; animal/day intercepts; unseen days have zero day offset.

Hold out entire animal-recording days, keeping repeated-error chains together.
Average score gains within day, animal, then equally across animals. Refit in
2,000 animal/whole-day bootstrap draws, grouping copies of the same source day.
Require positive score gain and balance coefficient with two-sided 95% intervals
excluding zero and positive gains in every leave-one-animal-out analysis.

## Calibration, scope and provenance

Separate development from one frozen 1,000-replicate validation bank per null
scenario in the JSON. Include actual cohort structure, readout/map uncertainty,
selection, aggregation and prediction; require upper 95% false-support bound
<=.075 in each null. Report recovery at odds ratios 1.25/1.5/2/3, interval coverage
and all failures. Unit tests are not that calibration bank. Sensitivities cannot
replace a failed primary. Unimplemented or missing downstream stages fail closed.

Audit/development is capped at five working days, excluding weekends; the manifest
records a persisted UTC deadline. Execute through the detached committed launcher
on gpuserver6000, with atomic checkpoints and per-input hashes. Keep v1/concurrent
studies untouched. Archive compact evidence only, with stages actually completed
distinguished from prospective work. No data search, new dynamics model or email.
# Causal RUN Readout Implementation

After the necessary temporal bound passes, `--stage feasibility` consumes the
hashed v2 audit. Only epochs with final-third eligible transitions need neural
fits; all original epochs remain in the cohort inventory. Training uses complete
first-third traversals, and its position array is physically cropped before any
interpolation or occupancy calculation. Both endpoints of every training and
validation tracking interval must have finite position and RUN speed above
4 cm/s. No interval can bridge a tracking gap above 250 ms. The nonoverlapping
20-ms validation grid is anchored at the middle-third boundary; complete windows
must lie in retained validation traversals. Zero-spike windows remain included.

Unique arms without training or validation support are unavailable, not measured
biological failures. Otherwise balanced accuracy and both recalls are compared
to the frozen thresholds. Complete per-bin validation predictions are retained
on the server for independent confusion-count verification. No prior-session
map or alternate encoder is substituted. A failed actual coverage floor stops
the study before sequence validation, simulation calibration or regression.

`verify_denovellis_post_error_v2.py` separately rebuilds raw behavioral visits,
temporal eligibility, exposure and the necessary coverage bound; its audit does
not depend on the producer's transition/coverage implementations.
