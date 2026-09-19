# Source-pinned first-pair RUN diagnostic

Frozen before examining reconstructed RUN metrics or rest content.

Question: was RAT2_SESS1 excluded because a four-epoch release was treated as
four independent contexts, rather than because its first two epochs cannot be
decoded? This is prerequisite evidence for the sampling/content study, not a
new replay result or a threshold-sensitivity search.

## Source Evidence

- Dataset: Dryad 10.5061/dryad.ksn02v76h, release 238435, pinned spike/position
  hashes in the script. The four epochs are chronologically disjoint.
- Original study code: bendor-lab/Elife_Tirole_Huelin_Gorriz_2022,
  commit 44ecf4275c2a7ca33dda6b67f11b4e854c3123e9, matching the paper's
  Software Heritage archive. This is not the later replay cross-validation repo.
- Its batch_remapping_pipeline.m reads a re-exposure flag from folders(:,2).
  That session lookup is not supplied in the pinned code or data release.
- Analysis/sort_replay_events.m lines 63-67 end POST at epoch 3 if the flag is set.
  The first-pair interval for RAT2_SESS1 is 125.379 min, consistent with the
  paper Table 1 reporting 125 min. This is corroboration, not proof of the flag.
- Generic code defaults pair epochs 1/3 and 2/4. We do not use that default to
  label/merge later exposures, nor to claim an additional independent session.

## Frozen Diagnostic

Use only source epochs 1 and 2. Retain the original regular clock, supplied
speed, coordinates, spike identities, and samples strictly before epoch 3.
Remove later spikes and both later position arrays. Keep thresholds unchanged:
10 cm maps; 5 < speed < 50 cm/s; original unit QC; >=20 common units;
>=80% occupied bins; five 10 s blocked folds plus 1 s guards; 250 ms decoding;
>=20 scored windows, context accuracy >=0.8, median conditional-position error
<=35 cm and occupancy >=0.8 for each of ten track/fold rows.

Reuse the preflight functions; fit maps and select units within each training
fold. Preserve all ten results even on failure. Do not try pairings 1/3, 2/4,
or 3/4 to search for a passing result. Do not inspect sleep-event outcomes.

The reconstruction does not mutate raw files, the original preflight, banks,
or the primary cohort. Passing RUN is necessary but insufficient for promotion.
Source clarification and LFP/position-clock checks remain separate prerequisites.
Duplicated RAT1_SESS2/RAT4_SESS1 data remain quarantined; no clock offset or
missing animal/session identity is guessed. Failure does not justify relaxing
the existing gates.

Sources: https://elifesciences.org/articles/79031 and the pinned original-study
repository above. Daniel Bush is the collaborator, not an author of this study.
