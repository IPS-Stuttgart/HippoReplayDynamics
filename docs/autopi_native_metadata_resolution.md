# AutoPI native metadata resolution before decoder outcomes

The first native extraction attempted all 39 recording directories but emitted
no caches: the reader required equality between multiple author exports and
between the generic cluster_info `sh` and explicit shank_neuron assignments.
Those assumptions did not match the supplied release. No decoder outcomes were
generated and no selection thresholds are changed.

Observed metadata issues:
- cluster_info uses `id` instead of `cluster_id` in one recording.
- Older cluster_group exports omit some author-good cluster_info entries.
- The generic cluster_info `sh` is constant zero in inspected recordings while
  the explicit shank_neuron table has multiple one-based physical assignments.
- Some explicit assignments exceed the number of rows in `.desel`; those cells
  cannot be assigned a CA1 region without inventing missing metadata.
- Several first circ80 epochs are not immediately followed by rest. These remain
  excluded under the frozen epoch rule, not replaced by a later favorable epoch.

Resolution, fixed before decoding:
1. Use the detailed cluster_info author `group == good` labels; preserve raw IDs.
   Record missing legacy cluster_group entries. Exclude direct curation conflicts.
2. Use the explicit shank_neuron assignment, not the uninformative generic `sh`.
3. Require an explicit assignment within `.desel` and region CA1. Retain excluded
   unit rows with `unknown_unmapped_exclude`; do not guess their brain region.
4. Unknown physiological cell types remain unknown; pyramidal *layer* is not a
   declaration that a unit is excitatory. All included units still face the same
   frozen RUN rate, stability and coverage checks.

This is an input-schema correction, not a relaxation of neuron-count, predictive
performance, candidate-support, or independent-validation gates. Original failed
extraction logs and catalog are retained; corrected caches use a new directory.
