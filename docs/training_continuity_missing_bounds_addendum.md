# Missing Arena Bounds: Technical Addendum

The first run, `training-continuity-prediction-all9225-20260910`, terminated
with status `failed`: all eight PF source caches record their arena bounds as
NaN, an explicitly unavailable metadata value. All 25 Tanni sessions completed,
but no stratified predictive results were calculated or interpreted.

Correction: preserve the primary valid-bin mask. When all arena bounds are
unavailable, retain a tagged no-op clipping row so the table remains complete,
but set `arena_bounds_available=false` and `outside_arena_centres=null` in the
manifest. Such rows are not an independent clipping sensitivity. Do not infer
arena walls from candidate activity, drop PF, or impute physical boundaries.
Partially missing or invalid bounds still fail. Tanni clipping is unchanged.

The primary classifier, neuron partitions, endpoints, thresholds and bootstrap
rules are unchanged. The initial failed artifact remains intact; the corrected
run uses a new `-v2` output directory and a new clean commit. The independent
auditor checks the availability flags and both known/unavailable support cases.
