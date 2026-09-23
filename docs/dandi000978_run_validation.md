# DANDI000978 regional RUN readout feasibility

Protocol frozen before the first regional validation run (2026-09-23).
This is not replay scoring or a confirmatory biological experiment.

## Anatomical eligibility

Use the pinned 0.240511.0307 release. A source clarification supports interpreting
JS14 unit references as tetrode IDs; explicit source crosswalks verify the two
ZT2 files. Keep these evidence levels distinct. The other six animals remain
anatomically unresolved and are not selected/excluded by decoder performance.
ZT2 is one animal; its two files are separate temporal blocks, not independent
animals. Match its source units by tetrode plus within-tetrode cluster ID, never
by local NWB row/ID. Crosswalk identity does not prove sorting stability.
The private source archive is a required external input and is not redistributed.

## Fixed readouts

- Leave one whole RUN epoch out within each file. No replay spikes, resting
  epochs, source SWR intervals or NREM intervals enter fitting or evaluation.
- Use non-overlapping 250 ms windows entirely inside native trial intervals,
  at least five finite position frames, no frame gap above 100 ms, and all
  included frames above 5 cm/s. Record every trial's eligible-bin count.
- All sorted units with at least 20 training RUN spikes are eligible; the source
  supplies no reliable pyramidal/interneuron label, so none is invented.
- Fit a flat-prior 2D Poisson position readout on an 8 cm grid, retaining only
  training bins with at least 0.5 s occupancy. Gaussian spatial smoothing has
  sigma 8 cm, cutoff 24 cm, with a 0.25 s global-rate pseudocount. Also decode
  conditional on each window's total population count (multinomial composition).
  There is no temporal transition prior. This is a 2D W-track position readout,
  not graph-linearized replay decoding; posterior means may lie off-track.
- Report MAP and posterior-mean Euclidean errors, train-support coverage, spike
  support, a training-position-median baseline, and population-vector circular
  shifts within held-out trials (99 draws; offsets at least 20% of trial length).
  Trials have equal weight within each held-out epoch; files then have equal
  weight within animal. Nulls are diagnostics, not a many-rat inference test.
- Directed canonical routes are defined independently by native start/end wells:
  center-to-left, center-to-right, left-to-center, right-to-center. Other routes
  remain in spatial QC but are explicitly excluded from the four-route readout.
  No exclusion based on correct/incorrect labels. Report inconsistencies with
  the native inbound/outbound field instead of overriding endpoint labels.
- Route readout fits rates from training trials and evaluates whole held-out
  trials, with uniform class priors and at least three training trials per route.
  Report Poisson, population-count-conditioned composition, and count-only
  readouts. Use 99 training-label shuffles within RUN epoch, refitting each time;
  do not permute individual time bins. Report balanced accuracy and held-out
  log loss. Four-route scores require all four classes in the test epoch.

## Gates and limitations

Technical completion and content evidence are separate. Failures/unsupported
folds remain in denominators. Per-file content support is descriptive: at least
half the held-out epochs have composition-route balanced accuracy above their
null p95, and median gain over the null is positive. Both areas passing does not
prove cross-area replay, abstract route coding, or independent cortical ground
truth. Route information can reflect current position, direction and behavior.
Count conditioning removes a total-rate-only classifier but not these covariates.
Only two animals currently have supported mappings. Do not use their performance
to infer anatomy in the rest. No data-dependent threshold tuning in this run.

Outputs contain fold/trial predictions, eligibility, anatomy provenance, nulls,
file/animal summaries and a run manifest with a clean producer commit and hashes.
Raw files are read-only; acquisition hashes are reused with size checks, explicitly
not represented as a fresh 323 GB rehash. Long runs execute under server tmux.
