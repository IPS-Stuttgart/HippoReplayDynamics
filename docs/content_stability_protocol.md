# Cross-dataset population-content stability diagnostic

Frozen before generating the new endpoint outcomes. Development: all eight
Pfeiffer/Foster recordings; external test: all 25 Tanni recordings (five animals).
Tanni has been used for other analyses, but none of the endpoint outcomes of this
new experiment will choose its features, coefficients, thresholds or gates.

## Question and scope

Can population A alone predict instability relative to a disjoint population B,
and can a frozen A-only selection policy reduce that instability in Tanni?
This extends the equal-count PF content intervention. It is a conditional
measurement diagnostic, not proof of correct biological replay or a correction
to the original goal-directed replay result. Region labels are fixed 3x3 tiles
of the represented coordinate extent, not author-verified goals or walls.

Use the original all-high-MUA candidate cache, with no sequence selection.
Decode the last complete 20 ms of each candidate's full 5 ms bins. Keep exact
half-open spike counts. Use independent flat-prior Poisson decoding, original
8 cm coordinates, first-half RUN maps, first-half occupancy >=0.05 s.
Original full-RUN unit eligibility is retained and explicitly conditioned on.
It is not a fully blinded recording-level train/test split.

For each recording, split eligible units into two equal, disjoint halves using
three fixed seeded permutations; drop one seeded unit if odd. No split is selected
for good replay or RUN agreement. Split 0 is primary; 1/2 are sensitivity only.
Require at least ten units per half, two supported states and 100 calibration RUN
windows. Never substitute a sensitivity split for primary failure.

RUN: first half supplies encoding. Third quarter supplies local recovery features;
fourth quarter is observed known-position testing. Take up to 1024 nonoverlapping
20 ms windows, regularly sampled from eligible 250 ms-spaced starts, speed 10-200
cm/s, fully inside supported intervals, position gaps <=100 ms. Store exact truth,
timestamps and counts. The local error feature at A's decoded endpoint is median
calibration error of up to 20 nearest tracked positions within 40 cm, at least five
required; otherwise missing, never an artificially low error. A global calibration
median is a separate feature. No B replay information enters any feature.

## Frozen model and policy

Fixed Ridge(alpha=10) regression predicts log1p(A/B posterior-mean endpoint
separation in cm). Train only on real PF split-0 candidates with equal rat,
session and event weights. Fit missing-value imputation and scaling on training
data only. No tuning. Baselines: training mean, spike-only Ridge, entropy-only
Ridge, spikes+entropy Ridge. Full model adds A posterior width and peak, cell
count, local and global RUN recovery, relative local tuning coverage and grid
extent. Evaluate PF leave-one-rat-out descriptively; freeze the full PF models
before generating/evaluating Tanni outcomes.

At deployment retain the predicted lowest-risk half of each session (ceil(n/2)),
using no test outcome. A deterministic event-index tie-break is used. This fixed
coverage ranking rule uses unlabeled test covariates, not a probability threshold
or refitting. Also report 25% and 75% sensitivity, not alternative primary choices.
Compare with lowest A entropy, highest A spikes and random retention at identical
coverage. Random retention's expected mean outcome is the unselected session
mean; do not falsely count random draws as independent biological data.

Primary outcome: mean A/B posterior-mean separation. Regional-content safety
outcome: total variation of A and B posterior probabilities summed within the
same nine fixed tiles. Additional outcomes: A entropy/width, B entropy/width,
per-tile signed mass differences, MAP region agreement and per-event spike counts.
No posterior is smoothed, broadened or replaced by a uniform distribution.

## Known-content checks

Evaluate the frozen predictor on fourth-quarter real RUN windows at the same
20 ms exposure. Also generate two independently seeded draws per candidate of
uniformly distributed true endpoint positions, conditioned on each population's
observed spike count. Matched generator uses the first-half maps. Drift generator
uses second-half maps and independent per-unit lognormal gain (log SD=0.4).
Decoder always uses first-half maps. These are conditional endpoint simulations,
not validated full replay-trajectory generators. Retain zero-spike cases. Check
that retained A predictions have lower true-position error, not merely increased
agreement, separately for RUN, matched and drift generators.

## Gates and accounting

First all inputs/hash/split/count/window/provenance checks must pass. Primary
validation requires all five Tanni animals and all 25 eligible sessions; exclusions
are explicit and cannot be silently removed. Mean outcomes average events within
session, sessions within animal and animals equally. Draws/splits are not animals.
Report descriptive 5000 animal-bootstrap intervals and each animal separately.

Full diagnostic must improve external log-separation squared prediction error by
>=5% versus the frozen training-mean baseline, and >=5% versus spikes+entropy.
At 50% retention, external endpoint separation must fall >=20% versus random
expectation, improve in >=4/5 animals, and improve >=5% versus each simple entropy
and spike-count policy. The animal-bootstrap interval for the difference versus
random must exclude zero. Regional TV must decrease; retained A and B entropy must
not increase. Require true-position error reductions separately for observed RUN
and both conditional generators. Missing/undefined checks are failures, not passes.

Failure means the complete remedy/diagnostic has NOT been validated. Do not retune
using Tanni and call it external confirmation. A narrower successful axis may be
reported descriptively, with the full gate failure alongside it. Replication of
this endpoint-stability diagnostic would still not establish true replay content,
inferred goals, or an unbiased estimate of population-level replay prevalence.

Run only on gpuserver6000/4090 in a detached, persistent systemd user service.
Store frozen protocol/code/input hashes, session checkpoints and a compact report.
