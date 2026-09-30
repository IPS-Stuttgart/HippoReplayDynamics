# Mixed spatial content: identifiability pilot

Date: 2026-09-21. Status: exploratory simulation calibration, NOT a biological finding.

## Question and distinction

Can an apparent failure of a single-location decoder reflect multiple spatial contributions within the analysed window, rather than only missing cells or uncertainty about one location? A multimodal posterior over one location is not equivalent to a model in which two locations contribute to the observed population vector.

This is a different hypothesis from the pending DANDI000978 CA1/PFC coverage-versus-content experiment. Do not infer regional labels, use unconfirmed unit mappings, or revise the frozen recording-coverage paper for this pilot.

## Models and scoring

All models use the same place-field dictionary. The single-source comparator retains ALL occupied single states, including full posterior uncertainty. The mixed-source model averages the firing rates at two spatially separated anchors, with equal weight. The serial comparator occupies the first anchor in the first half-window and the second in the second half-window; both temporal orders are integrated. Pair anchors are chosen from geometry, never test spikes. No states are selected using held-out observations.

The likelihood is a multinomial time/cell-identity model with the whole-window spike totals in inference population A and held-out population B conditioned on. Its score is log(sum_h p(h)L_A(h)L_B(h)) - log(sum_h p(h)L_A(h)), omitting only the B multinomial coefficient that cancels in comparisons. The held-out count vector is scored jointly, not as a product of posterior-averaged single-spike probabilities. Each model integrates a uniform prior on its own finite library. This is within-window cross-neuron predictive information, not future forecasting or raw Poisson event evidence. A separate scalar gain for each population/window is removed; unit-specific gain drift is NOT removed.

## Exact alias and mandatory limit

For independent Poisson emissions integrated over a window of length D,

    E[N_i] = D * (w*f_i(a) + (1-w)*f_i(b))

holds both for simultaneous rate mixing and for a serial source spending w*D at a and (1-w)*D at b. Their full count-vector laws are identical. Conditioning on population totals does not break that equality. The test suite verifies equal predictive scores after temporal aggregation. A source that switches within every retained subwindow also remains indistinguishable at that finer resolution. This is not a novel theorem or a claim of biological simultaneity.

## Frozen first pilot

Use one toy encoder and, if locally available, RUN-fitted Pfeiffer/Foster encoders from Rat1/Open1 through Rat4/Open1. Existing loader and 6-cm grid, 2-bin map smoothing, >=10-cm/s running, ripple-excluded map fitting, and >=0.02-s occupied-state support are used. Real events are NOT scored. The sample is four recording-derived encoders, not four positive biological replications. RUN map accuracy is not revalidated by a simulation using those same maps.

Fix 32 geometrically distributed pair anchors, minimum separation 40 cm, equal weights, one reproducible neuron split, and A/B total-count tiers (4,4), (12,12), (32,32). These counts are design settings, NOT measured typical ripple counts. For each tier generate 256 windows under each of five mechanisms: single, mixed, resolved serial, single with unit-specific lognormal gain drift (log-SD 0.5), and unresolved serial (identical retained-bin intensities to mixture).

The first 128 simulations calibrate thresholds; the last 128 evaluate them. A coarse positive requires mixture-minus-single above max(0, calibration single-null 95th percentile). A fine positive requires mixture-minus-max(single,serial) above max(0, the separate 95th percentiles under the single and resolved-serial nulls). This does not guarantee exact 5% false-positive rates in finite simulation samples. Do not tune to evaluation positives. The gain-drift and unresolved-serial controls are sensitivity checks, not secretly added to the calibration family.

Report every mechanism, count tier, source dictionary size, cell split, numerical alias error, source/script hashes, and simulation denominators. Favorable matched-family recovery is only a necessary feasibility check; it does not validate the generative model on sleep data.

## Conditions before a real-content claim

A real experiment must additionally include: a flexible continuous/multi-switch single-trajectory model; event-specific burst envelopes and independently calibrated unit gains; uncertainty in independently fitted RUN maps; correct-map versus cell-permuted-map controls; fixed LFP- or separate-population event detection; held-out neurons and whole-event calibration; a measured count/resolution feasibility check; no selection on evaluation scores; and animal-level replication. A one-switch toy comparator is not a substitute for this set.

If those tests succeed, the potential result is: selected events contain independently supported mixed spatial contributions at a specified temporal resolution. Do not call it simultaneous replay, representation of multiple memories, causal memory transfer, or evidence that every fragmented event is replay.

## Execution and outputs

    python scripts/spatial_mixture_identifiability.py --output out/toy
    python scripts/spatial_mixture_identifiability.py --dataset-root /verified/PF/root --output out/empirical_encoders

Outputs: simulation_scores.csv, simulation_summary.csv, manifest.json. No raw neural recordings or fitted cell maps are uploaded. The workflow uses the gpuserver6000 self-hosted runner by default, read-only repository permission, an isolated environment and a bounded runtime. It must report missing or ambiguous dataset discovery without inventing input paths.
