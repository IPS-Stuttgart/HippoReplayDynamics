# Kleinman/Foster source alignment and held-out RUN feasibility

Frozen 2026-09-23 before inspecting the new RUN-decoding outputs. This is a
measurement prerequisite for a biological manipulation question, not a result.
No reward, drug or replay-content effect is tested in this run.

## Biological target and existing work

Candidate: does reward change how far ordered replay reaches back along the
preceding approach, beyond event recruitment and starting location? VTA
perturbation is a secondary dissociation, not a new name for the source study's
localization result. This candidate is not yet a confirmatory biological protocol.

Reward-modulated reverse replay incidence is established by Ambrose et al. 2016
(doi:10.1016/j.neuron.2016.07.047). Reward-related replay fidelity is already
addressed by Bhattarai et al. 2020 (doi:10.1073/pnas.1912533117). Berners-Lee et al.
2022 (doi:10.1016/j.neuron.2022.03.010) tests experience-related slowing and reward
changes. Kleinman/Foster 2025 (https://elifesciences.org/articles/99678) studies
reward localization with VTA perturbation. Novelty must be judged against these
specific endpoints, not claimed merely because a new statistic is computed.

## Sources and alignment

Data: doi:10.6084/m9.figshare.28544036. Code: Zenodo record 10368995; six source
files downloaded and checked against its published MD5s. In trodes_analysis.m,
line 120 forms vel from animal_position(2:end,1). Therefore velocity row i maps
to position row i+1, and the first position sample has no released timestamp.
Visits are indexed into original smooth position (lines 152-168), so MATLAB
visit index j>=2 maps to Python velocity index j-2. The README explicitly defines
epoch_change indices in velocity: MATLAB j maps to Python j-1. Do not conflate
the two conventions. Preserve the source exit convention (first sample outside
the visit, except a possible terminal last sample). Audit boundary positions.

Do not sort or repair nonmonotonic clocks. Preserve all 135 spiking-session
denominators and reasons for exclusions. Each neuron is (tetrode, cluster), not
cluster alone. Ignore nonpositive unit IDs but count/report them. Native source
events are inventoried, not decoded. The previous endpoint-support failure is
not reversed by doing a different RUN feasibility test.

## Frozen cross-validation

- Experiment 1, all released spike sessions across the six spiking animals.
- Complete opposite-end traversals from previous visit exit to next visit entry;
  exclude same-end pairs and traversals crossing an epoch transition/gap.
- Consecutive pairs of traversals form lap groups. Five contiguous groups of
  laps define blocked folds; the entire lap is absent from training in its fold.
  Require at least five lap groups. No held-out spikes select units or maps.
- Independent Poisson decoding with a flat prior across training-supported
  position/direction states, 2-cm spatial bins, 4-cm Gaussian smoothing of counts
  and occupancy separately before division. Direction comes from the visited
  destination and must agree with local position change during training.
- Maps use RUN speed >8 cm/s. Drop sample intervals outside [1 ms,100 ms], speed
  >200 cm/s, or implied linear speed >200 cm/s; these are explicit tracking QC,
  not silently interpolated repairs. Use the released speed and position.
- Unit selection is training-only: >=10 RUN spikes and peak smoothed rate >=1 Hz
  in either direction. No cell-type assertion without waveform metadata.
- A position/direction state needs >=0.1 s raw training occupancy. Rate floor
  1e-5 Hz; Gaussian support is truncated at four standard deviations. No smoothing
  across direction states. Track bin edges use released behavior geometry only.
- Test with nonoverlapping 250-ms windows contained in the held-out traversal,
  speed >20 cm/s on average, >=20 cm inside both reward-visit thresholds, and
  at least six spikes from training-selected units. Require every intersecting
  tracking interval to pass timing/speed QC. Report support/exclusion counts.
- True position is the time-averaged linearly interpolated position in the test
  window. Record posterior-mean and joint-MAP error, direction probability/choice,
  uncertainty, log score above uniform, and training-state coverage. No temporal
  dynamics prior. Do not use replay data for any of these choices.

This is an independently held-out adaptation, not a bit-for-bit reproduction of
the authors' RUN QC. Their criteria motivate mean position error <=35 cm and
direction accuracy >=60%; this screen additionally requires >=20 eligible test
windows and >=5 selected units in every fold. Report posterior-mean and MAP
criteria separately; primary is posterior mean. Missing output is not a pass.
Do not change these thresholds after inspection. Preserve all per-fold failures.

## Decision and limitations

Report session and animal/condition coverage, including failures. Do not demand
a biological effect or make event-selection decisions. A usable RUN decoder does
not validate short replay windows, compressed rates, reward contrast calibration,
source event detection, or a Bayesian-smoothing interpretation. Only after this
prerequisite can we freeze and calibrate a replay-content endpoint, inspect its
measurement sensitivity, and then test a manipulation contrast.

Drug/context assignment is not fully crossed within track; three experimental
and three control animals, not 135 independent subjects. No pooled event count
can overcome this replication limit. Rest/replay scoring remains unauthorized by
this protocol alone.
