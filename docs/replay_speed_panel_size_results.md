# More Candidates Restore Availability, Not Universal Speed Calibration

## Frozen Experiment and Audit

Completed nested 30/100/300-candidate panels across all 33 sessions and nine
animals, using fresh simulated paths and spikes on the same empirical RUN
maps. This is a retrospective follow-up to the small-panel diagnosis, not
new biological replication. Each budget is a strict prefix of the same
300-event draw; cycling source durations generates new observations.

- Protocol: `replay_speed_panel_size_protocol.md`.
- Scorer: `4cd12f0320a0110dec81c2f14213122f9954f8b8`, clean at launch.
- Artifact: `/mnt/seagate10tb/florianpfaff/replay-speed-panel-size-all33-20260905`.
- Scoring manifest SHA256:
  `a5310c7bd72bf93a917630e85b12bc29b85f57175a1110e854ce39ef1ba50ebd`.
- Runtime: 2,175.89 s. Source profiles, map support, observation families,
  thresholds and calibration rules were frozen before this run.
- Independent audit passed: 743,688 panel rows, 1,900,800 local intervals and
  1,267,200 excluded-animal intervals. All stored event-moment statistics,
  interval endpoints and decisions reconstruct. Eight sampled draws per
  session additionally regenerate paths/spikes and compare prefix decoding
  against the earlier independent panel implementation. This is not a full
  reconstruction of every raw simulated spike draw.
- Non-rescoring report at commit `c1043a81`; two figures visually inspected.

Primary readout: posterior mean, >=2 active cells and >=3 spikes per decoding
bin, selected continuous core. Estimates require >=5 contributing events and
adequate spatial variation. A/Poisson is matching calibration; B/gain uses
the disjoint RUN map and shared-gain observation stress. All eight readouts,
four conditions and gradient strata remain in the tables.

## Candidate Availability Improves Substantially

Equal-animal means on uniform-gradient test draws:

|Dataset|Condition|30 candidates: finite statistic %|100|300|300: mean contributing events|
|---|---|---:|---:|---:|---:|
|PF|A/Poisson|62.38|93.38|100.00|78.58|
|PF|B/gain|32.75|71.50|97.25|35.29|
|Tanni|A/Poisson|4.88|48.36|77.68|15.16|
|Tanni|B/gain|0.36|16.76|60.96|7.18|

For 300 minus 30 candidates, matching availability improves by +37.63 pp
in PF (animal-bootstrap interval +4.25 to +74.50; three animals improve,
one was already saturated) and +72.80 pp in Tanni (+57.36 to +85.44; 5/5
improve). Combined-stress changes are +64.50 pp PF and +60.60 pp Tanni,
positive in all animals. The initial failure to produce statistics was
therefore partly a finite-candidate problem, not an immutable population
limit. These intervals are descriptive with only four/five animals.

## Available Does Not Mean Calibrated or Precise

At 300 candidates, primary inverse-conformal results are:

|Dataset|Condition|Calibration|Finite intervals %|All-panel coverage %|Finite-only coverage %|Finite width in g|
|---|---|---|---:|---:|---:|---:|
|PF|A/Poisson|Within session|100.00|94.50|94.50|0.726|
|PF|B/gain|Within session|97.25|67.50|67.05|0.726|
|PF|A/Poisson|Excluded animal|100.00|91.38|91.38|0.988|
|PF|B/gain|Excluded animal|97.25|82.25|80.78|0.988|
|Tanni|A/Poisson|Within session|67.80|96.16|94.51|1.353|
|Tanni|B/gain|Within session|58.68|91.00|85.68|1.353|
|Tanni|A/Poisson|Excluded animal|0.00|100.00|Unavailable|Unavailable|
|Tanni|B/gain|Excluded animal|0.00|100.00|Unavailable|Unavailable|

PF matching local coverage has interval 91.50-97.50%, versus 54.00-81.00%
under combined stress. Excluded-animal matching/stress coverage is
78.63-100% / 62.00-98.25%. Transfer intervals are wider and sometimes more
robust than local intervals, not universally worse. But broad coverage is
not accurate effect-size estimation. Width is the mean of finite session
median widths; changing availability changes its conditioning set.

Tanni's 100% excluded-animal coverage is entirely unbounded abstention, not
perfect inference. Missing calibration statistics are assigned infinite
residuals under the frozen rule, so pooled calibration still cannot yield a
finite primary conformal radius. Within-session calibration can be finite in
some sessions, but its average finite width is almost the entire tested
gradient range (g spans -0.75 to +0.75). Neither circumstance proves that a
different justified calibration procedure or larger dataset cannot help.

## Known-Gradient Recovery Is Not Uniform Across Populations

At 300 candidates with matching local calibration, PF detects the correct
nonzero direction for g=+0.5 in 64.38% of panels and g=-0.5 in 58.13%.
Tanni rates are 1.80% and 1.40%. Excluded-animal PF rates are 24.38% and
12.50%; Tanni is entirely abstention. This counts correct-sign intervals,
not merely any confidence excluding zero, and does not prove accurate
magnitude recovery. Animal uncertainty is broad; PF local positive-gradient
recovery has interval 28.13-98.75%.

Importantly, the new larger experiment DOES make some equivalence claims:
PF local matching calibration declares strict +/-0.25 equivalence on known
g=0 in 26.88% of panels, concentrated in two of four animals (interval
0-75%). Tanni primary local output and both datasets' excluded-animal output
make no primary equivalence claims. Do not repeat the earlier small-panel
statement that no calibrated method ever declares equivalence as if it applied
to this experiment.

False equivalence is not zero in all conditions. At the g=-0.25 boundary,
the local PF matching false-equivalence rate is 1.25%; under B/gain it is
9.38%. At g=+/-0.5 under B/gain, false equivalence inside +/-0.25 occurs in
1.25%/1.88%. These are conditional simulated rates with 20 draws per fixed
gradient/session, not biological error probabilities. The report retains
session Wilson intervals, +/-0.10 and +/-0.50 bands and all denominators.

## Updated Scientific Conclusion

More candidates solve an important availability bottleneck and can support
some informative within-population inference, especially in PF. They do not
automatically solve calibration under map/observation mismatch or transfer
to excluded recording populations. The operational limit depends on the
population, event selection, number of candidates and calibration conditions.

q is normalized horizontal position in a surrogate, not measured wall
distance. No result here establishes biological speed uniformity, the true
trajectory content of rejected candidates, or a neural mechanism for speed.
The paper contribution remains measurement validation with explicit failure
conditions, not a universal correction or an optimized replay detector.
