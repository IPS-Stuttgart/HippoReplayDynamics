# Native MAZE transfer: positive control, not a replay discovery

The frozen real-data experiment completed 1600 behavior-selected windows in
eight sessions/four animals, with 64000 model/baseline rows. All technical
gates passed. An independent auditor reconstructed every native/capped count
matrix, 5376 predictive scores, all contrast aggregates and cluster intervals.
Maximum independently recomputed predictive error was 1.55e-13.

Primary train-only unit QC, direction mixture, equal-animal means in nats per
200 ms, with exploratory animal-cluster 95% intervals:

| Contrast | Native counts | Sleep-total cap |
| --- | ---: | ---: |
| Measured position - nonspatial | 1.162 [0.531, 2.192] | 0.941 [0.477, 1.733] |
| Measured position real - wrong map | 4.010 [1.505, 8.150] | 3.339 [1.331, 6.636] |
| Independent decoder - nonspatial | 1.056 [0.247, 2.432] | 0.818 [0.202, 1.894] |
| IMM - independent decoder | 0.267 [0.081, 0.558] | 0.263 [0.081, 0.539] |

All these primary point estimates are positive in all four animals. Neural
inference excludes held-out cells. Measured-position scores are explicitly
supervised controls. The count cap is not exact sleep-information matching.

Important sensitivity: using the older full-MAZE-selected frozen unit list
gives a negative measured-position-minus-nonspatial score in Buddy (-0.908
nats/window), and the aggregate interval includes zero: 0.923 [-0.473, 2.452].
The primary training-only unit selection does not license suppressing this.

The experiment narrows gross RUN encoding failure as an explanation for the
earlier sleep-prediction result. It does not establish a RUN-versus-sleep
interaction, a theta-sweep result, or uniquely switching dynamics. The
temporal priors were inherited from the replay analysis, not calibrated for
normal movement. No sleep thresholds or event selections were changed.

Next diagnostic: freeze cross-event, relative-cell-rate recalibration of the
sleep observation model. Distinct sleep events may estimate only global cell
weights, never an event's held-out spikes. Success over a recalibrated
nonspatial baseline, independent snapshots and wrong maps would support
further transfer work; a rate-only improvement would not rescue replay.
Marginal rate changes can reflect content occupancy, not necessarily gain.
This is a model-transfer diagnostic with published precedent, not yet a novel
biological explanation.

Producer: 012a9b02, clean. Run manifest SHA256:
2d8687b40fd43f71fc4b44732883faf96d159cfb8790e35569c77e3e49f6ab8d.
Verifier: 10e1edb6, clean. Raw-count audit uses raw timestamps; map refits share
the frozen fitting helper. Posterior-mean errors and credible coverage were
not independently reconstructed. Four animals limit population inference.
