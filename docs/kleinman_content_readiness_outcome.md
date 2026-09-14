# Kleinman/Foster endpoint-support readiness: not a validated remedy

All jobs ran detached on gpuserver4090. Original inventory retained; reader
revision e59588f preserves missing Experiment 2 novelty labels. Independent
verifier 5d000ca uses direct raw spike masking, not the producer counter.

209 native sessions inventoried. 194 have readable monotonic velocity clocks;
15 contain a backwards timestamp and remain excluded, not silently sorted.
135 sessions have raw spike files; 133 have usable monotonic clocks. All 194
readable sessions have one extra position sample relative to velocity timestamps;
alignment remains unresolved. Neither drop-first nor drop-last is adopted.

Prospective primary stratum: Experiment 1 control/saline, 35 spiking sessions
across three within-experiment animals. Counts below use split0, two seeded
disjoint equal halves of ALL raw units. They precede place-field/cell-type QC.
Both-halves support means >=3 spikes and >=2 active units in each 20-ms window.

| Event / window | Con_1 | Con_2 | Con_3 | Total |
|---|---:|---:|---:|---:|
| SDE unchanged endpoint | 17/763 | 0/840 | 14/674 | 31/2,277 |
| SDE native-peak-centered | 336/791 | 58/886 | 523/720 | 917/2,397 |
| Ripple unchanged endpoint | 10/697 | 0/175 | 31/735 | 41/1,607 |
| Ripple native-peak-centered | 25/718 | 0/173 | 42/733 | 67/1,624 |

Denominators are valid immobile windows (|speed|<5 cm/s, clock gap <=100ms,
entire fixed window within native event), not all native events. All raw native
counts and invalid-window reasons remain in the source tables. Peak and endpoint
phases have different valid denominators and are not directly interchangeable.
Native event sets overlap; do not sum SDE and ripple rows as independent events.

The low endpoint support is not negative replay biology. Peak windows contain
much more spiking, but substituting them would change the destination-readout
question. This release is not currently a strong independent dataset for the
unchanged-endpoint, split-population remedy, and no replay decoder was fitted.
Unit medians in the prospective stratum are 25, 11, 40 for Con_1/2/3, before QC.

Verification passed: all 209 catalog rows, 5,880 direct raw window checks,
1,962 summary groups, hashes unchanged. Missing novelty does not discard
Experiment 2 summaries. The 15 clock exclusions and all zero-support cases
are explicit. Technical audit success is not scientific-goal completion.

Authoritative locations:
- /home/florianpfaff/results/kleinman-content-readiness-v2-20260914/
- /home/florianpfaff/results/kleinman-content-readiness-verify-20260914/
- code: /home/florianpfaff/HippoReplayDynamics-kleinman-content-20260914/

Separate metadata-only reconnaissance of Alme familiar-room begin1 files on
gpuserver6000 found seven animals, 25-66 raw units each, explicit monotonic
position timestamps and approximately 1m x 1m arenas. No Alme replay outcomes
or remedy results were inspected. That remains a possible independent resource,
not a successful transfer or a selected positive cohort.
