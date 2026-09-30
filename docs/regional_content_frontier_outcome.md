# Regional-content frontier outcome, 2026-09-15

Implemented the first two follow-ups to the 2026-09-14 bounds experiment:
expanded held-out RUN calibration and a full-population discrimination frontier.
No replay-specific cell-participation correction, distance-to-Home discriminator,
hc11 confirmation, or new replay-event selection was run.

## Verified results

- Eight PF sessions, four rats, 1,600 unchanged 20-ms candidate endpoints.
- 230,955 eligible nonoverlapping held-out RUN windows, including 21,475 Home
  windows. Alternate running bouts give 431-2,251 calibration Home windows per
  session, versus 2-11 in the previous restricted calibration sample.
- 573,440 synthetic windows, each scored with a matched likelihood and the frozen
  Poisson readout. Native full-population matched AUC: 0.956 for independent
  Poisson, 0.953 for observed-total-conditioned multinomial; balanced error 0.107
  and 0.117 respectively. These are specified-model results, not replay truth.
- Actual held-out RUN AUC: mean 0.846, range 0.791-0.922.
- Native-RUN-calibrated block-bootstrap compatibility intervals include actual
  validation RUN prevalence in 8/8 sessions, but real candidate distributions
  are incompatible with zero-transfer-slack mixtures in 8/8 sessions. This is
  not a statement of certified coverage or a proof of neural tuning change.
- RUN mean counts are lower than endpoint counts (3.06 versus 5.80 across
  sessions). Binomial thinning cannot reach 56.9-81.3% of sampled desired counts;
  the resulting control is not accurately count-matched to replay.

## Interpretation

The earlier wide bounds must not be used to conclude that regional information
is absent. Under the specified models, full-population native-count discrimination
is good. Conversely, the new calibration does not license corrected replay
prevalence: activity, candidate selection, within-region position distributions,
nonstationarity and transfer all remain possible sources of mismatch.

The next justified diagnostic is activity/selection-conditioned calibration on
independent known-truth controls. The proposed participation fit is not yet
validated; a scalar gain cannot absorb arbitrary cell-specific shared assemblies,
and held-out spike prediction alone does not certify spatial content. Low AUC
for one statistic would not prove a universal no-go result either.

## Artifacts and verification

Server production: /mnt/seagate10tb/florianpfaff/regional-content-frontier-pf-20260915

Report: /mnt/seagate10tb/florianpfaff/regional-content-frontier-pf-report-20260915

Independent audit: /mnt/seagate10tb/florianpfaff/regional-content-frontier-pf-20260915-independent-audit.json

Local report bundle: outputs/regional-content-frontier-20260915

32 new and prior regression tests passed; Ruff clean. Independent reconstruction
checked source RUN spikes/position, bout partitions, simulated RNG draws, both
likelihood ratios, AUC/error/losses and table hashes for all sessions. The figure
was visually checked. Code/input hashes and the fixed protocol are retained;
new code is uncommitted, and the earlier result artifacts were not overwritten.
