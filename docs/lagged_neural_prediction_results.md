# Lagged Neural Prediction: Verified Results, 2026-09-09

## Decision

Meaningful scientific progress, not a newly established high-importance
discovery. PF has forward-predictive neural organization beyond the specified
persistence controls. Tanni is weaker and heterogeneous. No model passes the
frozen full two-dataset criterion; do not change horizons, remove animals, or
replace the primary per-spike metric to convert that result into a pass.

This test differs from contemporaneous or smoothed held-cell reconstruction:
the model does not see intervening or target-time training spikes. It predicts
later held-out cell identities, conditional on their total spike count. This
does not show that the animal consciously forecasts or that activity encodes
an intended future behavioral route.

## Frozen Run And Coverage

- Scoring commit: `38d49ea7628092e77d086b97076e4aa036e1cb05` (clean).
- Run: `/mnt/seagate10tb/florianpfaff/lagged-neural-prediction-all9225-20260909`.
- Manifest SHA256:
  `342d5670f77706ad784af7b1f98368f8eaa66e88247cfe78bb188d2daa79195a`.
- 9,225 candidate events, 33 recordings, nine animals; 691,875
  event/split/model/horizon rows, including explicit too-short statuses.
- Runtime: 174.07 seconds with eight CPU workers on gpuserver6000.
- Complete nonoverlapping 20-ms bins; primary center lag 40 ms, leaving
  20 ms between the origin-window end and target-window start.
- At the primary lag: PF 3,883/4,001 temporally eligible, Tanni 5,164/5,224.
  After requiring any held-out target spikes in at least one neural partition,
  per-spike event support is PF 3,878 and Tanni 5,153.
- 20/80-ms lag sensitivities have different temporal coverage: PF 4,001/2,808;
  Tanni 5,224/4,604. Do not treat a horizon difference as a purely within-event
  effect without an additional matched-event sensitivity.
- Final partial bins discarded: 6,046 PF spikes, 6,389 Tanni spikes. These
  counts are repeated by horizon in coverage tables, not three different losses.

All candidates were selected previously from immobile MUA, not from the new
forecasts or continuity scores. They are not automatically validated replay.
Source RUN maps, event calibration gains and K50 neural HMMs were reused,
not refitted. Event medians over five neural splits precede equal session
and equal animal weights. CIs are exact animal bootstrap conditional on the
fixed data/fits, not a correction for the broader exploratory research program.

## Primary Contrasts

Numbers are animal-balanced nats per held-out spike; brackets are conditional
95% animal-bootstrap intervals. Positive-animal counts use the same metric.

| Model and comparison | PF | Tanni |
| --- | --- | --- |
| Neural HMM minus dwell control | +0.08266 [0.07048, 0.09483], 4/4 | +0.01572 [0.00335, 0.02606], 4/5 |
| Neural HMM minus frozen origin | +0.36433 [0.27765, 0.44082], 4/4 | +0.21078 [0.16605, 0.25844], 5/5 |
| Neural HMM minus no history | +0.14591 [0.12904, 0.16277], 4/4 | +0.02427 [-0.00007, 0.04732], 4/5 |
| Neural HMM minus global rates | +0.49286 [0.42951, 0.55485], 4/4 | +0.12877 [0.07953, 0.18309], 5/5 |
| Spatial IMM minus dwell control | +0.04210 [0.03852, 0.04502], 4/4 | +0.01178 [0.00066, 0.02391], 4/5 |
| Spatial IMM minus frozen origin | +0.03647 [0.02055, 0.05238], 4/4 | +0.03547 [0.01481, 0.05613], 5/5 |
| Spatial IMM minus no history | +0.11863 [0.10880, 0.12547], 4/4 | +0.04733 [0.01691, 0.07776], 5/5 |
| Spatial IMM minus global rates | +0.38607 [0.33904, 0.42497], 4/4 | +0.16850 [0.10247, 0.24782], 5/5 |
| Spatial diffusion minus dwell control | +0.10164 [0.09335, 0.10769], 4/4 | -0.01062 [-0.06111, 0.03762], 2/5 |
| Spatial IMM real minus permuted map | +0.03726 [0.03274, 0.04140], 4/4 | +0.00882 [-0.00029, 0.01920], 4/5 |
| Spatial IMM map-by-destination interaction | +0.04113 [0.03727, 0.04438], 4/4 | +0.01116 [0.00074, 0.02237], 4/5 |

PF diffusion also beats the frozen, no-history and global controls in all
four rats. This is not a uniquely IMM-positive result. The learned HMM predicts
better than spatial IMM in PF (paired IMM-minus-HMM -0.10338, CI
[-0.14015, -0.06661], negative in all rats). That does not prove a nonspatial
mechanism: learned states can encode space, and model capacities differ.

Tanni R2478 has the negative learned-HMM dwell increment (-0.00619/spike)
and negative no-history increment (-0.01715/spike). Tanni R2474 has the
negative spatial-IMM dwell increment (-0.00681/spike) and map increment
(-0.00670/spike). They are different animals: there is no single common
animal to remove to make all models pass. No animal was removed.

The learned-HMM Tanni dwell increment is positive in all five animals at
80 ms, but that is a sensitivity with different event coverage. It cannot
replace the predeclared 40-ms endpoint. Conversely, PF spatial IMM is worse
than its frozen origin at 20 ms in all four rats. Horizon dependence is real
in the evaluated scores and argues against a horizon-independent claim.

## What This Does And Does Not Resolve

1. Contemporaneous cross-cell prediction did not by itself establish a
   forecast. This experiment now demonstrates a positive forecast contrast
   for PF with a genuine within-event observation gap.
2. Beating frozen origin alone can reflect uncertainty growth. The additional
   dwell-preserving and no-history controls reduce that ambiguity, but do not
   uniquely identify a generative mechanism.
3. The spatial kernels are symmetric local diffusion, not directional
   momentum. Preferred spatial destinations here mean local transition
   geometry, not an intention or learned route.
4. The dwell controls preserve state self-transition probabilities, but not
   all stationary occupancy or full-joint recurrence statistics. An
   occupancy-preserving transition control would be a stronger discriminator
   before interpreting any effect as destination-specific sequence dynamics.
5. Both datasets show some positive contrasts. The full prespecified
   cross-dataset mechanism criterion fails. This is neither absence of
   temporal organization nor evidence for a universal replay mechanism.
6. No new continuity threshold, fuzzy classifier, positive event subset or
   speed-uniformity claim follows from this experiment.

## Verification

- Independent audit:
  `/mnt/seagate10tb/florianpfaff/lagged-neural-prediction-all9225-20260909-audit/lagged_prediction_audit.json`.
- All source/output hashes, every fold, every event's bin/cell support,
  all 4,289,625 split contrasts, 857,925 event contrasts, session/animal
  reductions, exact bootstrap intervals and decisions checked.
- Separately written filtering, transitions and multinomial likelihoods
  reconstructed 70,350 predictive scores from the first chronological event
  in each fold (165 events), all their neural partitions and eligible horizons/models.
  Maximum absolute discrepancy: 5.68e-14 nats.
- Native raw files were not reopened and source model fitting was not repeated.
- Audit provenance records a dirty tree because the new non-rescoring
  reporter/test were untracked at audit time. All frozen producer/kernel/
  protocol/source hashes were verified unchanged; scoring itself was clean.
- 53 forecasting/related tests and three reporter tests pass; Ruff passes.

## Next Decision

Keep this as a bounded forecasting result and a candidate component of a
methods/mechanisms paper. Do not call it a novel high-importance discovery.
Do not retune the failed Tanni criterion. A stronger scientific follow-up
would distinguish preferred transitions from occupancy/persistence under a
matched-information, predeclared control, then seek genuinely independent
confirmation. Forecasting neural activity itself is established, including
[predictive sequence learning](https://www.sciencedirect.com/science/article/pii/S0896627324003714);
the possible contribution would have to be the experimentally verified
dissociation, not merely another successful HMM.
