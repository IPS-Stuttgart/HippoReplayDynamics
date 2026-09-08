# Sleep rate recalibration improves a rate baseline, not temporal replication

All 320 frozen PRE/POST hc-11 events were scored across eight sessions/four
animals, five fixed neural splits, two encoding variants, three declared rate
conditions and two maps: 19200 wide score rows. There was no event selection
change or dynamical retuning. The primary is POST, direction mixture,
100 population pseudospikes of shrinkage toward a uniform-space RUN reference.

Calibration is cross-event: each event uses the opposite chronological half
of its session/phase candidates, with a one-second guard. Held-out cells may
contribute separate calibration-event spikes, never test-event spikes to
either calibration or latent inference. This differs from the original
RUN-only sleep encoder and is explicitly an observation-transfer diagnostic.

## Primary results

Equal-animal means of per-event medians across five splits, nats/event.
The 95% intervals resample four animal clusters only, holding fitted maps,
calibration observations and within-animal data fixed. They are exploratory;
do not confuse them with the original full hierarchical intervals.

| Contrast | Mean [95% CI] | Positive animals |
| --- | ---: | ---: |
| Recalibrated minus original nonspatial baseline | +0.963 [0.683, 1.244] | 4/4 |
| Recalibrated minus original IMM | +0.484 [-0.007, 0.889] | 3/4 |
| Adapted IMM minus independent positions | -0.035 [-0.126, 0.059] | 2/4 |
| Adapted IMM minus static location | +1.795 [0.743, 3.805] | 4/4 |
| Adapted IMM minus adapted nonspatial baseline | -0.163 [-0.284, -0.053] | 0/4 |
| Adapted real minus permuted IMM | -0.051 [-0.099, 0.020] | 1/4 |
| Change in IMM-minus-independent advantage | -0.096 [-0.231, 0.007] | 1/4 |

The interaction is formed within each event/split before taking medians. It
need not equal the difference between separately aggregated medians.

Stronger-shrinkage sensitivity does not restore IMM-versus-independent:
-0.013 [-0.098, 0.113], 1/4 positive. Pooled-direction primary recalibration is
also inconclusive: +0.029 [-0.043, 0.093], 3/4 positive. Its IMM-minus-global
contrast is mixed (+0.152 [-0.094, 0.500]); the primary nonspatial-baseline win
must not be generalized to every observation setting. PRE's primary
IMM-minus-independent remains negative (-0.050 [-0.074, -0.030], 0/4).

## Decision and scientific scope

`external_temporal_advantage_remains_unsupported`.

A better account of relative cell frequencies can improve held-out likelihood
without revealing temporal or spatial replay structure. The static-location
comparison stays strongly positive, but this does not suffice when independent
positions or a nonspatial rate predictor are competitive. The simple claim that
sleep rate recalibration would restore temporal replication is not supported.

This does not establish absence of sleep replay, identify biological gain or
remapping, or isolate a RUN-versus-sleep interaction. Calibration rates can
reflect which locations are represented as well as physiology and sampling.
This is a previously inspected cohort, so even a positive outcome would have
required new-event confirmation. Do not tune alpha or select favorable animals
to rescue the result. The standalone PF conditional-prediction result and the
recording-coverage measurement study are separate evidence.

## Provenance and verification

Producer commit 0c94b8e1, clean. Run:
`/mnt/seagate10tb/florianpfaff/hc11-sleep-rate-transfer-crossfit-320x5-20260908`.
Manifest SHA256:
149f6808080da85043b049a1a51ae10ab999de5355bd24de3c4d15eb41a751c3.

Independent audit commit 2a226cf0, clean: all 320 raw-count events, all 19200
global predictions, 25600 unadapted original-score regressions, all calibration
folds/weights and 96 aggregate/interval panels. 3072 separate dense neural
predictions agreed within 1.82e-12. All 76800 split and 15360 event contrasts
were reconstructed. Parent RUN maps were hash-pinned and previously audited,
not newly refitted in this diagnostic.

Next decision: close the simple relative-rate-transfer explanation. Further
work needs an independently justified observation/representation hypothesis,
known-generator identification and genuinely new-event validation, rather than
another correction selected for making IMM positive. This result does not
complete the search for a novel high-importance biological mechanism.
