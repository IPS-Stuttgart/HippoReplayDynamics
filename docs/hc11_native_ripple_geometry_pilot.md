# hc-11 native-ripple geometry pilot

## Question

Does the Pfeiffer/Foster trajectory-family and clean-IMM hierarchy generalize to
native POST-NREM ripple events on constrained linear and circular mazes?

This is not a direct 2D open-field replication. The public native-ripple cohort
contains five sessions from two animals: three linear-maze sessions and two
circular-maze sessions.

## Claim-taxonomy correction

The historical table below compared the best of diffusion, fragmented, and IMM
against stationary. It is a broad nonstationary/reactivation comparison, not an
ordered-trajectory count, because fragmented has no temporal path continuity.
The current paper-facing split is:

- ordered: diffusion and first-order IMM;
- nonordered: stationary and fragmented.

Under this stricter split, the native-envelope 10 ms tier has 1/100
ordered-confident events (median ordered-minus-nonordered margin -0.37), the
50 ms random tier has 4/100 (median -0.26), and the 50 ms spike-support tier has
16/100 (median +0.20). The larger historical counts must not be described as
ordered replay.

## Geometry and encoding controls

`scripts/score_hc11_webshare_native_ripple_evidence.py` implements:

- reflecting endpoint transitions for linear mazes;
- periodic transitions for circular mazes, so the coordinate seam is not a wall;
- MAZE-only, movement-filtered, sorted-CA1 place-field encoding;
- five-fold held-out RUN decoder QC;
- pooled and direction-conditioned encoding sensitivity;
- native ripple events intersected with POST and NREM;
- exact full-grid stationary, diffusion, fragmented, and first-order IMM rows.

Momentum is not in this first geometry pilot because the existing exact-sparse
momentum implementation is Euclidean and has not yet been generalized to a
periodic circular state space.

## Frozen pilots

The primary technical run used 20 events per session, 10 ms bins, and native
ripple envelope boundaries. Diagnostic sensitivity runs used 5 ms bins, 25 or
50 ms padding on each side, and three pre-evidence ranking rules:

- native ripple power;
- spike/active-unit support;
- deterministic random sampling.

The comparison is generated without rescoring by
`scripts/report_hc11_native_ripple_geometry_sensitivity.py`.

## Observed result

All five RUN decoders passed, with median cross-validated errors from about 6 to
19 cm. Direction conditioning changed decoder error and evidence margins only
modestly, so direction pooling is not the main explanation.

| Tier | Broad nonstationary-vs-stationary confident | Median broad margin | IMM confident over fragmented | Strict clean IMM |
| --- | ---: | ---: | ---: | ---: |
| Native envelope, 10 ms | 11/100 | +0.24 | 6/100 | 1/100 |
| Native envelope, 5 ms | 12/100 | +0.32 | 9/100 | 1/100 |
| 25 ms padding, ripple-power rank | 17/100 | +0.82 | 11/100 | 1/100 |
| 50 ms padding, random | 23/100 | +1.57 | 8/100 | 1/100 |
| 50 ms padding, ripple-power rank | 25/100 | +1.60 | 16/100 | 1/100 |
| 50 ms padding, spike-support rank | 47/100 | +3.77 | 26/100 | 5/100 |

`Strict clean IMM` requires all three conditions: trajectory-confident, first-order
IMM best among the four scored models, and IMM at least 5.5 log-evidence units
above fragmented. In the strongest tier, four of five strict events come from
Achilles_10252013; the fifth comes from Cicero_09102014.

## Interpretation

hc-11 contains a replay-rich native-ripple subset under the broad comparison.
Wider windows and high spike/active-unit support improve broad evidence even after
normalizing margins per time bin or spike. Ripple power itself does not predict
the margins.

The result does not currently replicate the Pfeiffer/Foster clean-IMM finding.
The strict clean-IMM intersection is sparse and session-localized, and the four
model pilot does not include periodic exact-sparse momentum. Linear versus
circular topology is not a persuasive explanation: geometry differences change
with the selection tier, while session/animal decoder support is a larger effect.

The correct role is external trajectory-subset and specificity evidence. Do not
launch or interpret the full Gate 2/3/4 ladder as dataset-wide IMM confirmation
unless a larger, predeclared exact-core cohort first produces a distributed
strict clean-IMM subset.

## Commands

```bash
PYTHONPATH=src python3 scripts/score_hc11_webshare_native_ripple_evidence.py \
  --max-events-per-session 20 \
  --decoder-max-bins 1000 \
  --output-dir results/hc11-native-ripple-geometry-pilot100

PYTHONPATH=src python3 scripts/score_hc11_webshare_native_ripple_evidence.py \
  --max-events-per-session 20 \
  --decoder-max-bins 100 \
  --event-padding-s 0.050 \
  --event-ranking spike_support \
  --output-dir results/hc11-native-ripple-geometry-pilot100-pad50ms-spike-support
```
