# PF-to-Tanni stability diagnostic: frozen external failure

Run: `population-content-stability-20260914`, gpuserver6000. Production commit
`e9c2b6e2`, protocol/measurement commit `b030e796`. All four detached pipeline
stages completed with exit code 0. Validation itself failed as specified.

Development used 4,069 PF candidates, eight sessions and four rats. The predictor
was frozen before generating outcomes for 8,072 Tanni candidates in all 25
recordings and five rats. Independent flat-prior Poisson, 20 ms endpoint windows,
first-half RUN maps, equal disjoint cell halves. Original pooled-MUA candidate
selection and full-RUN unit eligibility remain conditioning assumptions.

## Main external result

At fixed 50% retention per session, A/B mean endpoint separation fell from
42.375 to 33.567 cm (20.8%). The difference was positive in all five animals,
with descriptive animal-bootstrap interval [6.95, 10.72] cm. Regional posterior
total variation fell from 0.4452 to 0.3800.

However, A's normalized posterior entropy rose from 0.8629 to 0.9332; B's rose
from 0.8600 to 0.8726. Retained A true-position error INCREASED by 2.97 cm on
independent observed RUN windows, 3.58 cm on matched conditional simulations,
and 2.98 cm on drift/gain conditional simulations. The full predictor's external
log-separation MSE was 3.237, worse than the constant PF baseline's 1.034.

Therefore this is NOT a validated remedy. Selecting for decoder agreement can
prefer weak, broad posteriors whose means agree while localization deteriorates.
This interpretation is consistent with the entropy and known-position checks;
it is not proof of the unique causal mechanism. No threshold/feature refitting
on Tanni is allowed to turn this held-out failure into a confirmation.

## Audit

Independent reconstruction checked all 12,141 candidate endpoint counts across
33 recordings, sampled 4,713 real/RUN posterior readouts, reconstructed held-out
RUN spike counts, and verified fixed retention and equal-animal aggregates.
The audit passed. Its explicit scope does not independently reconstruct the
conditional simulation truths or Ridge coefficient fitting. The unit suite
checks deterministic simulation counts and train/external isolation.

Thirty new/related tests passed and Ruff passed. The original audit service was
stopped solely to eliminate repeated NPZ decompression inside its checking loop;
the corrected read-only audit completed successfully. No production scoring was
restarted or altered in response to these results.

Server artifacts:
`/mnt/seagate10tb/florianpfaff/population-content-stability-20260914`

Compact local package:
`/mnt/c/Users/emper/Documents/codex/2026-09-14/population-content-stability`

## Next experiment, not a success claim

Test RUN-only spatially balanced population sampling against equal-count random
sampling, using all events without abstention or changing the decoder. Freeze
that remedy on PF development data, then test Blackstad--Moser open-field data
as a separate external recording set. Available native data cover five animals
and 741 frozen high-MUA candidates. Their prior trajectory yield was low; that
does not make these events true replays, nor does it disqualify a conditional
decoder-measurement study. Full input/clock/encoding checks are still required.

The goal of a validated remedy or diagnostic remains open.
