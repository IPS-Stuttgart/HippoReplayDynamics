# Joint-order null: second assay, not a rescue of the first result

Frozen before its simulation outcomes, 2026-09-23. Biological target remains:
does experience change conditional sequence order beyond recruitment, occupancy
and persistence? No real sleep scoring is authorized by this calibration.

## Why a new model

The separately fitted PRE/POST transport assay flagged 17/32 occupancy/dwell-only
simulations despite unchanged routing. Matching fitted nuisance parameters does
not propagate estimation uncertainty or guarantee a valid cross-phase contrast.
Those results and code remain unchanged in their original artifact.

Now fit PRE and POST together under two nested fixed-emission models:

- Null: both phases share off-diagonal log affinities W[i,j]. POST may have its
  own destination propensity v[j]. Each phase has its own state-wise stay
  probabilities and initial distribution.
- Alternative: each phase has an unrestricted positive transition matrix.

For i != j, null transition probability is
  (1 - stay[p,i]) * exp(W[i,j] + I[p=POST]*v[j]) /
  sum_{l != i} exp(W[i,l] + I[p=POST]*v[l]).

This allows changed stationary occupancy and persistence without changing
off-diagonal routing cross-product odds. The null is fitted jointly, not by
transporting a noisy estimate from the other phase. Both models receive the same
fixed emission maps. The unrestricted model starts at the fitted null; the
regularized calibration likelihood must never worsen.

Use hmmlearn's forward/backward and expected-transition kernels. The constrained
M-step is a convex multinomial optimization of shared affinities and destination
offsets, using SciPy L-BFGS-B with analytic gradients. A 1/K pseudocount on each
transition and initial probability is applied in both models. Monitor the
correct regularized observed objective. EM: at most 100 iterations, change
<1e-4, numerical downward tolerance 1e-7. M-step max normalized gradient 1e-6.
Report nonconvergence; never delete failed fits to improve a scientific result.

## Unchanged simulation and evaluation design

Use the exact seeds and observations from the first calibration: five cases,
64 seeds starting 20260924, 80 calibration and 40 test events per phase, six
states, 24 neurons, 20 nonoverlapping 20-ms bins/event. Cases: unchanged,
rate-only, occupancy/dwell-only, emission-only, and order reversal. Exact
phase-specific emission maps and the same deliberately mismatched pooled maps
are supplied in separate arms. Hidden trajectories are never supplied.

Eighteen training neurons infer the state on held-out events. Six different
neurons evaluate count-conditioned future identity 40 ms ahead across one
unobserved bin. Null and alternative have separate causal filters, the same
phase emission map and common event-start prior (alternative equilibrium), and
cannot access future observations. Score alternative minus jointly fitted null,
per evaluation spike, averaging equally over events and the two phases.

This is a DIFFERENT, nested comparison from the original cross-phase transfer
contrast. A changed numerical outcome must not be called a replication of it.

As before, even unchanged-condition seeds calibrate a 95th-percentile threshold
and independent odd seeds evaluate it. Retain the coarse engineering gates:
<=15% exceedances in each negative control; >=80% on reversal; all fits
converged. With only 32 evaluation seeds per case, this is not precise
false-positive calibration, a biological p-value, or evidence of adequate power
against subtle biological changes. Do not adjust the cutoffs after the run.

## Structural identifiability check

Check the exact likelihood equivalence between (a) reversed transition order
with fixed emission identities and (b) unchanged transition order with reversed
state-to-neuron emission assignment. On a directed six-state ring these produce
the same observed process. A freely phase-specific emission model can therefore
absorb an apparent order reversal by state relabelling.

Consequently a real experiment needs an independently defined meaning for state
identity (for example externally validated RUN patterns). A shared fitted label
is not a biological identity. Passing known-map simulations alone cannot support
real-data learning, nor can extra animals fix this symmetry. Cell identities,
state identities and trajectories are distinct quantities.

Do not silently relax the biological question to an easier latent-model fit.
This calibration decides whether this proposed test deserves further work; a
positive biological conclusion still needs independent identity anchors,
estimated-map recovery, recording stability, PRE/POST matching and time/sleep
controls. No HMM novelty claim or causal learning conclusion is justified here.
