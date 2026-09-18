# Paired count-conditioned likelihood check

Freeze this diagnostic before scoring. The question is whether the previously
observed known-label selection distortion depends on fitting an unconditional
Poisson likelihood to count-conditioned observations. This does not replace the
failed real-data random/targeted coverage tests or seek a new biological positive.

## Fixed observations

Use both primary sessions' existing selection-transport calibration banks:
40 original count anchors, five splits, 20 realizations, both true tracks,
ordered and whole-bin-shuffled copies, full plus five fixed half subsets.
Regenerate exactly the same spikes, paths and random seeds from the frozen
producer. Validate every anchor's unchanged opportunity counts and independently
evaluated content values against the old tables. Do not generate new favorable
paths, count anchors or realizations. Known maps do not mean known decoded paths.

Only the sequence likelihood changes. For an observed population A:

    p_i(x) = max(rate_i(x), 1e-4) / sum_j max(rate_j(x), 1e-4)
    log L(x | counts, total count) = sum_i count_i log p_i(x)

The multinomial coefficient is location-independent and cancels in decoding.
Keep equal track/uniform valid-bin priors, zero-spike-bin exclusion, the same
weighted correlation, five active cells/five nonempty bins, 499 time/499 field
shuffles and p < .025 for both per track. Field-shift nulls normalize AFTER
shifting maps and restricting to the observed population. Do not floor the
normalized probabilities a second time. Do not change the Poisson defaults or
any existing real-data outputs.

This is the generator's conditional identity likelihood. For the full population,
the simulated count totals were set externally. For thinned subsets, their totals
also depend on the generated cell identities: conditioning discards that count
information. Thus this is not a claim that the thinned conditional posterior uses
all information in the full generative model. A change measures likelihood
sensitivity, not a uniquely identified original failure mechanism.

## Readout

Pair each new conditional result with its old Poisson result. Report acceptance
per true track, selected true track-2 fractions at full/half coverage, loss given
full acceptance, correct-label rates, and independent true-signed content among
lost copies. Truth proportions use known simulated labels, never decoded scores.
The generating experience mixture is exactly 50:50.

Primary strata remain POST ripple count anchors: 12 RAT3 and 14 RAT5. Average
realizations, splits and coverage repeats within each original anchor; bootstrap
original anchors 2,000 times with the same weights for both likelihoods and both
true tracks. Preserve even/odd-realization diagnostics but do not choose the
favorable half. Require >=95% finite draws for any interval.

Predeclared comparisons: conditional minus Poisson selected true track-2 fraction;
reduction in absolute full/half selection distortion from 0.5; and conditional
half-minus-full experience shift. Report raw signed effects and intervals even
if they contradict the proposed explanation. Report shuffled-null acceptance
and ordered detection sensitivity side by side: erasing bias by erasing all
detections is not recovery. No interval on simulated copies is a population
animal interval. No real experience-fraction correction is authorized.

## Interpretation

If distortion persists under this conditional likelihood, the original
observation mismatch is not a sufficient explanation in the tested simulation.
If it shrinks while ordered detection survives, likelihood assumptions contribute
to experience visibility here. Either finding remains conditional on this map,
count and path bank. It neither proves nor rules out replicated biological
experience bias. Do not change real replay thresholds based on this diagnostic.
