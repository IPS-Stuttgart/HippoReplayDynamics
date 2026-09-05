# Paper Prospect: Measuring Replay Under Limited Recording Coverage

Status: promising methodological study, not yet a completed paper-ready result.
This is a synthesis of verified experiments, not a new biological claim.

## Proposed Contribution

Quantify when recording coverage and decoding choices change apparent replay
continuity or conceal spatial speed variation, and validate a procedure that
reports when the data cannot distinguish the biological alternatives.

The contribution cannot simply be that sparse recordings create decoded jumps.
Ji et al. explicitly state that limited, uneven place-field coverage can create
apparent jumps in their 2026 replay-dynamics paper (Methods, after Eq. 37):
https://www.nature.com/articles/s41467-025-68042-3
Takigawa et al. also provide a framework for comparing replay detectors without
ground truth, including false-positive controls:
https://elifesciences.org/articles/85635
See `replay_coverage_novelty_scope.md` for additional calibration prior work.

The potentially distinctive package is paired real-population perturbation,
controlled recovery of known speed gradients, information-preserving versus
information-changing controls, and independently validated limits on inference.
This is a candidate contribution, not an established priority claim.

## Evidence Already in Hand

1. Real-data measurement sensitivity: 12,141 fixed high-MUA candidates from
   33 sessions/nine animals across Pfeiffer/Foster and Tanni. Removing half the
   recorded population decreases continuity acceptance in every animal, with
   and without bin-support filtering. The underlying candidate windows do not
   change. This does not identify which rejected candidates are real replays.
2. Spatial information beyond total counts: in empirical-map simulations,
   pooling removed-cell identities preserves every fine-bin spike total but
   retains most continuity loss. Oracle relabeling uses known position and must
   not be interpreted as a count-only rescue.
3. Sensitivity/specificity trade-off: in a prespecified large-arena RatInABox
   slice, 20 to 40 ms windows increase eligible continuous recovery from 11.34%
   to 64.34%, but shuffled acceptance from 0.52% to 18.23%. These are simulated
   geometric acceptances, not biological replay false-positive estimates.
4. Speed inference can lose substantial signal: injected spatial gradients
   are attenuated at source-comparable information budgets. Refining a 4/8/16 cm
   decoding grid barely improves recovery in the controlled factorial. A flat
   decoded profile is therefore not sufficient evidence of uniform true speed.
5. Uncertainty needs calibration: held-out RUN localization can be useful while
   nominal posterior coverage is poor. Neither a sharp posterior nor an
   uncalibrated uncertainty radius validates replay continuity by itself.
6. Known-map recovery can be optimistic: the separately estimated RUN-map
   benchmark changes PF true-path recovery from 35.82% to 19.60%, while the
   smaller Tanni change (7.94% to 6.80%) is not rat-uniform. Strong gradient
   attenuation survives independent maps and a declared shared-gain stress.
   This extends the validation beyond exact generator-map decoding without
   pretending the surrogate is real replay ground truth.
7. Calibration has a transfer and availability boundary: independent simulation
   test panels give approximately 95% conformal coverage before continuity
   selection under the matching generator. Under disjoint maps plus shared
   gain, coverage falls to 76.75% PF/89.40% Tanni in the bin-supported readout.
   The primary continuity-selected conformal analysis produces finite intervals
   only in Rat1 (PF) and abstains in every Tanni session. Neither calibrated
   baseline establishes the predeclared +/-25% gradient equivalence, even for
   simulated constant-speed truth. This is a quantified limitation, not a
   universal correction or evidence of biological uniformity.

Exact estimates, inclusion rules and provenance are in the real subsampling,
RUN validation, counterfactual, geometry and map-mismatch results documents. Synthetic
population intervals must not be presented as replication across animals.

## What Would Make the Paper More Useful

The final product should give an analyst a defensible decision: a speed effect
of a declared magnitude is recoverable under these conditions, or the data do
not support that inference. Merely cataloguing more decoder artifacts is weaker.

Required work before that claim:

- The first independently estimated map/shared-gain check is completed; retain
  its mixed outcome and test robustness beyond that declared stress family.
- The frozen inverse Gaussian/conformal and raw-bootstrap comparison is now
  completed on independent A/B simulation panels, with honest abstention and
  mixed transfer. It is not new-animal validation. Do not promote a calibrated
  method for biological replay without further independent-population checks.
- Compare with established replay detection/calibration baselines and carry
  real event-definition sensitivity into the conclusions.
- Evaluate a prespecified meaningful equivalence range only where injected
  alternatives are recoverable. Otherwise report an identifiability limit.

Possible figure sequence: real population removal; controlled field geometry;
window-length recovery/null trade-off; speed-gradient attenuation; calibration
transfer and abstention. All five now have results, with explicit scope limits.
The integrated paper and event-definition/baseline comparisons remain incomplete.
See `replay_speed_identifiability_results.md` for the last panel's evidence.
The real detector comparison now has frozen, audited core/fixed-window inputs
across all33, with ripple availability in 8/8 PF and 23/25 Tanni sessions.
This is preparation, not a demonstrated detector-robust coverage effect; see
`replay_coverage_event_definition_results.md`. Trace validation and paired
decoding remain necessary, especially given low Tanni detector overlap.

## Claims Not Supported by This Study

- Replay has uniform speed throughout either biological environment.
- Recording coverage fully explains the PF/Tanni trajectory-count difference.
- Replay implements Bayesian smoothing or surprise-driven credit assignment.
- A continuity heuristic alone establishes replay significance.

The IMM program is a separate possible methods story. Its novelty and
replication should not be inferred from these coverage experiments.
