# Exact conditional-risk diagnostic: informative observations can fail a class guard

## Result and scope

An exact Bayesian counterexample shows that an informative new observation can
reduce overall Brier risk and mean absolute error in BOTH classes, yet increase
Brier risk in the common negative class. The added observation is genuinely
informative and all truth/observation combinations have positive probability.
No wrong map, miscalibrated posterior, estimated parameter or finite-sample noise
is involved in this control.

This is a mathematical interpretation check, not an external-data diagnostic,
validated remedy, new theorem, or explanation established for the PF recordings.
It changes neither the acquisition protocol nor any failed run's classification.

## Fully specified example

Let Y=1 denote the rare region, with prior probability3/100. The coarse
observation gives no information, so its true posterior is always3/100.
After the informative observation, the true posterior is:

- 1/1000 with probability30/31;
- 9/10 with probability1/31.

These posterior values average to3/100 exactly. The joint distribution is
P(Y=1,Z=z)=P(Z=z)*q(z), and P(Y=0,Z=z)=P(Z=z)*(1-q(z)). Thus this is a valid
Bayesian observation channel, not an arbitrary change to decoder predictions.
Every joint cell is positive. Bayes' formula independently recovers both
posterior values from the serialized joint distribution.

| Expected risk | Coarse observation | Informative observation |
| --- | ---: | ---: |
| Overall Brier |0.029100|0.003870|
| Rare-class Brier |0.940900|0.041871|
| Common-class Brier |0.000900|0.00269471134|
| Rare-class mean absolute error |0.970000|0.129000|
| Common-class mean absolute error |0.030000|0.00398969072|

Overall Brier falls by86.7%, while common-class Brier nearly triples. Absolute
errors here describe a binary, unit-distance state space, not centimeters or
the geometry of the PF arena. Rare confident mistakes in the common class
can cost more squared error than an almost-always-small posterior probability.

For the simpler refined posterior pair{0,h}, the common-class Brier change is

  p/(1-p) * (h-p) * (1-p-h).

For0<p<1/2, it is positive whenever p<h<1-p, although overall Bayes risk improves.
At sufficiently high precision it becomes nonpositive. This therefore does NOT
prove that all class-wise safeguards are jointly impossible.

## Implication for the acquisition search

The screened three/four-cell producer reports seven validation failures, all
common-class Home Brier failures, with no physical-error failure among its frozen
winners. That pattern motivates this control but does not prove the same
mechanism explains the neural result. The neural independent audit was still
running when this note was written.

The correct interpretation remains: the acquisition rule does not meet the
predeclared full-cohort requirements. Do not additionally infer that the cells
carry no useful information, or that a worse common-class Brier score establishes
an incorrect Bayesian likelihood. Do not relax the guards post hoc, promote the
run to replay/external scoring, or substitute this toy for independent validation.

## Reproduction

- Computed on gpuserver6000; no neural recordings were read.
- Code commit:53890402e9e97ee41ea900b149f4deb0d7314d98.
- Script:scripts/conditional_brier_refinement.py.
- Tests:9 passed, including exact joint-distribution reconstruction, the analytic
  sign boundary, invalid channels and corrupted risk rejection; Ruff passed.
- All probabilities/risks were computed with fractions.Fraction, not Monte Carlo.
- Root:/mnt/seagate10tb/florianpfaff/conditional-brier-refinement-20260915.
- Manifest SHA256:b46698c9b302ade08788529fa28513b58867a3bba47f2cc22a079cf63e2a92d7.
- Execution:2026-09-15T07:18:25UTC.

The original acquisition source files and guards were not changed. The
mathematical control is a separate committed module; the concurrent neural audit
continues to verify its own frozen source hashes. The active goal remains unmet.
