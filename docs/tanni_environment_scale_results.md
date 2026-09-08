# Environment scale candidate: inconclusive, not a biological scaling result

## Why this experiment was run

The high-importance discovery goal remains active. The preceding response was
a literature/results synthesis, not a new experiment. This turn tested a
specific potential advance beyond the existing measurement paper: whether
within-animal environment scale changes independently predictive temporal
organization differently from geometric trajectory acceptance.

The source design is useful because each of five Tanni animals visited four
familiar arena sizes, with randomized B/C/D order and A repeated first/last.
However, area is coupled to context identity and intended recording duration.
This is not a pure environment-area intervention or a replay-speed experiment.

## Frozen experiment and result

Producer `79ed46ad`; independent auditor
`8d8b2172ddfc4dec888a404ed92cc8045c0aad82`.

Run on gpuserver6000:
`/mnt/seagate10tb/florianpfaff/tanni-environment-scale-structure-20260908`

Manifest SHA256:
`df60b23a82c1b5352a1fc7bf4460c1604974dfdfd32dd6f933daa086a93f84af`

All5,224 frozen high-MUA candidates from25recordings/fiveanimals; no new
evidence scoring or event selection. The primary comparison uses the15 B/C/D
recordings, with one session per size per animal. Five cell partitions were
retained separately; split0 was frozen as primary. All25recordings remain in
metadata, support and first/return diagnostics.

|Primary endpoint|Mean change per area doubling|95% t interval,4df|Positive/negative animals|Holm-adjusted permutation reference|
|---|---:|---|---|---:|
|Geometry acceptance| -2.305 percentage points|[-5.692,+1.083] pp|2/3|0.1829|
|Original-order predictive benefit per held-out spike|+0.006082 nats|[-0.003783,+0.015947]|4/1|0.3205|
|Order-by-map interaction per held-out spike|+0.000865 nats|[-0.004738,+0.006468]|4/1|0.6461|

Thus the primary test supports neither a consistent biological size effect nor
an equivalence/invariance claim. The reference enumerates all6^5 within-animal
B/C/D area-label permutations and adjusts the three primary endpoints. The
intervals are across five animal slopes; thousands of events are not thousands
of independent animals. All inference is exploratory in previously examined
data, and is conditional on the observational and model assumptions.

## Sensitivities do not rescue the claim

- Geometry slopes are negative in all five separate cell partitions, ranging
  -1.774 to -2.441 percentage points/doubling. Only split3 has a t interval
  excluding zero. Do not replace the primary split with that split.
- Original-order per-spike slopes range -0.001320 to +0.010697 across splits;
  interaction slopes -0.000348 to +0.003357. All corresponding t intervals
  include zero. Changing the held-out population can change the sign.
- Primary B/C/D session mean training cells increase61.2 ->71.6 ->88.2;
  median training spikes22.3 ->24.6 ->33.6; median event duration0.161 ->0.165
  ->0.178s. These are means of session medians/counts, not pooled events.
- Conditioning on animal, training cells, median training spikes and event
  duration leaves4.51% of within-animal area variance. Adding normalized
  training entropy leaves3.15%. This strong collinearity prevents a clean
  interpretation of a residualized area coefficient.
- The adjusted order slope flips to -0.009082 without entropy and -0.003621
  with entropy; interaction to -0.005516 and -0.001524. Leave-one-animal-out
  coefficients are unstable. These do not establish negative biological
  scaling or causal mediation.
- A-return changes are mixed across animals (positive order gain3/5;
  geometric fraction2/5). The first/return contrast does not give a universal
  experience-dependent effect and cannot isolate elapsed time or arousal.

The parent Tanni failure against other-event cell composition is unchanged.
An order-control benefit is not a substitute for that comparator. Neither
predictive order nor its scale slope measures physical trajectory velocity.

## Verification

44 focused tests pass; Ruff clean. Independent code, without importing the
producer, verifies:

- all25 native NWB arena sizes and animal identities;
-26,120 event/split prediction and geometry joins against the frozen parent;
-125 session aggregates, denominators and zero-held-out-spike handling;
-80 size-effect summaries, animal/t intervals, leave-one-animal-out values and
  exact permutation references;
-96 support-adjusted coefficients using a separate QR/FWL reconstruction.

The parent native-spike/model audits are reused, not claimed to have been
repeated. The plot was visually inspected; it shows individual animals rather
than an event-pooled apparent sample size.

## Publication decision

Close this particular area-scaling lead as inconclusive under the frozen test.
Do not increase events, choose a favorable split or adjust priors to manufacture
a size effect. Do not infer constant speed from the missing slope.

This is useful evidence delimiting the current measurement paper, but does not
meet the active goal of a new high-importance biological discovery. The broader
search remains open. The positive independently predictive-order result and
the coverage/kinematic-recovery study remain valid at their original scope;
this experiment adds no new broad biological claim.
