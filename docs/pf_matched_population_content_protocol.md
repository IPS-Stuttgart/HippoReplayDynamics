# Matched PF Populations: RUN-Only Selection

Frozen 2026-09-13 before selecting populations or inspecting their replay outputs.

## Aim

Test whether equally sized neuronal samples with similar firing rates, field
stability, field area, and global held-out RUN decoding nevertheless yield
different Home-content readouts for unchanged replay candidates. This is a
recording-composition diagnostic, not a test of the original PF goal-planning claim.

Use the same eight PF sessions, inferred Home metadata, fixed spatial mask and
4,001 source-high-MUA candidate cores as the prior content intervention. Do not
redetect events, optimize replay outcomes, or relax gates after seeing them.

## RUN Partitions and Matching

Split the supported RUN clock span into temporal quarters. Fit/obtain maps from
the first half only. Estimate field stability by first-quarter vs second-quarter
maps, on their common occupied states. Frozen original RUN-QC cell eligibility
is retained, additionally requiring finite early stability and a positive early
rate. Original cell eligibility was established using the full RUN recording;
the new subset choice and maps for held-out decoding use the early half only.

The third quarter is the matching set. The fourth is an untouched confirmation
set for the chosen pair; NEVER reselect after confirmation failure. Use up to
512 evenly spaced, non-overlapping 250 ms windows per block, position tracking
gaps <=100 ms, animal speed 10-200 cm/s throughout. Require >=100 windows in each.
Decode each window with the early-half rate maps and a uniform prior over the
fixed support. Position truth is tracked position at the window midpoint.

Population descriptors: mean early RUN rate, median early field stability, median
half-peak field area. Spatial coverage score is the population mean of each
unit's mean rate within 20 cm of Home divided by its mean rate over valid states.
This measures relative Home tuning, not a count of distinct covered bins.

Per session create 64 high-Home and 64 low-Home weighted random subsets, 64 random
subsets, and up to 256 unique subsets retaining entire tetrodes. Weighted subsets
have floor(eligible cells/2) units; weights are exp(+/-1.5*z(Home tuning)), with
z clipped to [-3,3]. Random subsets have the same count. Whole-tetrode subsets
retain a random half of tetrodes and are not trimmed to force a unit count;
their two matched populations must nonetheless have identical counts, each in
[35%,65%] of eligible cells. All keep/remove decisions concern eligible units.

Pair matching gates, all predeclared:

- Exactly equal unit counts, at least 20 cells.
- Mean rate relative difference <=10% (difference / pair mean).
- Median field stability absolute difference <=0.05.
- Median half-peak area relative difference <=20%.
- RUN posterior-mean median error difference <=2 cm; p75 difference <=5 cm.
- Both populations' median error <=20 cm and p75 <=40 cm.

Choose ONE high/low pair maximizing RUN-derived Home tuning contrast among
matching pairs; coverage ratio must be >=1.5. Choose one whole-tetrode pair by
the same rule. The random reference is the FIRST qualifying pair in the seeded
candidate order and requires no coverage contrast. Orient all pairs high-minus-low
using early Home tuning, not replay data. Record all candidates and criteria.

Freeze the selected cell IDs BEFORE evaluating fourth-quarter confirmation.
Apply the same RUN error gates on confirmation. No tuning or alternate pair
selection if it fails. Main replay comparisons require confirmation pass.
If too few pairs pass, report feasibility limitations; do not claim quality is
matched merely because matching-set scores agree. Do not extrapolate a result
from one rat to the dataset.

## Replay Readout

After all RUN selection artifacts are frozen and hashed, load original-order
replay inputs. Reuse the previous full-population fixed candidate endpoint and
full-accepted continuous-segment endpoint timestamps. Flat-prior Poisson, 20 ms
windows, common 8 cm maps. Primary uses frozen full-RUN rate maps for comparability;
early-half encoding is a predeclared sensitivity. No HMM or trajectory prior.

Primary: high-minus-low mean Home posterior mass at fixed candidate endpoints.
Also report MAP Home fraction, mean-estimate separation, and observed endpoint
spike/active-cell counts. Matching is on RUN rates, not on replay spike counts.
Analyze the frozen 816 full-accepted events separately at their original segment
endpoints. Do not classify/reselect replay events with the new populations.

Average events within session, sessions within rat, then equal rats. Report per-rat
effects and 5,000 descriptive rat-cluster bootstrap intervals. With <=4 rats these
are not a general population guarantee. Keep engineered high/low subsets, random
reference and whole-tetrode subsets separate. Adversarially targeted samples
demonstrate possibility, not the prevalence of bias in normal recordings.

## Claim Boundary and Stop

Home labels and locations remain inferred from behavior, not author-verified.
A Home-mass contrast is not a planning contrast against behavior-matched goals.
We can test whether global RUN quality matching protects content readouts; we
cannot conclude that the real neural representation changed or that published
goal effects are artifacts. An unsuccessful match is an informative failure of
this bounded experiment, not permission to weaken the protocol.

Run on gpuserver6000 with detached service, persistent user manager and session
checkpoints. Record code and exact input hashes. No additional dataset or broad
model-scoring campaign is authorized by this protocol.
