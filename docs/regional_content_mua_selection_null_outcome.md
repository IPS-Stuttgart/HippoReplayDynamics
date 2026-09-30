# Fixed-tuning selection experiment: outcome

2026-09-15. Executed on gpuserver6000; no new real replay scoring.

## What was tested

Can the original high-MUA detector and endpoint rule break regional-content
calibration when the spatial tuning maps are fixed and represented location is
known? Eight PF recordings, four rats, four 600-second replicas at each of
three frozen shared gain settings (1, 3, 6). Detector uses all recorded cells;
decoder uses the original RUN-qualified cells. All represented positions are
constant within two-second epochs. This is a regional-content null, not a
simulation of moving replay trajectories.

Independent gain-one synthetic calibration uses the same class-conditional
position distribution as the generator. The observed-RUN calibration is a
separate sensitivity analysis; it includes simulator-versus-real-RUN mismatch.

## Main result

With model-consistent calibration and the frozen Poisson readout:

| Peak gain | Readout cohort | Incompatible runs | Mean mismatch TV | Mean absolute prevalence error |
|---|---|---:|---:|---:|
| 1 | Fixed windows | 0/32 | 0.023 | 3.21 percentage points |
| 1 | Detected endpoints | 20/32 | 0.119 | 5.73 percentage points |
| 1 | Independent counts at those endpoints | 7/32 | 0.041 | 7.28 percentage points |
| 6 | Fixed windows | 25/32 | 0.087 | 2.65 percentage points |
| 6 | Detected endpoints | 32/32 | 0.168 | 2.23 percentage points |
| 6 | Independent counts at those endpoints | 28/32 | 0.116 | 2.56 percentage points |

The 32 runs are eight recordings times four Monte Carlo replicas, not 32 animals.
Metrics average replicas within recording, recordings within rat, then rats.
Incompatibility refers to the frozen zero-transfer-slack bootstrap compatibility
check, not a certified frequentist hypothesis test or coverage guarantee.

Selection increases mismatch relative to independent counts at the same times,
states and gains in all eight recordings at gain 6. Therefore conditioning on
the decoded spikes contributes, over and above selected timing and locations.
Those independent counts still share the detector-selected latent location
distribution, which explains why they are not an unconditional control.

At gain 6, AUC is 0.953 for selected endpoints versus 0.930 for fixed windows.
The calibrator can become incompatible while discrimination improves and the
estimated prevalence stays near truth. These are distinct failure/readout axes.
At gain 6, the observed-RUN calibrator also fails compatibility in 32/32 selected
runs, but absolute prevalence error is 5.93 percentage points, not 2.23.

Oracle gain in the likelihood does not repair the frozen gain-one category
calibration (31/32 selected runs still incompatible for the synthetic calibrator).
The oracle does not condition on event detection or recalibrate sensitivity.

## Conclusion and boundary

Activity gain and selection can reproduce calibration-transfer failure without
changing spatial tuning. Even the gain-one condition demonstrates an effect of
selection without any gain change. Thus the earlier real RUN-to-MUA mismatch
does not establish replay-specific tuning drift, contradictory representations,
or incorrect regional prevalence. The null also does not establish that all
real mismatch is explained by selection, or validate the latent-class proposal.

Next discriminating experiment: estimate calibration through the same detector
and endpoint rule, using independent simulations; validate on untouched replicas
with known truth. Check whether both readout compatibility and truth-covering,
informative bounds recover before attempting real replay-content calibration.
This next experiment has not been run here.

## Verification and locations

- 44 relevant tests pass; new code passes Ruff.
- Independent audit passes all 96 simulated trains, 27,322,061 spikes and
  17,528 selected endpoints. Detector rerun, independent recounts, unchanged
  maps, exposure integrals, independent draws, likelihoods and hashes verified.
- Raw: `/mnt/seagate10tb/florianpfaff/regional-content-mua-selection-null-pf-20260915`
- Report: `/mnt/seagate10tb/florianpfaff/regional-content-mua-selection-null-pf-report-20260915`
- Audit: `/mnt/seagate10tb/florianpfaff/regional-content-mua-selection-null-pf-20260915-independent-audit.json`
- Code: `/home/florianpfaff/HippoReplayDynamics-content-stability-20260914`

New files remain uncommitted. Measurement manifest records base commit, dirty
state, command, parameters and exact input/source hashes; commit alone is not
sufficient to identify this run. All raw synthetic spikes remain on the server.
