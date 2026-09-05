# Speed-Interval Calibration and Transfer Test

Frozen before production. This is a calibration baseline and identifiability
experiment, not a new conformal method or permission to claim biological
uniformity. Study requirements remain those in the original completion matrix.

## Estimand and Cases

The synthetic speed field is v(x)=1000*(1+g*q(x)) cm/s, q in [-1,1] across the
horizontal simulation domain. The primary meaningful-equivalence band is
|g|<0.25: up to +/-25% of the center speed, or a 1.67-fold end-to-end ratio
at the boundary. Report 0.10 and 0.50 sensitivity bands too. These are declared
benchmark scales, not empirically established biological tolerances. This
coordinate is not distance from a PF arena wall.

Use all 33 sessions and the first-half-A/second-half-B direction from the frozen
independent-map benchmark. A alone selects units, fits decoder rates and defines
state support. Reconstruct its full grid for the calibration generator, checking
that rates on all decoding states match the saved A model. B is the disjoint
RUN-half generator, never used to fit the inverse calibration. The two halves'
position extents already constrain the synthetic domain; this is not a strict
held-out biological position test. Neural populations are empirical; generated
paths and spikes are surrogate truth. No new biological animals are held out.

Every panel has one new path for each of that session's frozen source duration
profiles (up to 30, fewer where already recorded). Do not select source profiles
on recovered continuity. Reuse every panel's path across the four test conditions:
A/Poisson, A/shared-gain, B/Poisson, B/shared-gain. Counts differ across generators
and observation conditions; all are decoded with A. All rates are scaled by 3.
Shared gain has mean 1, CV=1, constant 20 ms blocks, independent of position.

## Independent Draw Sets

Root seed 20260914. Derive session, phase, draw and event seeds with the existing
stable seed helper. Train inverse calibrations on 40 A/Poisson panels with g
uniform on [-0.75,0.75]. Calibrate on 99 entirely new A/Poisson panels from that
same distribution. Test 100 new uniform-g panels plus 20 new panels at each
g=-0.5,-0.25,0,0.25,0.5, in each of the four test conditions. Panel IDs, parameters
and seeds are frozen and recorded; no path redraws based on outcomes. Reverse
chronology is not included in this initial calibration test and must not be
claimed. An explicitly capped two-session run is technical smoke only.
Session workers use independent, ID-derived seeds and single-threaded BLAS;
worker count must not change numerical results. Verify sequential/parallel
smoke parity before production. The production worker count is recorded and
bounded according to available server resources, without altering draw counts.

## Observable Statistic and Comparators

Independent flat-prior Poisson decoding, full A-selected cells, 8 cm grid, 20 ms
windows/5 ms stride. Compute speed from nonoverlapping 20 ms windows without
bridging unsupported bins. No HMM, trajectory prior or temporal smoothing.
Use equal-event moments to regress normalized decoded speed on decoded q, not
on true q. Require at least five contributing events and spatial variance 0.01.
Report MAP/posterior mean, unfiltered/2-cells-3-spikes, before selection/within
the selected continuous run separately. The primary inferential target is the
selected posterior-mean readout with bin support; all-data readouts diagnose
whether continuity selection itself destroys availability.

Three prespecified methods:

1. Raw decoded slope with 200 equal-event percentile-bootstrap draws (95%).
   Failed resamples yield an unbounded interval, not a silently reduced sample.
2. OLS inverse g=a+b*T from training panels, Gaussian 95% prediction interval.
3. The same inverse point estimate with split-conformal absolute residuals from
   the independent calibration panels. Radius is order statistic
   ceil((n_cal+1)*0.95), or infinity if that rank is unavailable. Missing
   calibration statistics contribute infinite residuals; they are not removed.

At least 20 finite training panels are required. Missing test statistics or
uninformative fits abstain with unbounded intervals. Do not clip intervals to
the simulated g range, which could manufacture equivalence. No tuning against
test gradients, B-map outcomes or real replay speed correlations.

Conformal coverage is marginal over exchangeable A/Poisson panels with the
declared uniform g distribution, conditional on the fixed learned map. It is
not a guarantee at each fixed g, under B maps/shared gain, across new animals,
or for real replay. Test those departures explicitly. The fixed-g and B/gain
tests are transfer/conditional stress tests, not exchangeable calibration data.
References: Shafer and Vovk (2008), https://jmlr.org/papers/v9/shafer08a.html;
Wei et al. (2024), https://doi.org/10.1523/JNEUROSCI.2158-23.2024.

## Reporting and Verification

Retain every planned test panel, including missing statistics. Report coverage
among all panels and among finite intervals separately, interval width, finite
availability, nonzero-claim sensitivity, and false/true equivalence at each band.
Boundary g=+/-band is NOT inside equivalence. An infinite interval covering all
truth is not successful informative inference. Report fixed-g denominators and
conditional Monte Carlo uncertainty; average sessions within animal and animals
equally, with animal-bootstrap intervals (only 4 PF and 5 Tanni animals).

Save fitted inverse parameters, calibration residual-radius inputs, all panel
statistics and test decisions, source/model/code hashes, runtime, exclusions
and frozen draw schedules. Technical gates require positive planned counts,
all required model/readout/condition rows and unchanged inputs, never positive
biological results. Reproduce every interval/decision from saved panel data and
independently reconstruct a declared sample of simulation/decoding panels in
every session. Hash verification is not a substitute for numerical checks.

Real-data application remains blocked until transfer, event-definition and
published-baseline comparisons justify it. This test can legitimately conclude
that calibration is informative only under its own generator and does not
transfer; it must not force a favorable equivalence verdict.
