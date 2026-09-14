# Count-conditioned content outcome: general remedy screen failed

2026-09-14. This preserves a negative prospective screen; no thresholds, event
times, cell masks, rate-map means or partition were changed after the outcomes.

## Provenance and checks

Producer/protocol `0d8db777`, independent auditor `427f96c9`, reporter `1db47210`.
All scientific computation ran on gpuserver6000 as detached user systemd services;
PF, hc11, audit and report completed with status0. Each dataset has1600frozen
real candidates, eight sessions and four rats; three partitions, split0primary.
Sources are the six prior audited real/RUN/fixed-total sources plus independent
unconditional Poisson stationary controls at gain1 and gain20.

Independent reconstruction passed for16/16recordings,229473readout rows,
458946population posteriors,6297native real/RUN endpoint recounts and
521600unconditional Poisson count entries. All entropy controls were available.
A separate source-table recomputation verified2880summary rows' before/after
values and paired differences at1e-10tolerance. Fifty-five focused tests passed;
Ruff and whitespace checks passed. Figure inspected for labels, axes and overlap.

## Real candidate results

Equal event weights within sessions, sessions within rats, then equal rats.
Four-rat intervals are descriptive and do not provide precise population proof.

| Dataset | Readout | Poisson | Conditional count | Entropy-matched Poisson |
| --- | --- | ---: | ---: | ---: |
| hc11 | A/B posterior-mean separation cm |28.794627|27.092060|26.824050|
| hc11 | regional3x3 posteriorTV |0.319761|0.300981|0.300807|
| PF | A/B posterior-mean separation cm |45.449571|45.119448|44.914853|
| PF | regional3x3 posteriorTV |0.509727|0.499991|0.502603|

hc11 separation decreases5.91%, regionalTV5.87%, both positive4/4rats, but both
miss the frozen10%target. Separation delta=-1.702567cm,
CI[-2.567781,-0.837354]; regionalTVdelta=-0.018780,
CI[-0.026494,-0.011067]. Entropy-matched Poisson does at least as well on both:
conditional-minus-entropy separation+0.268010cm, CI[-0.245709,+1.011824];
regionalTV+0.000174, CI[-0.002637,+0.002984].

The two alternative hc11 partitions reduce separation10.47%/6.29%, but regional
TV only7.26%/6.61%. Neither meets both requirements; no partition replaces
split0. PF changes are small: separation-0.73%, regionalTV-1.91%.

## Why agreement is not sufficient

hc11 has294both-silent and856one-silent primary endpoints;450have spikes in both
populations. Equal-rat fractions are18.3%,54.5%,27.1% respectively. Only7/1600
have >=3spikes and>=2active cells in BOTH populations. In a silent population,
conditional decoding is exactly uniform, not a spatial measurement.

Descriptive hc11 separation changes within fixed activity strata are:
both-silent -5.032cm, one-silent -1.540cm, both-nonzero +0.448cm. Thus the overall
disagreement decrease does not demonstrate better agreement among independently
observed, spiking populations. These conditional strata are not a replacement
selection rule or a primary validation analysis. Empty strata remain undefined.

## Known positions expose the trade-off

| Dataset | Control | Mean A/B error change cm | Mean regional Brier change |
| --- | --- | ---: | ---: |
| hc11 | held-out RUN |-0.051|-0.00057|
| PF | held-out RUN |+0.184|+0.00223|
| hc11 | Poisson stationary, gain1 |+0.588|+0.00495|
| PF | Poisson stationary, gain1 |+0.337|+0.00348|
| hc11 | Poisson stationary, gain20 |-7.431|-0.14371|
| PF | Poisson stationary, gain20 |-5.263|-0.10068|

All known-source/population error upper95bounds meet the2cmnoninferiority
tolerance. However, both datasets fail the mandatory nonworsening simulated
Brier condition on ordinary matched-Poisson observations; PF also fails the
nonworsening RUN error/Brier conditions. The gain20 improvement demonstrates
the expected benefit of ignoring a misscaled total rate, NOT that real replay
has gain20, or that population gain mismatch explains the original content bias.

Conditioning is invariant to global gain and discards spatial information in
total counts. The trade-off is mathematical and empirically reproduced here;
neither universal use of Poisson nor universal count conditioning is validated.

## Diagnostic screen

A-only total-count reliance is theTV difference between that population's two
posteriors. Its mean within-session/rat rho with real regional disagreement is
+0.111hc11 and+.058PF, but with real mean separation it is+.010/-0.004.
Against independent B held-outRUN location error, rho is+.008hc11 and+.001PF,
each positive only2/4rats. Other true-error correlations vary by simulation.
This does not supply a reliable held-out localization-error diagnostic or a
validated selector. No threshold was fitted.

## Decision

The original objective remains UNACHIEVED: no validated remedy or diagnostic
has yet been shown to predict and reduce the original regional-content
instability in independent data. Do not promote a uniform-posterior agreement
gain, simulated high-gain benefit, or alternative partition into that claim.

Raw random-half endpoints remain a different assay from the original matched
Home/accepted-trajectory endpoint contrast. This failed screen must not be used
to dismiss that original result or to claim its cause is settled. Further work
should distinguish recovery of an aggregate content distribution from prediction
of individual sparse endpoints, and directly retain the original frozen
matched-population comparison. No new biological claim follows here.

## Artifact locations

Server: `/mnt/seagate10tb/florianpfaff/count-conditioned-content-20260914/`.
Local compact copy:
`/mnt/c/Users/emper/Documents/codex/2026-09-14/count-conditioned-content-benchmark/`.
Subfolders `audit` and `report`; large count/posterior arrays remain on server.
