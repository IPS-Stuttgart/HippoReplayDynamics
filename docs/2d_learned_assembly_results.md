# Learned Assembly Comparison: Completed Diagnostic

The all-session experiment completed on gpuserver6000, producer commit
`d54009a1cc3becccfa1b3ebeea2c3460a1062ce2`. Authoritative output:
`/mnt/seagate10tb/florianpfaff/2d-learned-assembly-all9225-k20-v2-20260908`.
Manifest SHA256:
`a8507a6db14d07e7e6f8d5d709a571da949c9394a6c31e2ce8f8e54c88f7efa6`.
The initialization addendum documents the terminal first-run failure and
hash-verified reuse of 485 jobs; the ten missing jobs were completed separately.

All 9,225 frozen MUA candidates, 33 sessions and nine animals were retained.
All 495 fits converged. The separate reconstruction audited 2,905,875 score
rows and independently recomputed 1,245,375 predictions, maximum error
1.42e-13. This covers original and two shuffled orders per event/split/capacity,
not independent re-inference of every null or an independent optimizer.

## Primary K50 Results

Event medians across five neural partitions, then equal-session/animal means.
Intervals are the frozen hierarchical bootstrap, not additional independent
animals or correction for all exploratory analysis choices. Scores are proper
count-conditioned bin-marginal held-out predictive scores, not joint logZ.

| Contrast | PF mean [95% CI], positive animals | Tanni mean [95% CI], positive animals |
| --- | --- | --- |
| Learned HMM minus calibration global | +12.179 [8.056, 17.651], 4/4 | +2.358 [0.684, 4.142], 4/5 |
| Learned HMM minus same-emission IID | +0.988 [0.867, 1.115], 4/4 | +0.272 [-0.099, 0.655], 4/5 |
| Learned HMM order advantage | +1.383 [1.176, 1.577], 4/4 | +1.046 [0.586, 1.642], 5/5 |
| Spatial IMM minus learned HMM | -0.232 [-1.024, 0.771], 1/4 | +1.305 [0.293, 2.320], 4/5 |
| Spatial minus assembly order advantage | -0.512 [-0.589, -0.431], 0/4 | -0.517 [-0.984, -0.173], 0/5 |

The PF spatial advantage does not survive comparison with a properly learned
assembly model as a rat-uniform positive. This is not proof of equivalence or
absence of spatial content: the assembly model can learn spatial relations
without coordinates. In Tanni, spatial prediction is better on average, but
the comparator's global-baseline adequacy is not animal-uniform. The frozen
adequacy gate therefore prevents a strong spatial-specificity interpretation.
K20/K100 sensitivities remain reported, not substituted for the primary.

Learning temporal assemblies from population burst events, including PF, has
direct precedent in [Maboudi et al. (2018)](https://elifesciences.org/articles/34467).
This result does not establish a new replay mechanism, uniform physical speed,
replay prevalence, or a completed high-importance paper. It closes another
overly broad interpretation of the earlier spatial-versus-global comparison.

The non-rescoring report, figure, capacity/fit diagnostics and manifest are in
the corresponding `-v2-report-20260908` directory. The independent audit is in
`-v2-audit-20260908`. No events were rescored for this note.
