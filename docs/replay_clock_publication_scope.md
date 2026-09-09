# Replay Clock Inference: Publication Scope

Literature and claim-boundary check, 2026-09-10. This note does not change the
frozen fresh-path calibration protocol or authorize biological scoring.

## What Is Already Established

Forli and colleagues report that longer flight trajectories have faster
replays without correspondingly longer replay durations, alongside larger and
more widely separated place fields. Their sequence-based speed measure combines
cell recruitment rate with physical separation. Thus linking replay timing,
field spacing and physical distance is not itself a new idea.
[Forli et al., Nature 2025](https://doi.org/10.1038/s41586-025-09341-z).

Parra-Barrero and Cheng explicitly model internally generated sequences with
a fixed neuronal pace and a learned, nonuniform mapping onto physical space.
Their work concerns theta sequences and behavior-dependent spatial maps; it is
not a demonstration that our 2D replay candidates follow either proposed clock.
[Parra-Barrero and Cheng, PLOS Computational Biology 2023](https://doi.org/10.1371/journal.pcbi.1011101).

These papers motivate the question but rule out claiming conceptual novelty
for physical-versus-neuronal timing alone. This is a targeted check of close
precedents, not an exhaustive novelty review.

## What Our Current Test Can Establish

The fresh-path experiment tests whether two idealized temporal laws can be
distinguished after integrating over unknown 2D paths, while conditioning on
native event spike totals and retaining stationary/reset alternatives.
This is an inference-calibration question. It does not establish a biological
mechanism, and its matched simulator/encoder/prior is a best-case condition.

The recorded-population code uses Hellinger distance between normalized cell
firing-rate vectors. It is not a measurement of anatomical propagation across
the hippocampus, the complete neural population, or a literal neural sheet.
Cell sampling and uncertainty in RUN maps can change this metric.

Reusing a finite simulation path library and approximating a path integral
are different sources of error. The previous exhaustive run removes the
latter but not the former. Demonstrating this distinction would be useful
validation evidence; the underlying statistical principle is not a new
neuroscientific discovery.

## What Would Make The Biological Question Credible

1. Recover known timing laws across the relevant recording encoders, with
   fresh paths, independent RUN-map estimates and observation-model stress
   tests. Report useful power as well as false positives and interval coverage.
2. Freeze a real-data analysis using a candidate set chosen without the clock
   evidence. Validate prediction on held-out neural observations and assess
   all animals, rather than select successful encoders or event classes.
3. Compare flexible temporal warps as well as the two idealized clocks. A
   preference for constant physical speed over one neural-clock alternative
   does not exclude other forms of spatially varying speed.
4. To claim near-uniformity, declare a scientifically meaningful tolerance
   and show that an uncertainty interval lies within it, with calibrated power
   to detect meaningful deviations. A nonsignificant difference is insufficient.
5. Test sensitivity to the recorded-cell subset and encoder uncertainty. An
   apparent code-space effect that depends on which cells were recorded is
   not automatically a property of the full biological network.

The potential contribution is a calibrated distinction between competing
timing laws in independently recorded populations, or an informative limit
on what those recordings identify. Neither has yet been established as a
publication-ready result by this two-encoder calibration pilot. The broader
publication goal remains open.
