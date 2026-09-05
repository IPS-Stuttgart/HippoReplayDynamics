# Novelty and Claim Boundaries

This is a targeted literature check, not an exhaustive novelty certification.
The study must add a replay-specific quantitative result or validated procedure;
neither decoder bias nor uncertainty miscalibration is a new general observation.

## Direct Prior Work

- Silva, Feng, and Foster (2015),
  [Trajectory events across hippocampal place cells require previous experience](https://pmc.ncbi.nlm.nih.gov/articles/PMC6095134/),
  Figure 5, adds noise to saline place-field decoders to match or worsen RUN
  reconstruction relative to a drug condition, then reassesses trajectory
  events. Deliberately degrading a decoder to assess a replay conclusion is
  therefore not new. This experiment differs from hiding recorded cells, but
  is direct precedent for the measurement-control logic.
- Liu, Todorova, Tang, Oliva, and Fernandez-Ruiz (2023),
  [Associative and predictive hippocampal codes support memory-guided behaviors](https://pmc.ncbi.nlm.nih.gov/articles/PMC10894649/),
  Methods replay analysis and Figure S7C, progressively removes units from
  stimulation-OFF decoding until RUN errors exceed those in stimulation-ON
  decoding, then recomputes replay. Real-unit removal followed by replay
  reassessment is itself established practice, not a new intervention invented
  here. This is a stronger novelty constraint than the generic observation
  that fewer cells worsen localization.
- Wei, Tajik Mansouri, Wang, and Stevenson (2024),
  [Calibrating Bayesian Decoders of Neural Spiking Activity](https://doi.org/10.1523/JNEUROSCI.2158-23.2024),
  demonstrates neural-decoder overconfidence, including hippocampal position
  decoding, and examines latent-variable models and post-hoc calibration.
  It also studies changing population size. Rediscovering low empirical
  coverage of nominal credible regions would not be a sufficient novelty claim.
- Takigawa et al. (2024),
  [Evaluating hippocampal replay without a ground truth](https://elifesciences.org/articles/85635),
  evaluates replay detection using a distinct track-discriminability measure
  and randomized controls. Replay validation without ground truth is therefore
  not itself a new framework proposed by this study.
- Denovellis et al. (2021),
  [Hippocampal replay of experience at real-world speeds](https://elifesciences.org/articles/64505),
  explicitly discusses temporal discretization, uncertainty, and selection
  assumptions when inferring replay dynamics. A general criticism of fixed
  time bins or continuous-only selection would likewise be insufficient.
- Ji et al. (2026),
  [Dynamical modulation of hippocampal replay through firing rate adaptation](https://doi.org/10.1038/s41467-025-68042-3),
  explicitly notes that limited, uneven place-field coverage can create apparent
  jumps (Methods, discussion following Eq. 37). It also examines spike count
  and field-size covariates while interpreting replay step sizes and diffusivity.
  Therefore, neither the coverage-to-jumps hypothesis nor asking whether spike
  count confounds decoded dynamics is new. Our controlled removal/pooling and
  injected-gradient recovery experiments are a different validation strategy;
  they do not refute that paper's mechanisms or results without matched tests.
- Bakermans et al. (2025),
  [Constructing future behavior in the hippocampal formation through composition and replay](https://doi.org/10.1038/s41593-025-01908-3),
  uses memoryless Poisson decoding in 20 ms windows stepped by 5 ms and
  leave-one-neuron-out decoding to avoid contaminating spike-localization
  estimates with that neuron's own rate map. It validates first-half maps on
  second-half running. Neither cell exclusion to prevent circularity nor RUN
  validation is itself new here. This is a useful published pipeline comparator,
  distinct from our population-removal recovery/selection experiment.
- Huh, Yun, Lee, and Jung (2026),
  [A likelihood-based method for identifying replay from spike sequences](https://doi.org/10.1038/s41467-026-74822-2),
  published 4 July 2026, scores pairwise firing-order statistics learned during
  behavior and evaluates candidate sequences against cell-identity shuffles.
  Its simulation and experimental comparisons include conventional decoded
  time-position correlation. Thus neither a template-free likelihood detector
  nor demonstrating parameter sensitivity would be a new claim here. This is
  a potentially useful sequence-detection comparator, but it does not directly
  estimate physical speed or supply a calibrated spatial-speed equivalence test.
  Its applicability to freely varying 2D paths needs evaluation rather than
  assuming a trial-template comparison transfers unchanged. This targeted check
  was refreshed on 2026-09-05; no head-to-head benchmark has yet been run.

## Candidate Contribution and Priority Boundary

Connect recording coverage, uncertainty, geometric continuity selection, and
speed-gradient recovery in the SAME observed events and in ground-truth
simulations matched to two independent 2D datasets. Quantify both incorrectly
retained discontinuous/static events and incorrectly rejected continuous paths.
Separate spatial cell coverage from total spike information, and report what
the selected subset does to apparent speed profiles.

A useful output could be an independently validated calibration/abstention
procedure: identify recording and analysis settings where a biologically
meaningful speed gradient can be recovered, and refuse uniformity claims where
it cannot. Such a procedure must outperform or materially clarify established
baselines; simply producing another threshold is not enough.

The real-cell subsampling result and RUN validation alone are prerequisites
and supporting experiments. The completed recovery/selection, calibration,
independent-map, excluded-animal and nested-budget benchmarks now connect the
full measurement chain, with conditional successes and explicit failure limits.
This does not establish priority for the combined contribution or make the
baseline calibration a new universally valid method.
In particular, the 2015 degraded-decoder and 2023 real-unit-removal controls
mean that paired perturbations alone cannot carry a priority claim. The present
study must establish the additional quantitative connection to spatial-speed
recovery, selection and calibrated inference limits.
In particular, do not claim that a cell-count correlation or a population-size
perturbation alone establishes a new explanation of replay. The stronger target
is a tested operating range for recovery of continuity and spatial speed
variation, including explicit abstention when the recording cannot distinguish
large gradients from uniform speed. Comparison to published detection and
calibration methods remains required before claiming a new validated procedure.
The current project cannot claim biological uniformity, a neuronal mechanism
for constant replay speed, or that recording coverage explains all PF/Tanni
differences. Negative or mixed recovery results must be retained.
