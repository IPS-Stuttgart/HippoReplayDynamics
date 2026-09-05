# Novelty and Claim Boundaries

This is a targeted literature check, not an exhaustive novelty certification.
The study must add a replay-specific quantitative result or validated procedure;
neither decoder bias nor uncertainty miscalibration is a new general observation.

## Direct Prior Work

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

## Candidate Contribution, Still To Be Demonstrated

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

The real-cell subsampling result and RUN validation are prerequisites and
supporting experiments. They do not yet demonstrate the complete contribution.
In particular, do not claim that a cell-count correlation or a population-size
perturbation alone establishes a new explanation of replay. The stronger target
is a tested operating range for recovery of continuity and spatial speed
variation, including explicit abstention when the recording cannot distinguish
large gradients from uniform speed. Comparison to published detection and
calibration methods remains required before claiming a new validated procedure.
The current project cannot claim biological uniformity, a neuronal mechanism
for constant replay speed, or that recording coverage explains all PF/Tanni
differences. Negative or mixed recovery results must be retained.
