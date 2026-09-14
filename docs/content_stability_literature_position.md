# Literature position: a controlled limitation, not yet a validated remedy

Checked2026-09-14. Priority/novelty is not proven by this targeted literature check.

## Closest statements in the literature

1. van der Meer, Carey and Tanaka(2017), *Optimizing for generalization in the
   decoding of internally generated activity in the hippocampus*, Hippocampus
   27:580-595, https://doi.org/10.1002/hipo.22714.
   They advocate cross-validation where covert activity has no known truth and
   show that validation design affects errors, optimal settings and spatial
   error distributions. Therefore neither decoder-QC nor spatially heterogeneous
   decoding error is new. They do NOT promise that one global error statistic
   certifies the content of replay.

2. Ciliberti, Michon and Kloosterman(2018), *Real-time classification of
   experience-related ensemble spiking patterns for closed-loop applications*,
   https://doi.org/10.7554/eLife.36275,
   https://elifesciences.org/articles/36275.
   Figure6 progressively removes tetrodes and evaluates replay identification
   against a fixed full-data reference, alongside cross-validated RUN accuracy.
   RUN accuracy and replay identification change differently; the Results and
   Discussion explicitly state that low RUN error is not sufficient for good
   replay-content identification. This is an empirical precedent, not merely a
   theoretical warning. Our distinction is equally sized populations matched on
   global RUN descriptors/accuracy, with a regional posterior-content contrast
   at unchanged event times. Neither the broad insufficiency of RUN QC nor
   neuronal-subsampling sensitivity is new.

3. van der Meer, Kemere and Diba(2020), *Progress and issues in second-order
   analysis of hippocampal replay*, https://doi.org/10.1098/rstb.2019.0238,
   https://pmc.ncbi.nlm.nih.gov/articles/PMC7209917/.
   They explicitly identify unequal place-cell and behavioral sampling as causes
   of replay-content bias, recommend spatial decoding-error diagnostics and
   generative controls, and propose accounting for uncertainty in estimated
   firing rates instead of only plugging in their means. Thus the general bias
   concern, simulations and uncertainty-aware likelihood idea already exist.

4. Takigawa et al.(2024), *Evaluating hippocampal replay without a ground truth*,
   https://doi.org/10.7554/eLife.85635,
   https://pubmed.ncbi.nlm.nih.gov/39606951/.
   Sequence-based detection is evaluated using independently formulated track
   discriminability in a two-track paradigm. Absence of replay ground truth and
   the need to validate detectors are not new. Our question is different: how
   stable is a spatial-content readout of the SAME event under a changed recorded
   population? Neither measure automatically supplies true replay destinations.

5. Pfeiffer and Foster(2013), *Hippocampal place cell sequences depict future
   paths to remembered goals*, https://pmc.ncbi.nlm.nih.gov/articles/PMC3990408/.
   Their Home-overrepresentation analysis explicitly includes occupancy and
   place-field-distribution controls (Supplementary Figures14-15), as well as
   vectorized-trajectory and behavioral-path comparisons. Our candidate endpoint
   statistic is not a reproduction of that full inferential chain. Changing a
   population changes its observations; without internal replay ground truth,
   disagreement does not establish which readout is wrong or overturn their
   published planning interpretation.

## Narrow empirically demonstrated addition

In the audited PF matched-population experiment, cell subsets had the same size
and matched RUN firing rate, field area, stability and global held-out accuracy,
yet gave different posterior probabilities near inferred Home at fixed event
endpoints. Candidate difference+8.58 percentage points; previously accepted
trajectory-segment endpoint difference+6.87points. Successful targeted matches
cover4/8sessions and3rats; failed matches were retained as failures, not replaced.
See the source experiment `pf-matched-population-content-20260913` and its audit.

This is an empirical counterexample to the sufficiency of GLOBAL matched decoder
QC for stable REGIONAL replay-content estimates. It quantifies an already
recognized concern under controlled cell-subset comparisons. It is not evidence
that any particular decoder has recovered the animal's true internal destination,
not a refutation of published goal-directed replay, and not proof of a new
biological mechanism. Local Home accuracy and replay spike support were not
matched; population contrasts were deliberately maximized and Home was inferred.
In particular, the2018 result means the contribution cannot be framed as the
first demonstration that good behavioral decoding fails to certify replay.

Call this a rigorously audited, bounded demonstration, not proof of universal
novelty. Establishing first-in-literature priority would need a broader review of
population-subsampling and content-bias controls, especially paper supplements.

## What the new uncertainty test contributes

The prospective fixed-time uncertainty experiment directly tests one practical
implementation of the2020 suggestion. On hc11, the primary Gamma-Poisson method
reduced separation and regional disagreement by only~0.7%; most of the small
gain was matched by likelihood tempering to the same entropy. Known-position
controls and independent likelihood reconstruction prevented treating mere
agreement as success. It failed the frozen10% remedy screen.

This is useful negative validation of THIS estimator, not a disproof of the
literature's general proposal. A method that predicts and reduces regional
instability on independent data without harming true localization would be a
stronger methodological advance. That result has NOT been demonstrated yet.
