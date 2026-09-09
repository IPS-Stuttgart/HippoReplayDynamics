# Paper Assessment: A Measurement Result, Not Uniform Replay Speed

2026-09-10. This is an evidence-based authoring recommendation. Journal
acceptance and absolute publication priority cannot be certified by an
analysis pipeline or a literature search.

## Recommended Insight

**Recording-dependent trajectory detectability, independent neural information,
and identifiable physical kinematics are different properties. A replay-speed
conclusion needs validation of the whole measurement chain, not just more
accepted trajectories or a weak speed-position correlation.**

I judge the quantitative evidence sufficient to develop a methods paper around
that distinction. This is a narrower scientific claim than biological uniform
speed, but it is not merely the observation that fewer cells worsen decoding.
It combines real-event perturbations, independent neural validation and
known-kinematics recovery, with positive operating regimes and failure limits.

Working title: **Recording coverage shapes estimates of replay continuity and
spatial speed variation**.

## Why It Is Worth A Paper

1. The same real population events change classification when neurons are
   hidden. Under a transferred PF-style continuity plus two 5,000-shuffle
   criterion, equal-animal high-MUA acceptance falls from 22.19% to 8.53% in
   PF and 5.77% to 1.70% in Tanni. The primary change is negative in all nine
   animals, also in the ripple cohorts. The event itself was not changed.
2. A separate training-only geometric screen rejects events that, as a group,
   still predict other neurons in PF. All five predictive contrasts pass in
   all four animals, including the nonspatial composition comparator and
   the order-by-map interaction. Tanni does not pass the full comparator set.
   Thus rejection is not interchangeable with absence of neural structure,
   but neither predictive success nor full-cell acceptance provides replay truth.
3. Increasing the number of continuous-looking decoded paths is not a
   sufficient validation target. In the frozen large-arena surrogate, changing
   20 ms windows to 40 ms raises genuine continuous-path recovery from 11.34%
   to 64.34%, and randomized-path acceptance from 0.52% to 18.23%. The
   observations are unchanged; the measurement changes.
4. This selection problem matters for the original kinematic question. In
   independently estimated RUN-map simulations, a true positive-minus-negative
   gradient contrast of 1.0 yields decoded responses of 0.321 PF and 0.121
   Tanni. Window averaging explains only part of this attenuation. A flat-looking
   result therefore does not establish uniform biological speed.
5. The study does not end with an impossibility claim. At 300 simulated
   candidates, matching local PF calibration can recover some gradients and
   sometimes establish a declared equivalence band. Its coverage nevertheless
   falls from 94.50% to 67.50% under independent-map/shared-gain stress. More
   candidates, more observed cell identities and adequate calibration solve
   different problems; they are not interchangeable remedies.

These points belong to distinct experiments and denominators. They are not
an event-level demonstration that every predictive event has an incorrect
speed estimate. The main real biological replication is four PF/five Tanni
animals; simulations and repeated splits do not increase it.

## Novelty Assessment

| Prior work | Already established | Incremental contribution sought here |
| --- | --- | --- |
| [Silva et al., 2015](https://pmc.ncbi.nlm.nih.gov/articles/PMC6095134/) | Degraded behavioral decoding and reassessment of trajectory events | Quantitative coupling to selection, known speed gradients and inference limits in two 2D recording populations |
| [Liu et al., 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10894649/) | Unit-removal controls followed by replay reassessment | Same-event loss linked to independent neural support and controlled kinematic recovery; removal alone is not novel |
| [Wei et al., 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11063820/) | Neural-decoder overconfidence and post-hoc/conformal calibration | Replay-specific selection, map-transfer and candidate-budget effects on calibrated speed-gradient inference |
| [Takigawa et al., 2024](https://elifesciences.org/articles/85635) | Replay validation using a distinct track-discriminability measure without ground truth | Separate-neuron validation inside continuity-screening splits, linked to recorded-cell perturbation in 2D |
| [Denovellis et al., 2021](https://elifesciences.org/articles/64505) | Temporal resolution, uncertainty and selection assumptions matter for replay dynamics | Controlled observation-to-selection-to-speed benchmark; no claim to invent switching models |
| [Ji et al., 2026](https://doi.org/10.1038/s41467-025-68042-3) | Uneven limited place-field coverage can create apparent jumps | Quantified recovery/null trade-offs and independent-map kinematic recovery, not invention of the coverage hypothesis or refutation of their mechanism |
| [Huh et al., 2026](https://doi.org/10.1038/s41467-026-74822-2) | A firing-order likelihood alternative to conventional replay metrics | A measurement-validation study, not a newly invented likelihood detector |

The literature check was targeted, not exhaustive. The original contribution
is the linked quantitative evidence and reusable benchmark, not each component
in isolation. An exact author-encoding reproduction or a head-to-head claim
that a new detector outperforms these methods has not been established and
must not be implied by the manuscript.

## What Is Demonstrated And What Is Not

| Requirement for the proposed insight | Authoritative evidence | Assessment |
| --- | --- | --- |
| Real-data measurement sensitivity, not only a toy example | All33 two-shuffle benchmark, with fixed candidate identities and nine-animal breakdown | Demonstrated within the transferred common-encoding criterion |
| More than a bespoke geometric threshold | Two separate 5,000-shuffle tests and documented ten/eleven-frame, support and detector sensitivities | Demonstrated as a transferred published-style criterion; not exact author reproduction |
| Independent value of rejected events | Training-only screen, untouched held-out neurons, five comparator/ordering contrasts | PF group-level support; Tanni full criterion fails |
| Kinematic error distinguishable from true uniformity | Prescribed positive/negative/zero gradients, window-truth reference and disjoint RUN maps | Strong attenuation demonstrated; no biological speed-equivalence conclusion |
| A usable inference boundary rather than only criticism | Matched/mismatched calibration, excluded-animal transfer, nested candidate budgets and explicit abstention | Conditional success and failure regimes demonstrated; no universal correction |
| Reproducibility and current-state integrity | 15-stage registry; all 1,204 unique indexed files rehashed on 2026-09-10; linked reconstruction audits; new overview values traced to sources | Current artifacts intact; historical scientific audits retain their stated partial/full scopes |
| Publication relevance beyond known generic caveats | Direct-prior comparison above; integrated manuscript and source-linked figures | Credible methods-paper contribution in my assessment; priority/acceptance remains an author/editorial judgment |

No biological mechanism, uniform-speed finding, validated fuzzy classifier,
false-negative rate for true replay, universal dataset difference or uniquely
IMM explanation has been established. No new biological inference follows
merely from the 1,204-file integrity pass or software tests.

## Deliverables And Provenance

- Integrated manuscript: `docs/replay_measurement_manuscript.md`, extending the
  earlier coverage draft with the separate-neuron experiment while preserving
  its denominators and calibration limits.
- Compact overview and all source selectors:
  `/mnt/seagate10tb/florianpfaff/replay-measurement-paper-evidence-20260910/`.
- Current historical artifact integrity:
  `/mnt/seagate10tb/florianpfaff/replay-measurement-paper-integrity-20260910/`.
- New predictive study: `docs/training_continuity_prediction_results.md` and
  the linked complete production, classification audit and report audit.
- Source coverage manuscript and detailed notes remain preserved in
  `/home/florianpfaff/HippoReplayDynamics-recording-coverage/docs/`.

The overview is purely non-rescoring. Every plotted estimate retains its
source table, exact selector, column, scale and original interval. Its source
cohorts, simulation assumptions and negative findings are in the companion
markdown. The predictor was not used to generate the independent geometric
or speed estimates.

## Suggested Message To Dan

The strongest result is no longer that replay speed is constant. We can show
that the same candidate events change trajectory classification when fewer
neurons are visible, even with a Foster-style significance screen. Some of
the geometrically rejected PF events still contain structure that predicts
other neurons. Simulations then show why simply obtaining smoother paths or
more accepted events is not enough: longer windows can also admit randomized
paths, and substantial real speed variation can decode as much flatter. I
think the defensible paper is about when replay continuity and kinematics are
measurable, with an explicit recovery/calibration benchmark. Tanni's weaker
predictive result and the successful as well as failed simulation regimes
would stay in the paper.

## Recommendation

Develop this bounded methods manuscript with the collaborator. This answers
the search for a publication-worthy direction without manufacturing a positive
uniform-speed or novel-mechanism claim. Journal-specific formatting, author
agreement and editorial acceptance remain outside what this computational
evidence can establish. Additional untargeted hypothesis searches or threshold
changes are not needed to make the current evidence a different, stronger
claim than it actually is.
