# Tanni uniform-prior replay-speed test

This analysis separates three operations that must not be conflated:

1. Candidate detection uses ripple power, immobility, spike count, and active-cell support.
2. Replay validation uses independent 20 ms Poisson decoding with a uniform prior over valid locations, followed by a whole-population-bin temporal-order shuffle.
3. Stationary, diffusion, fragmented, first-order IMM, and momentum labels are secondary characterizations and never define the primary path or select an event.

The conservative replay statistic projects each event's posterior-mean locations onto the principal spatial axis and measures the absolute Spearman association between represented position and time. This permits variable speed while requiring monotonic progression. Events must span at least 32 cm and survive Benjamini-Hochberg correction across all spatially extended candidates. Overlapping source windows are reduced to one representative before wall-distance analysis.

The primary question is whether the previously observed decoded-speed relationship with wall distance survives this model-independent replay gate, posterior-quality controls, animal-balanced inference, and the existing constant-physical-speed decoder null.

Passing the sequence gate establishes only a strict approximately linear ordered subset. It is not designed to recover curved or branching trajectories and therefore is not a prevalence estimator for all replay.
