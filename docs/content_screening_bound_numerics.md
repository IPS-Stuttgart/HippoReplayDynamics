# Numerical repair of the screening bound

The initial run completed all 384 optimizations but FAILED independent audit.
The first rejected certificate was native Q4, Rat1/Open1, q=0.75, free-event,
truth-guarded: its high-population Brier constraint had residual 1.77257e-7,
exceeding the frozen 1e-7 primal audit tolerance. This run is not certified.

Very small positive Brier coefficients can be ignored by the solver while
their aggregate contribution exceeds the audit tolerance. The repaired solve
multiplies every inequality and equality, including its right-hand side, by
10,000 and tightens solver feasibility tolerances to 1e-9. This is an equivalent
linear program; no scientific target, coverage, source, or truth safeguard changes.
Saved inequality/equality dual multipliers are multiplied by 10,000 to return
to the ORIGINAL unscaled constraints. The independent auditor remains unchanged
and verifies those original constraints at its original tolerances.

The first failed measurement and its terminal completion record are retained.
The repaired run uses a separate output directory and a new code commit.
No scientific bounds are interpreted before independent certification passes.
