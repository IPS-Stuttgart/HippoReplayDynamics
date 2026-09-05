# Numerical Boundary Addendum, Before Published-Budget Scoring

The K=40 two-session timing run at commit 2ea9a052 failed independent
reconstruction. No full-budget result was run or interpreted from that version.
The specific discrepancy was a 32-by-24 cm displacement on PF Rat1/Open1's
decimal-offset grid: scalar norm returned 40.0, batched norm returned
39.99999999999999. Squared norms likewise differed by approximately 2e-13.
The source event was a half-cell cell-identity shuffle, not a biological
selection made after a positive result. K=40 cannot meet p<0.02 at all.

Both implementations now use a declared 1e-9 cm numerical comparison tolerance:
strict jumps <20-1e-9 cm and inclusive displacement >=40-1e-9 cm. This treats
roundoff representations of equality consistently, while retaining the stated
20 cm/40 cm scientific thresholds. The regression test includes the observed
40 cm diagonal and a truly subthreshold displacement. No maps, candidates,
shuffle seeds, cell subsets, or significance thresholds are changed.

The original protocol and failed timing artifact are preserved. Repeat the
timing run and independent audit in a new directory before full K=5,000 scoring.
The scoring manifest hashes this addendum as well as the original protocol.
