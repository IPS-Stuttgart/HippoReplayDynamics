# Fixed-choice calibration-transfer failure diagnosis

The frozen joint-exchange rule failed93/192 held-out classwise risk safeguards.
This post-hoc diagnostic asks whether failure already appears in nativeQ3
observations that were excluded from optimization, or only in laterQ4/data-model
shifts. It is not another selection experiment or a validation of a remedy.

For each of the original4pairs, retain exactly the audited final choice. Decode
nativeQ3 andQ4 with early maps at20ms, through the existing two independently
implemented likelihood calculations. Separate first20ms per250ms parent (the
Q3 calibration sample), remaining20ms bins in those same parents, and all bins.
Reconstruct the original Q3-first objective and8classwise risks from selection.
No replay observations or alternative assignments are evaluated.

Report both classwise physical error/Home Brier and the squared class-conditional
Home-mass-gap objective. No class or pair is omitted if it fails. Preserve native
parent counts and timestamp checks. Q3-remaining is adjacent to selected data,
not statistically independent; all-bin rows overlap both phase rows. Q4 is a
later RUN block, not an independent animal/recording. No confidence claim is
made from treating these phases or nearby observations as independent.

If gains are confined toQ3-first, record calibration optimism as a practical
failure mode and require any future proposal rule to validate internal native
transfer before independent-recording evaluation. Do not call that a validated
diagnostic until prediction and safe reduction are shown in independent data.
