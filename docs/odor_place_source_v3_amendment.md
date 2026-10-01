# Original-source amendment: Figshare v3

Authorized by the user's follow-up on 1 October 2026. The completed version-1
source stopgate remains immutable. This is a change to the source release pin,
not a change to the biological endpoint, eligibility floors or decoding rules.

Acquire only `Figure1-6.zip` from Figshare 19620783 version 3, using its published
size and MD5. Store it separately under the home-filesystem dataset root, with
at least 30 GB reserve. Do not substitute the accompanying compiled Figure-7
file or download full NWB recordings. Do not extract the entire archive.

First inventory all source members and read original documentation, task/trigger,
epoch and tracking records. Reconcile the eight animals with the already pinned
DANDI inventory. Conditions require documentation or task metadata, not filenames
alone. Original days/epochs require source records, not conversion dates.

Independent left/right odor triggers must come from stimulus records, separately
from choice/outcome. Reconstructed choice and error annotations must agree with
tracking and source trial records where available. Preserve aborted samples,
missing visits, ambiguous matches and conflicting records with explicit reasons.
No fitted clock offsets and no reconstruction of the next cue from its outcome.

Audit first incorrect-choice well dwells under the unchanged 10 cm, <=4 cm/s,
0.5 s minimum, 10 s cap and 100 ms tracking-gap rules. Count repeated/changed cue
conditions, next choice relative to the preceding error cue, and within-animal
coverage. These are source-feasibility counts, not decoder-qualified observations.
No content-outcome association, coefficient fitting or replay rescoring is allowed.

A source pass permits later neural feasibility work only. It cannot itself emit
`ready_for_calibration=true`: CA1 unit/spike crosswalk, preceding-RUN decoder
validation and ripple opportunity measurement still need independent checks.
If original cues, incorrect-choice pauses or primary cohort coverage cannot be
verified, stop this branch without threshold relaxation or endpoint substitution.

Published source metadata: https://api.figshare.com/v2/articles/19620783/versions/3
Archived author code: https://github.com/JadhavLab/Jadhav-Lab-Codes-JHB/tree/88d5f8d2bb39796b8656dc42bd42969a7bbe8697/BetaOdorProject
