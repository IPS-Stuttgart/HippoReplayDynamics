# Exhaustive calibration-only audit of the stopped exchange search

Frozen2026-09-15 after the bounded exchange experiment stopped with zero accepted
swaps in all four sessions. That experiment remains FAILED and unchanged. Its
protocol explicitly forbids enlarging its proposal pool after results; this is a
separate diagnostic, not an amended result or a replacement scoring run.

Question: did the32-proposal ranking overlook calibration-admissible single-cell
exchanges, or do all single exchanges fail at least one frozen safeguard?

Use the same four PF pairs, early RUN maps, and first20-ms Q3 observation per
250-ms parent. Read NO replay observations, Q4 outcomes or simulated test banks.
Read the prior manifest/audit and frozen assignments for identity/provenance.
Check every original high-exclusive x low-exclusive pair exactly once. Shared
cells, cell counts, union and intersection are unchanged. There are12,300 legal
single swaps in these four sessions. No sequential or multi-cell search is done.

Keep J=(D_Home^2+D_nonHome^2)/2, and all eight classwise physical-error/Home-Brier
constraints at their original thresholds. Admissibility requires J improvement
>1e-10 and all eight risks no greater than baseline+1e-10. Record exact objective,
all eight risk changes, original derivative rank, and each violated safeguard.

Producer updates baseline log likelihoods by subtracting/adding the exchanged
cell contribution. Independent reconstruction recomputes full Poisson likelihood
from each resulting population, including log factorials, and verifies EVERY
proposal and derived classification. Check exact pair-set coverage and agreement
with all32 proposals in the original frozen trace. No optimization certificate
is claimed for multi-cell moves.

Outputs: per-session proposal CSVs; session summary; rejection counts among
objective-improving swaps; manifest; independent audit; markdown readout.
All calculations on gpuserver6000 in a detached service. Original artifacts are
read-only. Result hashes and git commit are recorded.

Decision: any admissible proposal outside the first32 establishes a search
limitation worth testing in a NEW frozen development intervention. No admissible
single swaps closes only this one-step neighborhood; it does not establish
impossibility of multi-cell exchange or a validated remedy. In neither case do
we run independent confirmation or change replay estimates in this diagnostic.
