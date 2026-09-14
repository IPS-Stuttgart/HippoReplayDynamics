# hc-11 fixed-endpoint observability: a measurement-design limitation

## Decision

The fixed-endpoint observation design FAILS its counts-only preflight. Do not
fit another endpoint-content predictor on these recordings and call it an
independent confirmation. This is progress in diagnosing the failed assays,
not a validated remedy. The requested remedy/diagnostic goal remains open.

All computation used detached systemd services on gpuserver6000. The native
data are independent of PF, Tanni, Blackstad and AutoPI. The cohort is eight
linear/circular-track recordings from four rats, not an open-field replication.
57,460 POST population bursts were detected; these are NOT validated replays,
NREM episodes, ripple events or necessarily immobile episodes.

## Result

The activity screen used native excitatory CA1 labels and first-half RUN spikes
only, retaining 34-115 cells per recording. Place-field quality and local spatial
recovery were not evaluated, so this is an optimistic availability screen.
Partitioned cells are disjoint and equal-count; split 0 is primary, with two
additional frozen splits retained as sensitivity, not extra animals.

Each percentage below is the fraction of candidate windows with at least three
spikes from two active cells in EACH of two disjoint populations. Sessions are
averaged equally within rat. Original candidate endpoints are unchanged. The
separate peak window is diagnostic and never replaces an endpoint.

| Rat | Fixed last-complete 20 ms (%) | Peak-centered 20 ms (%) |
|---|---:|---:|
| Achilles | 3.03 | 73.14 |
| Buddy | 0.00 | 12.87 |
| Cicero | 0.34 | 33.52 |
| Gatsby | 0.16 | 23.97 |

In the lower-count half, endpoint mean counts were 0.875, 0.236, 0.511 and
0.428 spikes, respectively. At least one half was silent at many endpoints.
The three-population design was more constrained: only Achilles had more than
20% of peak windows supported in all three groups; the primary endpoint
fractions were essentially zero in every rat.

All eight sources and all four rats were retained; all detected endpoints
could be measured. The frozen requirement of >=20% activity-supported
endpoints in at least three rats failed: zero rats reached it. This result
does not depend on dropping failed sessions or selecting a favorable split.

## What this establishes, and what it does not

Mean-return MUA boundaries occur where smoothed population activity returns
toward the epoch baseline. It is therefore unsurprising that an unsmoothed
20-ms window near that edge can contain little activity after partitioning.
The peak/endpoint contrast is consistent with this design mechanism. It does
not prove the cause of every previous failed diagnostic or quantify spatial
information from spike counts alone; a few selective cells can be informative.

Crucially, the original Pfeiffer-Foster paper already adjusted candidate edges
inward until each boundary decoding window contained at least two spikes.
Source: Pfeiffer & Foster (2013), Nature, DOI 10.1038/nature12112, Methods,
"Sequential event analysis"; inspected primary-paper PDF:
https://www.snn.ru.nl/~bertk/acns/pfeifer_foster_nature2013.pdf

Our raw fixed-boundary assay deliberately avoided moving endpoints, but is not
that published edge-trimmed assay. The present sparse-boundary finding is NOT a
new state-of-the-art remedy, not a failure of the published method, and not a
new biological result. The earlier matched-population study also found a content
contrast at previously accepted trajectory-segment endpoints; that separate
result is not erased or explained away by this counts-only audit.

More spikes at a peak are necessary evidence for considering a different
readout, NOT sufficient proof of accurate or stable content. Peak content and
endpoint content are different quantities when a trajectory moves. Any future
trimmed/core readout must preserve this distinction and validate spatial error,
population stability and availability rather than merely increase spike counts.

## Native-clock correction

The first run rejected Gatsby_08022013 because sorted spikes extended past the
declared POSTEpoch end (30413.628 s). Native XML and the EEG byte length verify
an acquisition duration of 32002.4 s, consistent with those spike timestamps.
The corrected reader retains the original POSTEpoch limit and does not analyze
the extra tail or shift any time. It records XML checksum and EEG size/mtime;
EEG waveform payload was not decoded or hashed by this counts audit. All other
source events and screening rules were unchanged; the first run is preserved.

## Verification and artifacts

- 55 relevant tests passed; focused Ruff checks passed.
- An independent unique-boundary histogram/CDF implementation reconstructed
  6,958,958 native cell/window counts across all eight recordings.
- 689,520 population rows were checked, including deterministic partitions,
  dropped IDs, activity support flags, spike counts and active-cell counts.
- Source checksums, native physiological labels, fixed window identities,
  session/rat aggregates and every feasibility gate were verified.
- The four-panel PNG/PDF was visually inspected; labels and legends are legible.
- Tests detect corrupt population counts even after their file hash is updated.
- This does not independently verify native spike sorting or prove biological
  replay ground truth. No spatial decoder or predictive model was fitted.

Producer/protocol: 4f0b6783; clock correction: 20cb3592; independent reporter:
6cdf3b44. All source and output hashes are in artifact manifests.

Server code:
`/home/florianpfaff/HippoReplayDynamics-content-stability-20260914`

Authoritative corrected outputs:
`/mnt/seagate10tb/florianpfaff/hc11-content-observability-v2-20260914`

Original seven-source result:
`/mnt/seagate10tb/florianpfaff/hc11-content-observability-20260914`

Compact local report, figure and audited tables (no raw spikes):
`/mnt/c/Users/emper/Documents/codex/2026-09-14/hc11-content-observability`

## Next constraint on the goal

The next proposed remedy must address the observation model/readout explicitly,
including the published edge-support baseline. Do not repeat an uninformative
endpoint assay on another dataset and call that a validation attempt with
adequate power. Do not silently move to burst peaks and describe their content
as the original endpoint. A supported core/trimmed readout is a new controlled
comparison whose stability and true-position errors must still be tested.
