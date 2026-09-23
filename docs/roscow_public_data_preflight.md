# Roscow 2025 public-data preflight

Purpose: assess an outcome-labelled neural dataset for a new biological question.
This does not score replay or establish novelty of an outcome-dependent effect.

Source: https://github.com/EmmaRoscow/QlearningReplay, pinned at
`d8564850791358b7909c787990b6a22b0d1e01b9`.
Study: https://doi.org/10.1038/s41467-025-65354-2.

Server storage:
`gpuserver6000:/mnt/seagate10tb/florianpfaff/datasets/roscow-2025`.
Do not use the nearly full lexar volume for another copy.

## Acquisition

`scripts/acquire_roscow2025.py --dataset-root <root>` downloads a frozen subset:
spike timestamps, epochs, ripple intervals, task timestamps, running speed,
behavioral tables and author source. Precomputed binned firing rates, fitted
Q-learning outputs and serialized analysis objects are omitted explicitly.
398 files / 2,195,811,118 bytes were verified against published Git blob hashes;
additional SHA-256 hashes are stored in `download_status.json`.
No account or private credentials are required.

Use tmux and the download terminal-status file for disconnect-safe operation.
Existing complete files are hash-checked, not silently overwritten. Partial files
resume; corrupted files fail verification and remain available for inspection.

## Audit conventions

`scripts/preflight_roscow2025.py --dataset-root <root> --output-dir <new-dir>`
verifies the acquisition hashes and writes sessions, trials, units, by-animal and
by-outcome tables, a manifest and a summary. It refuses to overwrite outputs.

Session folder spelling varies (`SessionN` and `sessionN`). Original numeric N
indexes behavioral cell N; missing neural sessions must never shift the join.
The first `nNAcUnits` spike trains are ventral-striatal; remaining trains are CA1,
following the author's EphysAnalysis class. Counts across days are session-unit
records, not confirmed distinct neurons. The RewardRelatedFiring class defines
trial actions, arm legitimacy and reward labels; illegitimate choices are not
probabilistic omissions. Reward-probability categories are task variables, not
subjective expectations or estimated prediction errors.

Initial audit: 45 session folders / 3 neural animals; 38 sessions have aligned
trial records (598 trials), and 36 pass all implemented metadata checks. The
other sessions are retained in the inventory with explicit issues. No sorting,
duplicate deletion, epoch shifting or trial-label repair is performed silently.

## Important unresolved checks

- Trial labels follow the published crosswalk, not an independent video audit.
- Some arrivals are duplicated/out of order; one session has overlapping epochs;
  another has an illegitimate trial marked rewarded. Seek clarification before
  recovering these sessions.
- Central-platform times are not validated as complete ordered trial cycles.
- The release has speed/timestamps but no verified continuous x/y position table.
- A task-state decoder must pass held-out trial validation before sequence use.
- Native rest ripples are not automatically temporally linked to single trials.
- Only three neural animals are available. Condition counts and unit availability
  must be inspected per animal rather than pooled into apparent replication.

The source paper already concerns reward-prediction-related reactivation.
Any proposed temporal-order/content extension requires a separate prior-art
check and fixed, independently calibrated endpoint. Data availability is not a
positive biological finding.
