# Replay structural updating: bounded feasibility protocol

This study asks whether the existing Widloski/Foster recordings could test
updating at **locally unchanged junctions before renewed outgoing traversal**.
It performs inventory, geometry/history opportunity counting and design checks
only. It never decodes replay, reads represented paths, fits biological outcomes,
or implements downstream calibration. Other studies and their stop decisions
remain untouched. Initial implementation/assessment is capped at five working
days, excluding author-response time.

## Existing work and current access

Rapid adaptation to barriers is already established by
[Widloski and Foster, 2022](https://doi.org/10.1016/j.neuron.2022.02.002).
Spatial remoteness alone is also not a new contribution. Review of that paper,
its available supplementary descriptions and named code, the
[2024 planning reanalysis](https://www.nature.com/articles/s41593-024-01675-7),
and the [2025 study](https://www.nature.com/articles/s41467-025-65181-5)
has not established that the exact proposed experience-conditioned comparison
is new. The novelty gate remains unresolved, not passed by absence of a keyword.
The source manifest records review limits, URLs, code hashes and field meanings.

The three-file [processed release](https://zenodo.org/records/16916108) is not
the full recording set. `Data.ratLocs` contains per-event animal locations;
`Data.replays` contains decoded replay coordinates. Neither is continuous
behavioral tracking. Aggregate rate maps are not sorted spikes or cross-session
unit identities. Event indices are not start/end timestamps. Shuffle and event
statistics tables have different row counts and must not be joined positionally.
Missing event identity rows are retained as unresolved, never repaired by row
number. This is not evidence that those rows or events are biologically invalid.

## Execution

Use an isolated clean, committed checkout on `gpuserver6000`. For example:

```bash
python scripts/launch_widloski_structural_updating.py \
  --job-dir /mnt/lexar4tb/analysis/widloski-structural-updating-RUN/job \
  -- \
  --dataset-root /mnt/lexar4tb/datasets/widloski-foster-2025/archive \
  --source-manifest docs/widloski_structural_updating_source_manifest.json \
  --protocol docs/widloski_structural_updating_protocol.json \
  --output-dir /mnt/lexar4tb/analysis/widloski-structural-updating-RUN/audit
```

The launcher creates a detached supervisor, durable `job.log`, PID, commit and
stage status files. It runs `inventory`, `opportunities`, then `report`. Individual
stages can also be invoked through
`scripts/audit_widloski_structural_updating_feasibility.py --stage ...` with the
same driver arguments. Supply `--verified-metadata PATH` only when actual reviewed
metadata are available. No placeholder geometry or inferred clocks are allowed.

Completed stages verify their artifacts and input hashes on restart. Changed
inputs require a new output directory. Interrupted stages may be rerun; the
inventory also checkpoints its per-file ledger. Reporting verifies archived
upstream artifacts only, so it works without the recordings. Exit zero means
technical completion; scientific readiness is exclusively in `decision.json`.

## Verified metadata contract

The reviewed JSON has `schema_version: 1`, `reviewed_by`, `documentation`, `graph`
and `sessions`. Every source session must appear, including unavailable sessions;
do not silently filter the cohort. All declared input files carry `path`,
`sha256`, and `documentation` explaining author field meanings, transformations
and units. Behavioral tables additionally carry the session's `clock_id`.

Each session records:

- `animal`, source-matching `session_id` (`animal/YYYYMMDD/runN`), ISO `date`, and
  author-verified positive `session_order`.
- `entry_time_s`, `end_time_s`, `clock_id`, `time_alignment_verified: true`;
  synchronizing multiple clocks is an upstream, documented responsibility.
- `configuration_id`, `home_goal_node`, `previous_session_id`, and
  `intervening_exposure_complete`. A missing intervening session or unrecorded
  exposure excludes that change; numerical adjacency alone is insufficient.
- `traversals_complete: true`, and checked `position`, `events`, `traversals`
  CSV specifications. Completeness is an explicit reviewed assertion, not inferred
  from the absence of a traversal row.
- `capabilities`: all eight capabilities in the source manifest, each with
  `status: verified|absent|unresolved`. A verified capability needs a hashed
  evidence file and documentation; invalid witnesses become unresolved.
  Sorted-spike/unit-ID/pre-change-encoding witnesses must identify the actual
  available neural resources and coverage, not just assert that a paper used them.

CSV contracts:

| Input | Columns | Meaning |
| --- | --- | --- |
| position | `t_s,x_cm,y_cm,speed_cm_s,supported` | Full-session observed tracking, strictly increasing timestamps, explicit supported samples |
| events | `event_id,start_time_s,end_time_s,validated` | Native author event IDs and validated intervals, not new detection or validation |
| traversals | `traversal_id,edge_id,from_node,to_node,start_time_s,end_time_s` | Independent, complete, directed behavior annotations checked against geometry and position |

Graph JSON: `documented: true`, `behaviorally_checked: true`, `documentation`,
`length_unit: cm`; fixed `nodes` with `id,x_cm,y_cm,radius_cm`; fixed undirected
`edges` with `id,u,v,length_cm`; `configurations` with `id,open_edges`; fixed
`destinations` naming documented graph nodes. Physical graph construction from
well locations or replay paths is not supported. Coordinates must share the
documented tracking frame.

## Opportunity definition

Compare adjacent verified configurations at every documented fixed destination.
Outgoing route cost is the incident edge length plus the shortest remaining
path **without returning through the origin junction**. Preserve all shortest
ties within numerical tolerance. A junction must retain identical incident
edges and lengths; an affected junction changes relative outgoing costs. Keep
unreachable alternatives as explicit exclusions, not invented infinite effects.
Changing the goal alone cannot become a barrier effect.

Require an observed previous visit to the junction. Exposure starts at entry
into the new configuration and ends at the first outgoing departure, the last
supported sample before a gap over 250 ms/invalid tracking, or session end.
Unsupported entry supplies no exposure. Never resume untraversed status after
a gap. Count only fully contained validated intervals during contiguous support
with endpoint speeds strictly below 5 cm/s. No content array is inspected.

Unaffected controls match within episode and destination, with the same number
of outgoing alternatives and closest old shortest cost; resolve ties by fixed
ID. Match with replacement without looking at event counts. Count unique native
events separately from event-opportunity memberships. Events may belong to
several opportunities; opportunities are not independent neural observations.
Distinct native IDs with overlapping intervals are flagged and retained, while
duplicate IDs are rejected. Their overlap does not create extra unique events.

## Confounding, coverage and decisions

Allow simultaneous goal/barrier changes. Report changed/unchanged goals,
old/new goals and configurations, animal/day, elapsed-time and exposure support.
Structural predictor is old minus new relative outgoing route cost, always at
the same destination. On matched exposed opportunities, inspect variation within
goal-pair strata across multiple episodes and configuration changes. Check the
predictor's rank increment after animal-day, goal-pair, junction, destination,
alternative, old relative cost and exposure terms. Full rank is necessary support,
not causal identification. No replay-dependent response is fit.

Require each of Billy3, Curly2 and Goethe2 to have at least two eligible
reconfiguration episodes on two days, with affected events and matched unaffected
events. This is a minimal engineering floor, not power or generalization.

All gates are emitted. The single decision prioritizes missing/inconsistent data,
then metadata, unresolved novelty, confounding and coverage; only all-pass becomes
`ready_for_readout_feasibility`. Partial input failures produce unavailable
coverage, not biological zeros. A ready result also requires verified sorted
spikes, unit identities and pre-change encoding support. Counts alone cannot pass.

Artifacts live under `inventory/`, `opportunities/`, and `report/`. Empty
opportunity tables under `blocked_data` or `blocked_metadata` denote unmeasured
quantities. Figures, when justified, show geometry and behavioral timelines only;
the figure ledger explicitly reports unavailable panels otherwise. The author
request remains unsent. Archive these small artifacts, not raw recordings.

Not retraversed must never be described as unknown to the animal: barriers were
visible. The next separately specified milestone, only after readiness, is
pre-change encoder validation and effect-blind calibration.
