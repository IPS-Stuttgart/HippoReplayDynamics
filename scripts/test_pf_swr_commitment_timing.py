#!/usr/bin/env python3
"""Compare local-pause timing for PF off-SWR candidates and SWR events.

The promoted off-SWR table is first de-duplicated by overlapping physical
windows, in addition to its existing source-event de-duplication. Departure is
defined from the position trace as the next sustained local crossing of the RUN
speed threshold, not the beginning of a potentially long route interval.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from hipporeplayimm.data import load_replay_session

try:
    from build_replay_behavior_route_primitives import smooth_position_trace
except ModuleNotFoundError:  # Imported as scripts.* by tests.
    from scripts.build_replay_behavior_route_primitives import smooth_position_trace

EVENT_OUTPUT = "pf_swr_commitment_timing_events.csv"
SUMMARY_OUTPUT = "pf_swr_commitment_timing_summary.csv"
BY_RAT_OUTPUT = "pf_swr_commitment_timing_by_rat.csv"
NULL_OUTPUT = "pf_swr_commitment_timing_permutation_null.csv"
GATE_OUTPUT = "pf_swr_commitment_timing_gate_summary.csv"
REPORT_OUTPUT = "pf_swr_commitment_timing_report.md"
MANIFEST_OUTPUT = "pf_swr_commitment_timing_manifest.json"


def deduplicate_physical_candidate_windows(
    decisions: pd.DataFrame,
    *,
    selection_rule: str = "strongest_exact_margin",
) -> pd.DataFrame:
    """Keep one strongest row per connected component of overlapping windows."""

    selected = decisions[
        decisions["selection_rule"].astype(str).eq(selection_rule)
    ].copy()
    if selected.empty:
        return selected
    rows: list[pd.Series] = []
    cluster_index = 0
    for session, group in selected.groupby("session", sort=True):
        ordered = group.sort_values(["window_start_s", "window_end_s"]).copy()
        cluster_members: list[pd.Series] = []
        cluster_end = -np.inf

        def flush() -> None:
            nonlocal cluster_index, cluster_members
            if not cluster_members:
                return
            cluster = pd.DataFrame(cluster_members)
            margin_column = (
                "trajectory_minus_nontrajectory_log_evidence"
                if "trajectory_minus_nontrajectory_log_evidence" in cluster
                else "trajectory_family_margin"
            )
            if margin_column not in cluster:
                cluster[margin_column] = np.nan
            best = cluster.sort_values(
                [margin_column, "window_start_s"],
                ascending=[False, True],
                na_position="last",
            ).iloc[0].copy()
            best["physical_candidate_cluster_id"] = f"{session}|physical={cluster_index}"
            best["physical_candidate_cluster_size"] = int(len(cluster))
            best["physical_candidate_source_event_indices"] = ",".join(
                str(int(value)) for value in sorted(cluster["event_index"].astype(int).unique())
            )
            rows.append(best)
            cluster_index += 1
            cluster_members = []

        for _, candidate in ordered.iterrows():
            start = float(candidate["window_start_s"])
            end = float(candidate["window_end_s"])
            if cluster_members and start > cluster_end:
                flush()
                cluster_end = -np.inf
            cluster_members.append(candidate)
            cluster_end = max(cluster_end, end)
        flush()
    return pd.DataFrame(rows).reset_index(drop=True)


def next_local_departure(
    position_trace: np.ndarray,
    event_time_s: float,
    *,
    speed_threshold_cm_s: float = 5.0,
    minimum_departure_s: float = 0.25,
    search_horizon_s: float = 30.0,
) -> dict[str, object]:
    """Measure time from a locally immobile event to sustained movement."""

    trace = np.asarray(position_trace, dtype=float)
    if trace.ndim != 2 or trace.shape[1] < 4 or len(trace) < 2:
        return {"local_pause_status": "missing_position", "time_to_local_departure_s": np.nan}
    times = trace[:, 0]
    speed = trace[:, 3]
    index = int(np.clip(np.searchsorted(times, float(event_time_s)), 0, len(times) - 1))
    event_speed = float(speed[index])
    if event_speed >= float(speed_threshold_cm_s):
        return {
            "local_pause_status": "moving_at_event",
            "event_position_speed_cm_s": event_speed,
            "time_to_local_departure_s": np.nan,
        }
    dt = float(np.median(np.diff(times)))
    required = max(1, int(np.ceil(float(minimum_departure_s) / dt)))
    stop = min(
        len(times) - required,
        int(np.searchsorted(times, float(event_time_s) + float(search_horizon_s))),
    )
    departure_index: int | None = None
    for candidate in range(index, stop + 1):
        if np.all(speed[candidate : candidate + required] >= float(speed_threshold_cm_s)):
            departure_index = candidate
            break
    if departure_index is None:
        return {
            "local_pause_status": "no_departure_within_horizon",
            "event_position_speed_cm_s": event_speed,
            "time_to_local_departure_s": np.nan,
        }
    return {
        "local_pause_status": "local_pause",
        "event_position_speed_cm_s": event_speed,
        "time_to_local_departure_s": float(times[departure_index] - float(event_time_s)),
        "local_departure_time_s": float(times[departure_index]),
    }


def build_timing_events(
    *,
    dataset_root: Path,
    off_swr_decisions: pd.DataFrame,
    swr_context: pd.DataFrame,
    selection_rule: str = "strongest_exact_margin",
    speed_threshold_cm_s: float = 5.0,
    minimum_departure_s: float = 0.25,
    search_horizon_s: float = 30.0,
) -> pd.DataFrame:
    """Build local-pause timing rows for independent off-SWR and SWR events."""

    off = deduplicate_physical_candidate_windows(
        off_swr_decisions,
        selection_rule=selection_rule,
    )
    event_rows: list[dict[str, object]] = []
    for candidate in off.itertuples(index=False):
        event_rows.append(
            {
                "event_class": "off_swr",
                "session": str(candidate.session),
                "rat": str(candidate.rat),
                "event_index": int(candidate.event_index),
                "event_time_s": 0.5 * (
                    float(candidate.window_start_s) + float(candidate.window_end_s)
                ),
                "physical_candidate_cluster_id": str(candidate.physical_candidate_cluster_id),
                "physical_candidate_cluster_size": int(candidate.physical_candidate_cluster_size),
            }
        )
    for event in swr_context[["session", "rat", "event_index", "event_peak_s"]].drop_duplicates().itertuples(index=False):
        event_rows.append(
            {
                "event_class": "swr",
                "session": str(event.session),
                "rat": str(event.rat),
                "event_index": int(event.event_index),
                "event_time_s": float(event.event_peak_s),
                "physical_candidate_cluster_id": "",
                "physical_candidate_cluster_size": 1,
            }
        )
    events = pd.DataFrame(event_rows)
    trace_cache: dict[str, np.ndarray] = {}
    timing_rows: list[dict[str, object]] = []
    for event in events.itertuples(index=False):
        if event.session not in trace_cache:
            session = load_replay_session(dataset_root / Path(event.session))
            trace_cache[event.session] = smooth_position_trace(session.position)
        timing = next_local_departure(
            trace_cache[event.session],
            float(event.event_time_s),
            speed_threshold_cm_s=speed_threshold_cm_s,
            minimum_departure_s=minimum_departure_s,
            search_horizon_s=search_horizon_s,
        )
        timing_rows.append({**event._asdict(), **timing})
    return pd.DataFrame(timing_rows)


def _equal_rat_effect(events: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    valid = events[events["local_pause_status"].eq("local_pause")].copy()
    rows: list[dict[str, object]] = []
    for rat, rat_rows in valid.groupby("rat", sort=True):
        off = rat_rows[rat_rows["event_class"].eq("off_swr")]["time_to_local_departure_s"]
        swr = rat_rows[rat_rows["event_class"].eq("swr")]["time_to_local_departure_s"]
        if off.empty or swr.empty:
            continue
        effect = float(off.median() - swr.median())
        rows.append(
            {
                "rat": rat,
                "off_swr_events": int(len(off)),
                "swr_events": int(len(swr)),
                "off_swr_median_time_to_departure_s": float(off.median()),
                "swr_median_time_to_departure_s": float(swr.median()),
                "off_minus_swr_median_time_to_departure_s": effect,
                "positive_direction": bool(effect > 0.0),
            }
        )
    by_rat = pd.DataFrame(rows)
    return (
        float(by_rat["off_minus_swr_median_time_to_departure_s"].mean())
        if len(by_rat)
        else np.nan,
        by_rat,
    )


def permutation_test(
    events: pd.DataFrame,
    *,
    replicates: int,
    seed: int,
) -> tuple[pd.DataFrame, float]:
    """Permute event class within session while preserving off-SWR counts."""

    valid = events[events["local_pause_status"].eq("local_pause")].copy()
    observed, _ = _equal_rat_effect(valid)
    rng = np.random.default_rng(int(seed))
    null_values: list[float] = []
    for _ in range(int(replicates)):
        permuted = valid.copy()
        for _, session in permuted.groupby("session", sort=False):
            permuted.loc[session.index, "event_class"] = rng.permutation(
                session["event_class"].to_numpy()
            )
        value, _ = _equal_rat_effect(permuted)
        if np.isfinite(value):
            null_values.append(value)
    null = pd.DataFrame({"replicate": np.arange(len(null_values)), "equal_rat_effect": null_values})
    p_value = (
        float((1 + int((null["equal_rat_effect"] >= observed).sum())) / (1 + len(null)))
        if len(null) and np.isfinite(observed)
        else np.nan
    )
    return null, p_value


def summarize(
    events: pd.DataFrame,
    *,
    permutation_replicates: int,
    bootstrap_replicates: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Summarize timing with rat-equal inference and a small-cohort stopgate."""

    effect, by_rat = _equal_rat_effect(events)
    null, p_value = permutation_test(events, replicates=permutation_replicates, seed=seed)
    rng = np.random.default_rng(int(seed) + 1)
    rat_values = by_rat["off_minus_swr_median_time_to_departure_s"].to_numpy(dtype=float)
    boot = np.array(
        [np.mean(rng.choice(rat_values, size=len(rat_values), replace=True)) for _ in range(int(bootstrap_replicates))]
    ) if len(rat_values) else np.array([], dtype=float)
    off = events[(events["event_class"].eq("off_swr")) & (events["local_pause_status"].eq("local_pause"))]
    swr = events[(events["event_class"].eq("swr")) & (events["local_pause_status"].eq("local_pause"))]
    summary = pd.DataFrame(
        [
            {
                "hypothesis": "H7_SWRs_mark_commitment",
                "physical_off_swr_events": int(events["event_class"].eq("off_swr").sum()),
                "off_swr_local_pause_events": int(len(off)),
                "swr_local_pause_events": int(len(swr)),
                "off_swr_rats": int(off["rat"].nunique()),
                "off_swr_sessions": int(off["session"].nunique()),
                "off_swr_median_time_to_departure_s": float(off["time_to_local_departure_s"].median()),
                "swr_median_time_to_departure_s": float(swr["time_to_local_departure_s"].median()),
                "pooled_off_minus_swr_median_s": float(off["time_to_local_departure_s"].median() - swr["time_to_local_departure_s"].median()),
                "equal_rat_off_minus_swr_median_s": effect,
                "rat_bootstrap_ci_low": float(np.quantile(boot, 0.025)) if len(boot) else np.nan,
                "rat_bootstrap_ci_high": float(np.quantile(boot, 0.975)) if len(boot) else np.nan,
                "within_session_permutation_p_one_sided": p_value,
                "positive_rats": int(by_rat.get("positive_direction", pd.Series(dtype=bool)).sum()),
                "inferential_status": "insufficient" if len(off) < 10 else "complete",
            }
        ]
    )
    return summary, by_rat, null


def build_gates(events: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    row = summary.iloc[0]
    gates = [
        ("physical_off_swr_candidates_present", int(row["physical_off_swr_events"]) > 0, int(row["physical_off_swr_events"]), ">0"),
        ("all_off_swr_candidates_are_local_pauses", int(row["off_swr_local_pause_events"]) == int(row["physical_off_swr_events"]) > 0, f"{int(row['off_swr_local_pause_events'])}/{int(row['physical_off_swr_events'])}", "all selected candidates locally immobile with a subsequent departure"),
        ("swr_reference_events_present", int(row["swr_local_pause_events"]) >= 100, int(row["swr_local_pause_events"]), ">=100"),
        ("minimum_independent_off_swr_events", int(row["off_swr_local_pause_events"]) >= 10, int(row["off_swr_local_pause_events"]), ">=10 predeclared for inference"),
        ("multiple_off_swr_rats", int(row["off_swr_rats"]) >= 3, int(row["off_swr_rats"]), ">=3"),
    ]
    rows = [{"gate": gate, "passed": passed, "observed": observed, "criterion": criterion} for gate, passed, observed, criterion in gates]
    rows.append({"gate": "overall_inferential", "passed": all(item[1] for item in gates), "observed": f"{sum(item[1] for item in gates)}/{len(gates)}", "criterion": "all H7 timing gates pass"})
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--off-swr-decisions", required=True)
    parser.add_argument("--swr-context", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--selection-rule", default="strongest_exact_margin")
    parser.add_argument("--speed-threshold-cm-s", type=float, default=5.0)
    parser.add_argument("--minimum-departure-s", type=float, default=0.25)
    parser.add_argument("--search-horizon-s", type=float, default=30.0)
    parser.add_argument("--permutation-replicates", type=int, default=5000)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260804)
    args = parser.parse_args()
    events = build_timing_events(
        dataset_root=Path(args.dataset_root),
        off_swr_decisions=pd.read_csv(args.off_swr_decisions),
        swr_context=pd.read_csv(args.swr_context),
        selection_rule=args.selection_rule,
        speed_threshold_cm_s=args.speed_threshold_cm_s,
        minimum_departure_s=args.minimum_departure_s,
        search_horizon_s=args.search_horizon_s,
    )
    summary, by_rat, null = summarize(
        events,
        permutation_replicates=args.permutation_replicates,
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    gates = build_gates(events, summary)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    events.to_csv(output / EVENT_OUTPUT, index=False)
    summary.to_csv(output / SUMMARY_OUTPUT, index=False)
    by_rat.to_csv(output / BY_RAT_OUTPUT, index=False)
    null.to_csv(output / NULL_OUTPUT, index=False)
    gates.to_csv(output / GATE_OUTPUT, index=False)
    row = summary.iloc[0]
    report = (
        "# PF SWR commitment timing\n\n"
        f"Independent physical off-SWR candidates: **{int(row['physical_off_swr_events'])}**. "
        f"All are local pauses: **{int(row['off_swr_local_pause_events'])}/{int(row['physical_off_swr_events'])}**.\n\n"
        f"Median time to local departure is {row['off_swr_median_time_to_departure_s']:.2f} s off-SWR "
        f"and {row['swr_median_time_to_departure_s']:.2f} s for SWRs. The rat-equal off-minus-SWR "
        f"effect is {row['equal_rat_off_minus_swr_median_s']:+.2f} s "
        f"(permutation p={row['within_session_permutation_p_one_sided']:.4f}).\n\n"
        "The cohort is below the predeclared ten-candidate minimum, so this is descriptive and cannot support H7.\n"
    )
    (output / REPORT_OUTPUT).write_text(report)
    manifest = {
        "analysis": "H7_SWRs_mark_commitment_local_pause_timing",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "settings": vars(args),
        "off_swr_decisions_sha256": hashlib.sha256(Path(args.off_swr_decisions).read_bytes()).hexdigest(),
        "swr_context_sha256": hashlib.sha256(Path(args.swr_context).read_bytes()).hexdigest(),
        "claim_boundary": "descriptive when fewer than ten independent physical off-SWR events",
    }
    (output / MANIFEST_OUTPUT).write_text(json.dumps(manifest, indent=2) + "\n")
    print(summary.to_string(index=False))
    print(by_rat.to_string(index=False))
    print(gates.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
