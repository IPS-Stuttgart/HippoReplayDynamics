#!/usr/bin/env python3
"""Test whether IMM boundaries align with behavior-route composition by chance."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _provenance import build_script_provenance  # noqa: E402
from compute_replay_commitment_composition_metrics import (  # noqa: E402
    _path_dictionary,
    _primitive_library_ids,
    _training_route_ids,
    best_path_fit,
    path_length,
)


SCORE_OUTPUT = "replay_dynamics_behavior_boundary_shift_scores.csv"
DECISION_OUTPUT = "replay_dynamics_behavior_boundary_shift_decisions.csv"
SUMMARY_OUTPUT = "replay_dynamics_behavior_boundary_shift_summary.csv"
BY_RAT_OUTPUT = "replay_dynamics_behavior_boundary_shift_by_rat.csv"
GATE_OUTPUT = "replay_dynamics_behavior_boundary_shift_gate_summary.csv"
MANIFEST_OUTPUT = "replay_dynamics_behavior_boundary_shift_manifest.json"
REPORT_OUTPUT = "replay_dynamics_behavior_boundary_shift_report.md"


def circular_switch_count(modes: np.ndarray) -> int:
    values = np.asarray(modes, dtype=int)
    return int(np.sum(values != np.roll(values, -1))) if len(values) > 1 else 0


def circular_dwell_signature(modes: np.ndarray) -> tuple[tuple[int, int], ...]:
    """Return a rotation-invariant multiset of circular mode/dwell pairs."""

    values = np.asarray(modes, dtype=int)
    if not len(values):
        return ()
    changes = np.flatnonzero(values != np.roll(values, 1))
    if not len(changes):
        return ((int(values[0]), int(len(values))),)
    start = int(changes[0])
    rotated = np.roll(values, -start)
    boundaries = np.flatnonzero(rotated[1:] != rotated[:-1]) + 1
    chunks = np.split(rotated, boundaries)
    return tuple(sorted((int(chunk[0]), int(len(chunk))) for chunk in chunks))


def shifted_mode_timeline(modes: np.ndarray, offset: int) -> np.ndarray:
    values = np.asarray(modes, dtype=int)
    if not len(values):
        return values.copy()
    shifted = np.roll(values, int(offset) % len(values))
    if not np.array_equal(np.bincount(shifted, minlength=3), np.bincount(values, minlength=3)):
        raise AssertionError("mode counts changed under circular shift")
    if circular_switch_count(shifted) != circular_switch_count(values):
        raise AssertionError("circular switch count changed under circular shift")
    if circular_dwell_signature(shifted) != circular_dwell_signature(values):
        raise AssertionError("circular dwell durations changed under circular shift")
    return shifted


def _continuous_runs(modes: np.ndarray) -> list[np.ndarray]:
    continuous = np.asarray(modes, dtype=int) == 1
    starts = np.flatnonzero(continuous & ~np.r_[False, continuous[:-1]])
    stops = np.flatnonzero(continuous & ~np.r_[continuous[1:], False]) + 1
    return [np.arange(start, stop, dtype=int) for start, stop in zip(starts, stops, strict=True)]


def evaluate_mode_segmentation(
    event_bins: pd.DataFrame,
    modes: np.ndarray,
    *,
    primitive_ids: Sequence[str],
    primitive_paths: dict[str, np.ndarray],
    route_ids: Sequence[str],
    route_paths: dict[str, np.ndarray],
    routes_by_id: pd.DataFrame,
    minimum_bout_bins: int = 3,
    minimum_bout_path_cm: float = 10.0,
) -> dict[str, object]:
    """Measure route alignment and composition for one mode segmentation."""

    ordered = event_bins.sort_values("time_bin").reset_index(drop=True)
    xy = ordered[["imm_posterior_mean_x_cm", "imm_posterior_mean_y_cm"]].to_numpy(
        dtype=float
    )
    if len(xy) != len(modes):
        raise ValueError("mode timeline must have one value per posterior time bin")
    bouts: list[dict[str, object]] = []
    for indices in _continuous_runs(modes):
        path = xy[indices]
        length = path_length(path)
        if len(indices) < int(minimum_bout_bins) or length < float(minimum_bout_path_cm):
            continue
        primitive_id, primitive_distance = best_path_fit(
            path,
            primitive_ids,
            primitive_paths,
        )
        route_id, route_distance = best_path_fit(path, route_ids, route_paths)
        route_class = ""
        if route_id:
            route = routes_by_id.loc[route_id]
            route_class = (
                f"{int(route['origin_well_id'])}->{int(route['destination_well_id'])}"
            )
        bouts.append(
            {
                "n_bins": int(len(indices)),
                "primitive_id": primitive_id,
                "primitive_distance_cm": primitive_distance,
                "route_id": route_id,
                "route_class": route_class,
                "route_distance_cm": route_distance,
            }
        )
    continuous_path = xy[np.asarray(modes, dtype=int) == 1]
    _, whole_route_distance = (
        best_path_fit(continuous_path, route_ids, route_paths)
        if len(continuous_path) >= 2
        and path_length(continuous_path) >= float(minimum_bout_path_cm)
        else ("", np.nan)
    )
    weighted_bout_distance = (
        float(
            np.average(
                [float(row["primitive_distance_cm"]) for row in bouts],
                weights=[int(row["n_bins"]) for row in bouts],
            )
        )
        if bouts
        else np.nan
    )
    route_classes = [str(row["route_class"]) for row in bouts]
    route_changes = int(
        sum(
            left != right
            for left, right in zip(route_classes[:-1], route_classes[1:], strict=True)
        )
    )
    evaluable = bool(
        len(bouts) >= 2
        and np.isfinite(weighted_bout_distance)
        and np.isfinite(whole_route_distance)
    )
    return {
        "eligible_continuous_bout_count": int(len(bouts)),
        "distinct_route_classes": int(len(set(route_classes))),
        "route_identity_changes": route_changes,
        "switch_alignment": (
            float(route_changes / (len(route_classes) - 1))
            if len(route_classes) >= 2
            else np.nan
        ),
        "composition_index": (
            float(whole_route_distance - weighted_bout_distance) if evaluable else np.nan
        ),
        "composition_evaluable": evaluable,
    }


def _event_offsets(n_bins: int, n_shifts: int, rng: np.random.Generator) -> np.ndarray:
    available = np.arange(1, int(n_bins), dtype=int)
    if len(available) <= int(n_shifts):
        return available
    return np.sort(rng.choice(available, size=int(n_shifts), replace=False))


def build_scores(
    *,
    event_metrics: pd.DataFrame,
    posterior_bins: pd.DataFrame,
    routes: pd.DataFrame,
    route_points: pd.DataFrame,
    primitives: pd.DataFrame,
    primitive_points: pd.DataFrame,
    eligibility: pd.DataFrame,
    n_shifts: int = 50,
    seed: int = 1,
    minimum_bout_bins: int = 3,
    minimum_bout_path_cm: float = 10.0,
) -> pd.DataFrame:
    selected = event_metrics[event_metrics["composition_evaluable"].astype(bool)].copy()
    keys = set(zip(selected["session"].astype(str), selected["event_index"].astype(int)))
    route_paths = _path_dictionary(route_points, id_column="route_id")
    primitive_paths = _path_dictionary(primitive_points, id_column="primitive_id")
    routes_by_id = routes.set_index("route_id", drop=False)
    eligibility_lookup = eligibility.set_index(["session", "event_index"], drop=False)
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(seed)
    for (session, event_index), bins in posterior_bins.groupby(
        ["session", "event_index"], sort=True
    ):
        key = (str(session), int(event_index))
        if key not in keys:
            continue
        context = eligibility_lookup.loc[key]
        fold = int(context["excluded_cv_fold"])
        primitive_ids = _primitive_library_ids(
            primitives,
            session=str(session),
            excluded_fold=fold,
        )
        route_ids = _training_route_ids(
            routes,
            session=str(session),
            excluded_fold=fold,
        )
        ordered = bins.sort_values("time_bin").reset_index(drop=True)
        original_modes = ordered["map_mode_index"].to_numpy(dtype=int)
        conditions = [("original", 0, original_modes)]
        conditions.extend(
            (
                "circular_boundary_shift",
                int(offset),
                shifted_mode_timeline(original_modes, int(offset)),
            )
            for offset in _event_offsets(len(original_modes), int(n_shifts), rng)
        )
        for condition, offset, modes in conditions:
            metrics = evaluate_mode_segmentation(
                ordered,
                modes,
                primitive_ids=primitive_ids,
                primitive_paths=primitive_paths,
                route_ids=route_ids,
                route_paths=route_paths,
                routes_by_id=routes_by_id,
                minimum_bout_bins=int(minimum_bout_bins),
                minimum_bout_path_cm=float(minimum_bout_path_cm),
            )
            rows.append(
                {
                    "session": str(session),
                    "rat": str(session).split("/", 1)[0],
                    "event_index": int(event_index),
                    "condition": condition,
                    "circular_offset_bins": int(offset),
                    "n_bins": int(len(ordered)),
                    "mode_counts_preserved": bool(
                        np.array_equal(
                            np.bincount(modes, minlength=3),
                            np.bincount(original_modes, minlength=3),
                        )
                    ),
                    "circular_switch_count_preserved": bool(
                        circular_switch_count(modes)
                        == circular_switch_count(original_modes)
                    ),
                    "circular_dwell_signature_preserved": bool(
                        circular_dwell_signature(modes)
                        == circular_dwell_signature(original_modes)
                    ),
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def build_decisions(scores: pd.DataFrame, event_metrics: pd.DataFrame) -> pd.DataFrame:
    reference = event_metrics.set_index(["session", "event_index"], drop=False)
    rows: list[dict[str, object]] = []
    for key, group in scores.groupby(["session", "event_index"], sort=True):
        original = group[group["condition"].eq("original")].iloc[0]
        null = group[group["condition"].eq("circular_boundary_shift")]
        null_alignment = pd.to_numeric(null["switch_alignment"], errors="coerce").dropna()
        null_composition = pd.to_numeric(null["composition_index"], errors="coerce").dropna()
        observed_alignment = float(original["switch_alignment"])
        observed_composition = float(original["composition_index"])
        ref = reference.loc[(str(key[0]), int(key[1]))]
        rows.append(
            {
                "session": str(key[0]),
                "rat": str(key[0]).split("/", 1)[0],
                "event_index": int(key[1]),
                "analysis_role": ref["analysis_role"],
                "delta_momentum_minus_imm": float(ref["delta_momentum_minus_imm"]),
                "original_switch_alignment": observed_alignment,
                "reference_switch_alignment": float(ref["switch_alignment"]),
                "median_shifted_switch_alignment": float(null_alignment.median())
                if len(null_alignment)
                else np.nan,
                "switch_alignment_advantage": observed_alignment
                - float(null_alignment.median())
                if len(null_alignment)
                else np.nan,
                "switch_alignment_empirical_p": float(
                    (1 + int(np.sum(null_alignment >= observed_alignment)))
                    / (1 + len(null_alignment))
                )
                if len(null_alignment)
                else np.nan,
                "original_composition_index": observed_composition,
                "reference_composition_index": float(ref["composition_index"]),
                "median_shifted_composition_index": float(null_composition.median())
                if len(null_composition)
                else np.nan,
                "composition_boundary_advantage": observed_composition
                - float(null_composition.median())
                if len(null_composition)
                else np.nan,
                "composition_empirical_p": float(
                    (1 + int(np.sum(null_composition >= observed_composition)))
                    / (1 + len(null_composition))
                )
                if len(null_composition)
                else np.nan,
                "valid_alignment_shifts": int(len(null_alignment)),
                "valid_composition_shifts": int(len(null_composition)),
            }
        )
    return pd.DataFrame(rows)


def _rat_bootstrap_ci(
    decisions: pd.DataFrame,
    column: str,
    *,
    replicates: int,
    seed: int,
) -> tuple[float, float]:
    rats = sorted(decisions["rat"].astype(str).unique())
    if len(rats) < 2:
        return np.nan, np.nan
    groups = {rat: decisions[decisions["rat"].astype(str).eq(rat)] for rat in rats}
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(int(replicates)):
        sample = pd.concat(
            [groups[str(rat)] for rat in rng.choice(rats, size=len(rats), replace=True)],
            ignore_index=True,
        )
        value = pd.to_numeric(sample[column], errors="coerce").median()
        if np.isfinite(value):
            values.append(float(value))
    return (
        (float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975)))
        if values
        else (np.nan, np.nan)
    )


def summarize_decisions(
    decisions: pd.DataFrame,
    *,
    bootstrap_replicates: int = 2000,
    seed: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for metric in ("switch_alignment_advantage", "composition_boundary_advantage"):
        values = pd.to_numeric(decisions[metric], errors="coerce").dropna()
        low, high = _rat_bootstrap_ci(
            decisions,
            metric,
            replicates=int(bootstrap_replicates),
            seed=int(seed + len(rows)),
        )
        rows.append(
            {
                "metric": metric,
                "events": int(len(values)),
                "rats": int(decisions.loc[values.index, "rat"].nunique()),
                "median_advantage": float(values.median()) if len(values) else np.nan,
                "mean_advantage": float(values.mean()) if len(values) else np.nan,
                "positive_fraction": float(np.mean(values > 0.0)) if len(values) else np.nan,
                "rat_bootstrap_ci_low": low,
                "rat_bootstrap_ci_high": high,
            }
        )
    by_rat = (
        decisions.groupby("rat", as_index=False)
        .agg(
            events=("event_index", "count"),
            median_switch_alignment_advantage=("switch_alignment_advantage", "median"),
            median_composition_boundary_advantage=(
                "composition_boundary_advantage",
                "median",
            ),
        )
        .sort_values("rat")
    )
    return pd.DataFrame(rows), by_rat


def build_gate_summary(
    scores: pd.DataFrame,
    decisions: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    expected_events: int,
    minimum_events: int = 20,
    reproduction_tolerance: float = 1e-6,
) -> pd.DataFrame:
    alignment_error = pd.to_numeric(
        decisions["original_switch_alignment"] - decisions["reference_switch_alignment"],
        errors="coerce",
    ).abs()
    composition_error = pd.to_numeric(
        decisions["original_composition_index"] - decisions["reference_composition_index"],
        errors="coerce",
    ).abs()
    alignment = summary[summary["metric"].eq("switch_alignment_advantage")]
    estimate = float(alignment["median_advantage"].iloc[0]) if not alignment.empty else np.nan
    low = float(alignment["rat_bootstrap_ci_low"].iloc[0]) if not alignment.empty else np.nan
    by_rat = decisions.groupby("rat")["switch_alignment_advantage"].median()
    gates = [
        (
            "all_composition_events_present",
            len(decisions) == int(expected_events) and expected_events > 0,
            int(len(decisions)),
            int(expected_events),
        ),
        ("boundary_cohort_size", len(decisions) >= int(minimum_events), len(decisions), f">={minimum_events}"),
        (
            "actual_metrics_reproduced",
            bool(
                len(alignment_error)
                and alignment_error.max() <= float(reproduction_tolerance)
                and composition_error.max() <= float(reproduction_tolerance)
            ),
            float(max(alignment_error.max(), composition_error.max())),
            f"<={reproduction_tolerance}",
        ),
        (
            "mode_counts_and_circular_dwell_structure_preserved",
            bool(
                not scores.empty
                and scores[
                    [
                        "mode_counts_preserved",
                        "circular_switch_count_preserved",
                        "circular_dwell_signature_preserved",
                    ]
                ].all().all()
            ),
            True,
            True,
        ),
        (
            "shifted_alignment_available_for_most_events",
            float(np.mean(decisions["valid_alignment_shifts"] > 0)) >= 0.8,
            float(np.mean(decisions["valid_alignment_shifts"] > 0)),
            ">=0.8",
        ),
        (
            "median_switch_alignment_advantage_positive",
            bool(np.isfinite(estimate) and estimate > 0.0),
            estimate,
            ">0",
        ),
        (
            "rat_bootstrap_switch_alignment_ci_excludes_zero",
            bool(np.isfinite(low) and low > 0.0),
            low,
            ">0",
        ),
        (
            "all_rat_median_switch_alignment_advantages_positive",
            len(by_rat) == 4 and bool((by_rat > 0.0).all()),
            int((by_rat > 0.0).sum()),
            4,
        ),
    ]
    rows = [
        {"gate": gate, "passed": bool(passed), "value": value, "required": required}
        for gate, passed, value, required in gates
    ]
    rows.append(
        {
            "gate": "boundary_specificity_supported",
            "passed": bool(all(row["passed"] for row in rows)),
            "value": int(sum(bool(row["passed"]) for row in rows)),
            "required": len(rows),
        }
    )
    return pd.DataFrame(rows)


def run_analysis(
    *,
    event_metrics_csv: str | Path,
    posterior_dir: str | Path,
    route_dir: str | Path,
    output_dir: str | Path,
    n_shifts: int = 50,
    seed: int = 1,
    bootstrap_replicates: int = 2000,
    minimum_bout_bins: int = 3,
    minimum_bout_path_cm: float = 10.0,
) -> dict[str, Path]:
    event_metrics = pd.read_csv(event_metrics_csv)
    posterior_root = Path(posterior_dir)
    route_root = Path(route_dir)
    input_paths = {
        "event_metrics": Path(event_metrics_csv),
        "posterior_bins": posterior_root
        / "replay_commitment_composition_posterior_bins.csv",
        "routes": route_root / "replay_behavior_route_segments.csv",
        "route_points": route_root / "replay_behavior_route_segment_points.csv",
        "primitives": route_root / "replay_behavior_route_primitives.csv",
        "primitive_points": route_root / "replay_behavior_route_primitive_points.csv",
        "eligibility": route_root / "replay_event_route_library_eligibility.csv",
    }
    loaded = {name: pd.read_csv(path) for name, path in input_paths.items() if name != "event_metrics"}
    scores = build_scores(
        event_metrics=event_metrics,
        posterior_bins=loaded["posterior_bins"],
        routes=loaded["routes"],
        route_points=loaded["route_points"],
        primitives=loaded["primitives"],
        primitive_points=loaded["primitive_points"],
        eligibility=loaded["eligibility"],
        n_shifts=int(n_shifts),
        seed=int(seed),
        minimum_bout_bins=int(minimum_bout_bins),
        minimum_bout_path_cm=float(minimum_bout_path_cm),
    )
    decisions = build_decisions(scores, event_metrics)
    summary, by_rat = summarize_decisions(
        decisions,
        bootstrap_replicates=int(bootstrap_replicates),
        seed=int(seed),
    )
    expected = int(event_metrics["composition_evaluable"].astype(bool).sum())
    gates = build_gate_summary(scores, decisions, summary, expected_events=expected)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, frame in {
        SCORE_OUTPUT: scores,
        DECISION_OUTPUT: decisions,
        SUMMARY_OUTPUT: summary,
        BY_RAT_OUTPUT: by_rat,
        GATE_OUTPUT: gates,
    }.items():
        path = output / name
        frame.to_csv(path, index=False)
        paths[name] = path
    provenance = build_script_provenance(input_paths=input_paths, cwd=ROOT)
    manifest = {
        "analysis": "replay_commitment_composition_boundary_shift_null",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "null": "circular_shift_complete_MAP_IMM_mode_timeline_relative_to_posterior_path",
        "preserves": ["mode_counts", "circular_switch_count", "circular_mode_dwell_signature"],
        "n_shifts": int(n_shifts),
        "seed": int(seed),
        "minimum_bout_bins": int(minimum_bout_bins),
        "minimum_bout_path_cm": float(minimum_bout_path_cm),
        "provenance": provenance,
    }
    manifest_path = output / MANIFEST_OUTPUT
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    paths[MANIFEST_OUTPUT] = manifest_path
    supported = bool(
        gates.loc[gates["gate"].eq("boundary_specificity_supported"), "passed"].iloc[0]
    )
    alignment_row = summary[summary["metric"].eq("switch_alignment_advantage")].iloc[0]
    report = [
        "# Replay commitment/composition boundary-shift null",
        "",
        f"- Composition-evaluable events: {len(decisions)}",
        f"- Median route-class alignment advantage: {float(alignment_row['median_advantage']):.4g}",
        f"- Rat-bootstrap 95% CI: [{float(alignment_row['rat_bootstrap_ci_low']):.4g}, {float(alignment_row['rat_bootstrap_ci_high']):.4g}]",
        f"- Boundary specificity: {'SUPPORTED' if supported else 'NOT ESTABLISHED'}",
        "",
        "The null rotates the complete MAP IMM mode sequence relative to the decoded path. It preserves mode occupancy, circular switch count, and circular dwell durations. No replay-model evidence is used to define behavioral route classes.",
    ]
    report_path = output / REPORT_OUTPUT
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    paths[REPORT_OUTPUT] = report_path
    return paths


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-metrics", required=True)
    parser.add_argument("--posterior-dir", required=True)
    parser.add_argument("--route-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-shifts", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--minimum-bout-bins", type=int, default=3)
    parser.add_argument("--minimum-bout-path-cm", type=float, default=10.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_analysis(
        event_metrics_csv=args.event_metrics,
        posterior_dir=args.posterior_dir,
        route_dir=args.route_dir,
        output_dir=args.output_dir,
        n_shifts=args.n_shifts,
        seed=args.seed,
        bootstrap_replicates=args.bootstrap_replicates,
        minimum_bout_bins=args.minimum_bout_bins,
        minimum_bout_path_cm=args.minimum_bout_path_cm,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
