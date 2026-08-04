#!/usr/bin/env python3
"""Test whether behavior-defined policy branching predicts PF IMM mode changes."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _provenance import build_script_provenance  # noqa: E402


KEYS = ["session", "rat", "event_index"]
TRANSITION_OUTPUT = "pf_policy_branching_transition_rows.csv"
EVENT_OUTPUT = "pf_policy_branching_event_summary.csv"
TEST_OUTPUT = "pf_policy_branching_hypothesis_test.csv"
BY_RAT_OUTPUT = "pf_policy_branching_by_rat.csv"
LOO_OUTPUT = "pf_policy_branching_leave_one_rat_out.csv"
NULL_OUTPUT = "pf_policy_branching_circular_shift_null.csv"
GATE_OUTPUT = "pf_policy_branching_gate_summary.csv"
REPORT_OUTPUT = "pf_policy_branching_report.md"
MANIFEST_OUTPUT = "pf_policy_branching_manifest.json"


def build_branching_field(transition_graph: pd.DataFrame) -> pd.DataFrame:
    """Compute behavior-only outgoing transition entropy at each CV spatial bin."""

    group_columns = [
        "session",
        "rat",
        "excluded_cv_fold",
        "from_x_bin",
        "from_y_bin",
        "from_x_cm",
        "from_y_cm",
    ]
    rows: list[dict[str, object]] = []
    for key, group in transition_graph.groupby(group_columns, sort=True, dropna=False):
        probabilities = pd.to_numeric(
            group["transition_probability"], errors="coerce"
        ).to_numpy(dtype=float)
        probabilities = probabilities[np.isfinite(probabilities) & (probabilities > 0.0)]
        if not len(probabilities):
            continue
        probabilities /= probabilities.sum()
        rows.append(
            {
                **dict(zip(group_columns, key, strict=True)),
                "branch_entropy_nats": float(-np.sum(probabilities * np.log(probabilities))),
                "effective_outgoing_actions": float(np.exp(-np.sum(probabilities * np.log(probabilities)))),
                "observed_out_degree": int(group["observed_out_degree"].max()),
                "outgoing_transition_rows": int(len(group)),
            }
        )
    return pd.DataFrame(rows)


def map_branching_to_replay_transitions(
    transitions: pd.DataFrame,
    posterior_bins: pd.DataFrame,
    event_context: pd.DataFrame,
    branching: pd.DataFrame,
    *,
    maximum_mapping_distance_cm: float = 15.0,
) -> pd.DataFrame:
    """Map CV branching entropy to emission-only represented transition positions."""

    context = event_context[[*KEYS, "excluded_cv_fold"]].drop_duplicates(KEYS)
    if context.duplicated(KEYS).any():
        raise ValueError("event context must contain one excluded fold per event")
    bin_columns = [
        *KEYS,
        "time_bin",
        "emission_only_mean_x_cm",
        "emission_only_mean_y_cm",
    ]
    bins = posterior_bins[bin_columns].copy()
    source = bins.rename(
        columns={
            "time_bin": "transition_index",
            "emission_only_mean_x_cm": "source_represented_x_cm",
            "emission_only_mean_y_cm": "source_represented_y_cm",
        }
    )
    destination = bins.copy()
    destination["time_bin"] = destination["time_bin"] - 1
    destination = destination.rename(
        columns={
            "time_bin": "transition_index",
            "emission_only_mean_x_cm": "destination_represented_x_cm",
            "emission_only_mean_y_cm": "destination_represented_y_cm",
        }
    )
    frame = (
        transitions.merge(source, on=[*KEYS, "transition_index"], validate="one_to_one")
        .merge(destination, on=[*KEYS, "transition_index"], validate="one_to_one")
        .merge(context, on=KEYS, validate="many_to_one")
    )
    frame["represented_x_cm"] = 0.5 * (
        frame["source_represented_x_cm"] + frame["destination_represented_x_cm"]
    )
    frame["represented_y_cm"] = 0.5 * (
        frame["source_represented_y_cm"] + frame["destination_represented_y_cm"]
    )
    fields = {
        (str(session), int(fold)): group.reset_index(drop=True)
        for (session, fold), group in branching.groupby(
            ["session", "excluded_cv_fold"], sort=False
        )
    }
    mapped_rows: list[dict[str, object]] = []
    for row in frame.itertuples(index=False):
        field = fields.get((str(row.session), int(row.excluded_cv_fold)))
        if field is None or field.empty:
            mapped_rows.append(
                {
                    "branch_entropy_nats": np.nan,
                    "effective_outgoing_actions": np.nan,
                    "observed_out_degree": np.nan,
                    "branch_mapping_distance_cm": np.nan,
                    "branch_mapping_valid": False,
                }
            )
            continue
        xy = field[["from_x_cm", "from_y_cm"]].to_numpy(dtype=float)
        target = np.array([row.represented_x_cm, row.represented_y_cm], dtype=float)
        distances = np.linalg.norm(xy - target, axis=1)
        nearest = int(np.argmin(distances))
        distance = float(distances[nearest])
        valid = bool(np.isfinite(distance) and distance <= maximum_mapping_distance_cm)
        mapped_rows.append(
            {
                "branch_entropy_nats": (
                    float(field.loc[nearest, "branch_entropy_nats"]) if valid else np.nan
                ),
                "effective_outgoing_actions": (
                    float(field.loc[nearest, "effective_outgoing_actions"]) if valid else np.nan
                ),
                "observed_out_degree": (
                    int(field.loc[nearest, "observed_out_degree"]) if valid else np.nan
                ),
                "branch_mapping_distance_cm": distance,
                "branch_mapping_valid": valid,
            }
        )
    return pd.concat([frame.reset_index(drop=True), pd.DataFrame(mapped_rows)], axis=1)


def _correlation_at_offset(x_rank: np.ndarray, y_rank: np.ndarray, offset: int) -> float:
    x = np.asarray(x_rank, dtype=float)
    y = np.roll(np.asarray(y_rank, dtype=float), int(offset))
    x -= x.mean()
    y -= y.mean()
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(np.dot(x, y) / denominator) if denominator > 0.0 else np.nan


def event_branching_effects(
    transition_rows: pd.DataFrame,
    *,
    minimum_transitions: int = 8,
) -> tuple[pd.DataFrame, dict[tuple[str, str, int], np.ndarray]]:
    """Compute equal-event Spearman effects and every admissible circular offset."""

    rows: list[dict[str, object]] = []
    offset_effects: dict[tuple[str, str, int], np.ndarray] = {}
    for key, group in transition_rows.groupby(KEYS, sort=True):
        selected = group[
            group["branch_mapping_valid"].astype(bool)
            & np.isfinite(pd.to_numeric(group["branch_entropy_nats"], errors="coerce"))
            & np.isfinite(
                pd.to_numeric(
                    group[
                        "stationary_continuous_switch_probability_given_nonfragmented"
                    ],
                    errors="coerce",
                )
            )
        ].sort_values("transition_index")
        entropy = selected["branch_entropy_nats"].to_numpy(dtype=float)
        switch = selected[
            "stationary_continuous_switch_probability_given_nonfragmented"
        ].to_numpy(dtype=float)
        evaluable = bool(
            len(selected) >= int(minimum_transitions)
            and np.unique(entropy).size >= 2
            and np.ptp(switch) > 0.0
        )
        correlations = np.full(max(len(selected), 1), np.nan, dtype=float)
        if evaluable:
            x_rank = rankdata(entropy, method="average")
            y_rank = rankdata(switch, method="average")
            correlations = np.array(
                [_correlation_at_offset(x_rank, y_rank, offset) for offset in range(len(selected))]
            )
            offset_effects[(str(key[0]), str(key[1]), int(key[2]))] = correlations
        rows.append(
            {
                **dict(zip(KEYS, key, strict=True)),
                "mapped_transitions": int(len(selected)),
                "event_evaluable": evaluable,
                "branch_switch_spearman_r": float(correlations[0]),
                "mean_branch_entropy_nats": float(np.mean(entropy)) if len(entropy) else np.nan,
                "mean_nonfragmented_sc_switch_probability": (
                    float(np.mean(switch)) if len(switch) else np.nan
                ),
                "median_branch_mapping_distance_cm": (
                    float(selected["branch_mapping_distance_cm"].median())
                    if len(selected)
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows), offset_effects


def _rat_equal_mean(event_effects: pd.DataFrame, column: str) -> float:
    selected = event_effects[event_effects["event_evaluable"].astype(bool)].dropna(
        subset=[column]
    )
    return float(selected.groupby("rat")[column].mean().mean()) if len(selected) else np.nan


def _rat_bootstrap(
    event_effects: pd.DataFrame,
    *,
    replicates: int,
    seed: int,
) -> tuple[float, float, int]:
    selected = event_effects[event_effects["event_evaluable"].astype(bool)]
    groups = {
        str(rat): group["branch_switch_spearman_r"].dropna().to_numpy(dtype=float)
        for rat, group in selected.groupby("rat", sort=True)
    }
    groups = {rat: values for rat, values in groups.items() if len(values)}
    if len(groups) < 2:
        return np.nan, np.nan, 0
    rats = sorted(groups)
    rng = np.random.default_rng(seed)
    draws = [
        float(np.mean([np.mean(groups[str(rat)]) for rat in rng.choice(rats, len(rats), replace=True)]))
        for _ in range(int(replicates))
    ]
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975)), len(draws)


def run_branching_test(
    event_effects: pd.DataFrame,
    offset_effects: dict[tuple[str, str, int], np.ndarray],
    *,
    permutations: int = 2000,
    bootstraps: int = 2000,
    seed: int = 20260804,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected = event_effects[event_effects["event_evaluable"].astype(bool)].copy()
    estimate = _rat_equal_mean(selected, "branch_switch_spearman_r")
    low, high, completed = _rat_bootstrap(
        selected, replicates=bootstraps, seed=seed
    )
    rng = np.random.default_rng(seed + 1)
    null_rows: list[dict[str, object]] = []
    keys = [
        (str(row.session), str(row.rat), int(row.event_index))
        for row in selected.itertuples(index=False)
    ]
    for replicate in range(int(permutations)):
        values_by_rat: dict[str, list[float]] = {}
        for key in keys:
            correlations = offset_effects[key]
            offset = int(rng.integers(1, len(correlations)))
            values_by_rat.setdefault(key[1], []).append(float(correlations[offset]))
        null_estimate = float(
            np.mean([np.nanmean(values) for values in values_by_rat.values()])
        )
        null_rows.append(
            {
                "replicate": replicate,
                "null_estimate": null_estimate,
                "null_control": "within_event_circular_switch_probability_shift",
            }
        )
    null = pd.DataFrame(null_rows)
    p_value = float(
        (1 + np.sum(null["null_estimate"].to_numpy(dtype=float) >= estimate))
        / (1 + len(null))
    )
    by_rat = (
        selected.groupby("rat", as_index=False)
        .agg(
            events=("event_index", "size"),
            mean_effect=("branch_switch_spearman_r", "mean"),
            median_effect=("branch_switch_spearman_r", "median"),
        )
        .sort_values("rat")
    )
    by_rat["positive_direction"] = by_rat["mean_effect"] > 0.0
    loo_rows = []
    for omitted in sorted(selected["rat"].astype(str).unique()):
        retained = selected[~selected["rat"].astype(str).eq(omitted)]
        value = _rat_equal_mean(retained, "branch_switch_spearman_r")
        loo_rows.append(
            {
                "omitted_rat": omitted,
                "events": int(len(retained)),
                "effect": value,
                "positive_direction": bool(value > 0.0),
            }
        )
    loo = pd.DataFrame(loo_rows)
    status = "inconclusive"
    if np.isfinite(low) and low > 0.0 and p_value <= 0.05:
        status = "directional_pass_unadjusted"
    elif np.isfinite(high) and high < 0.0:
        status = "contradicted"
    test = pd.DataFrame(
        [
            {
                "hypothesis": "H5",
                "test": "behavior_branch_entropy_predicts_nonfragmented_sc_switching",
                "role": "primary",
                "events": int(len(selected)),
                "rats": int(selected["rat"].nunique()),
                "sessions": int(selected["session"].nunique()),
                "estimate": estimate,
                "rat_bootstrap_ci_low": low,
                "rat_bootstrap_ci_high": high,
                "bootstrap_replicates_completed": completed,
                "permutation_p_value": p_value,
                "positive_rats": int(by_rat["positive_direction"].sum()),
                "leave_one_rat_out_positive": bool(loo["positive_direction"].all()),
                "status_before_campaign_fdr": status,
                "null_control": "within_event_circular_switch_probability_shift",
            }
        ]
    )
    return test, by_rat, loo, null, selected


def build_gates(
    transitions: pd.DataFrame,
    mapped: pd.DataFrame,
    event_effects: pd.DataFrame,
    test: pd.DataFrame,
) -> pd.DataFrame:
    mapped_fraction = float(mapped["branch_mapping_valid"].mean()) if len(mapped) else 0.0
    evaluable = event_effects[event_effects["event_evaluable"].astype(bool)]
    rows = [
        ("transition_rows_present", len(transitions) > 0, len(transitions), ">0"),
        (
            "pairwise_transition_posterior_present",
            "stationary_continuous_switch_probability_given_nonfragmented" in transitions,
            int("stationary_continuous_switch_probability_given_nonfragmented" in transitions),
            1,
        ),
        ("branch_mapping_fraction", mapped_fraction >= 0.80, mapped_fraction, ">=0.80"),
        ("evaluable_events", len(evaluable) >= 100, len(evaluable), ">=100"),
        ("all_four_rats", evaluable["rat"].nunique() == 4, evaluable["rat"].nunique(), 4),
        ("all_eight_sessions", evaluable["session"].nunique() == 8, evaluable["session"].nunique(), 8),
        ("primary_test_computed", test["estimate"].notna().all(), int(test["estimate"].notna().sum()), 1),
        ("circular_null_complete", test["permutation_p_value"].notna().all(), int(test["permutation_p_value"].notna().sum()), 1),
    ]
    gates = pd.DataFrame(
        {"gate": gate, "passed": bool(passed), "value": value, "required": required}
        for gate, passed, value, required in rows
    )
    gates.loc[len(gates)] = {
        "gate": "overall_technical",
        "passed": bool(gates["passed"].all()),
        "value": int(gates["passed"].sum()),
        "required": len(gates),
    }
    return gates


def run_analysis(
    *,
    posterior_transitions_csv: str | Path,
    posterior_bins_csv: str | Path,
    event_context_csv: str | Path,
    behavior_transition_graph_csv: str | Path,
    output_dir: str | Path,
    maximum_mapping_distance_cm: float = 15.0,
    minimum_transitions: int = 8,
    permutations: int = 2000,
    bootstraps: int = 2000,
    seed: int = 20260804,
) -> dict[str, Path]:
    inputs = {
        "posterior_transitions_csv": Path(posterior_transitions_csv),
        "posterior_bins_csv": Path(posterior_bins_csv),
        "event_context_csv": Path(event_context_csv),
        "behavior_transition_graph_csv": Path(behavior_transition_graph_csv),
    }
    transitions = pd.read_csv(inputs["posterior_transitions_csv"])
    bins = pd.read_csv(inputs["posterior_bins_csv"])
    context = pd.read_csv(inputs["event_context_csv"])
    graph = pd.read_csv(inputs["behavior_transition_graph_csv"])
    branching = build_branching_field(graph)
    mapped = map_branching_to_replay_transitions(
        transitions,
        bins,
        context,
        branching,
        maximum_mapping_distance_cm=maximum_mapping_distance_cm,
    )
    events, offsets = event_branching_effects(
        mapped, minimum_transitions=minimum_transitions
    )
    test, by_rat, loo, null, _ = run_branching_test(
        events,
        offsets,
        permutations=permutations,
        bootstraps=bootstraps,
        seed=seed,
    )
    gates = build_gates(transitions, mapped, events, test)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    tables = {
        TRANSITION_OUTPUT: mapped,
        EVENT_OUTPUT: events,
        TEST_OUTPUT: test,
        BY_RAT_OUTPUT: by_rat,
        LOO_OUTPUT: loo,
        NULL_OUTPUT: null,
        GATE_OUTPUT: gates,
    }
    paths: dict[str, Path] = {}
    for name, table in tables.items():
        path = output / name
        table.to_csv(path, index=False)
        paths[name] = path
    row = test.iloc[0]
    report = "\n".join(
        [
            "# PF policy branching and IMM switching",
            "",
            f"Technical status: **{'pass' if gates.iloc[-1]['passed'] else 'fail'}**.",
            "",
            "The primary test uses behavior-only cross-validated outgoing-transition entropy and ",
            "the stationary-to-continuous posterior transition probability conditional on neither ",
            "side being fragmented. Whole-event level effects are weighted equally.",
            "",
            f"Rat-equal mean within-event Spearman effect: **{row['estimate']:+.3f}** ",
            f"(rat-bootstrap CI {row['rat_bootstrap_ci_low']:+.3f} to ",
            f"{row['rat_bootstrap_ci_high']:+.3f}; circular-shift p={row['permutation_p_value']:.4f}).",
            f"Positive rats: {int(row['positive_rats'])}/4. Status before campaign FDR: ",
            f"`{row['status_before_campaign_fdr']}`.",
            "",
            "This result is not promoted to supported until the campaign-wide H1-H10 FDR is applied.",
        ]
    )
    report_path = output / REPORT_OUTPUT
    report_path.write_text(report + "\n", encoding="utf-8")
    paths[REPORT_OUTPUT] = report_path
    manifest = {
        "analysis": "pf_policy_branching_imm_switching_h5",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "represented_position": "emission_only_posterior_mean_transition_midpoint",
        "switch_metric": "stationary_continuous_probability_given_nonfragmented",
        "maximum_mapping_distance_cm": float(maximum_mapping_distance_cm),
        "minimum_transitions": int(minimum_transitions),
        "permutations": int(permutations),
        "bootstraps": int(bootstraps),
        "seed": int(seed),
        "campaign_fdr_applied": False,
        "outputs": {name: str(path) for name, path in paths.items()},
        "provenance": build_script_provenance(input_paths=inputs, cwd=ROOT),
    }
    manifest_path = output / MANIFEST_OUTPUT
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    paths[MANIFEST_OUTPUT] = manifest_path
    return paths


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--posterior-transitions", required=True)
    parser.add_argument("--posterior-bins", required=True)
    parser.add_argument("--event-context", required=True)
    parser.add_argument("--behavior-transition-graph", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--maximum-mapping-distance-cm", type=float, default=15.0)
    parser.add_argument("--minimum-transitions", type=int, default=8)
    parser.add_argument("--permutations", type=int, default=2000)
    parser.add_argument("--bootstraps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260804)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_analysis(
        posterior_transitions_csv=args.posterior_transitions,
        posterior_bins_csv=args.posterior_bins,
        event_context_csv=args.event_context,
        behavior_transition_graph_csv=args.behavior_transition_graph,
        output_dir=args.output_dir,
        maximum_mapping_distance_cm=args.maximum_mapping_distance_cm,
        minimum_transitions=args.minimum_transitions,
        permutations=args.permutations,
        bootstraps=args.bootstraps,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
