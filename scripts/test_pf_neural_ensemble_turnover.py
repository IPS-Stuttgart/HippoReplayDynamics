#!/usr/bin/env python3
"""Test held-out neural ensemble turnover at training-defined PF IMM switches."""

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
from audit_pfeiffer_imm_gate_convergence import (  # noqa: E402
    partial_spearman,
    rat_cluster_bootstrap_partial,
)


KEYS = ["session", "rat", "event_index"]
EVENT_OUTPUT = "pf_neural_ensemble_turnover_event_medians.csv"
TEST_OUTPUT = "pf_neural_ensemble_turnover_hypothesis_tests.csv"
BY_RAT_OUTPUT = "pf_neural_ensemble_turnover_by_rat.csv"
LOO_OUTPUT = "pf_neural_ensemble_turnover_leave_one_rat_out.csv"
NULL_OUTPUT = "pf_neural_ensemble_turnover_exchangeability_null.csv"
GATE_OUTPUT = "pf_neural_ensemble_turnover_gate_summary.csv"
REPORT_OUTPUT = "pf_neural_ensemble_turnover_report.md"
MANIFEST_OUTPUT = "pf_neural_ensemble_turnover_manifest.json"

PRIMARY = "heldout_assembly_turnover_excess_event_median"
HELDOUT_DELTA = "heldout_delta_imm_minus_fragmented_event_median"
COMPANION_CONTROLS = (
    "log1p_median_train_cell_count",
    "log1p_median_test_spikes",
    "median_real_imm_train_posterior_entropy",
    "log1p_median_n_time",
)


def build_event_medians(split_scores: pd.DataFrame) -> pd.DataFrame:
    required = {
        "assembly_turnover_evaluable",
        "heldout_assembly_turnover_excess",
        "real_frozen_heldout_delta_imm_minus_fragmented",
    }
    missing = sorted(required - set(split_scores.columns))
    if missing:
        raise ValueError(f"split scores lack H6 instrumentation: {missing}")
    success = split_scores[
        split_scores["status"].astype(str).eq("success")
        & split_scores["assembly_turnover_evaluable"].astype(str).str.lower().isin(
            {"true", "1", "1.0"}
        )
    ].copy()
    rows: list[dict[str, object]] = []
    numeric = (
        "heldout_assembly_turnover_excess",
        "assembly_boundary_heldout_turnover_hellinger",
        "assembly_control_heldout_turnover_median",
        "assembly_boundary_switch_probability",
        "real_frozen_heldout_delta_imm_minus_fragmented",
        "train_cell_count",
        "test_cell_count",
        "train_spikes",
        "test_spikes",
        "real_imm_train_posterior_entropy",
        "n_time",
    )
    for key, group in success.groupby(KEYS, sort=True):
        medians = {
            column: float(pd.to_numeric(group[column], errors="coerce").median())
            for column in numeric
        }
        row = {
            **dict(zip(KEYS, key, strict=True)),
            "completed_evaluable_splits": int(group["cell_split_index"].nunique()),
            PRIMARY: medians["heldout_assembly_turnover_excess"],
            "boundary_turnover_hellinger_event_median": medians[
                "assembly_boundary_heldout_turnover_hellinger"
            ],
            "control_turnover_hellinger_event_median": medians[
                "assembly_control_heldout_turnover_median"
            ],
            "boundary_switch_probability_event_median": medians[
                "assembly_boundary_switch_probability"
            ],
            HELDOUT_DELTA: medians[
                "real_frozen_heldout_delta_imm_minus_fragmented"
            ],
            "median_train_cell_count": medians["train_cell_count"],
            "median_test_cell_count": medians["test_cell_count"],
            "median_train_spikes": medians["train_spikes"],
            "median_test_spikes": medians["test_spikes"],
            "median_real_imm_train_posterior_entropy": medians[
                "real_imm_train_posterior_entropy"
            ],
            "median_n_time": medians["n_time"],
        }
        row["log1p_median_train_cell_count"] = float(
            np.log1p(row["median_train_cell_count"])
        )
        row["log1p_median_test_spikes"] = float(
            np.log1p(row["median_test_spikes"])
        )
        row["log1p_median_n_time"] = float(np.log1p(row["median_n_time"]))
        rows.append(row)
    return pd.DataFrame(rows)


def _rat_equal_mean(events: pd.DataFrame, column: str = PRIMARY) -> float:
    selected = events.dropna(subset=[column])
    return float(selected.groupby("rat")[column].mean().mean()) if len(selected) else np.nan


def _rat_bootstrap(
    events: pd.DataFrame,
    *,
    column: str,
    replicates: int,
    seed: int,
) -> tuple[float, float, int]:
    groups = {
        str(rat): group[column].dropna().to_numpy(dtype=float)
        for rat, group in events.groupby("rat", sort=True)
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


def exchangeability_null(
    split_scores: pd.DataFrame,
    *,
    permutations: int,
    seed: int,
) -> pd.DataFrame:
    selected = split_scores[
        split_scores["status"].astype(str).eq("success")
        & split_scores["assembly_turnover_evaluable"].astype(str).str.lower().isin(
            {"true", "1", "1.0"}
        )
    ].copy()
    slots: list[tuple[tuple[str, str, int, int], np.ndarray]] = []
    for row in selected.itertuples(index=False):
        controls = np.asarray(
            json.loads(str(row.assembly_control_heldout_turnovers_json)), dtype=float
        )
        values = np.concatenate(
            [[float(row.assembly_boundary_heldout_turnover_hellinger)], controls]
        )
        if len(values) >= 2 and np.isfinite(values).all():
            slots.append(
                (
                    (
                        str(row.session),
                        str(row.rat),
                        int(row.event_index),
                        int(row.cell_split_index),
                    ),
                    values,
                )
            )
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for replicate in range(int(permutations)):
        split_rows = []
        for key, values in slots:
            pseudo_boundary = int(rng.integers(0, len(values)))
            controls = np.delete(values, pseudo_boundary)
            split_rows.append(
                {
                    "session": key[0],
                    "rat": key[1],
                    "event_index": key[2],
                    "cell_split_index": key[3],
                    "pseudo_excess": float(values[pseudo_boundary] - np.median(controls)),
                }
            )
        split_frame = pd.DataFrame(split_rows)
        event_frame = (
            split_frame.groupby(KEYS, as_index=False)["pseudo_excess"].median()
            if len(split_frame)
            else pd.DataFrame(columns=[*KEYS, "pseudo_excess"])
        )
        rows.append(
            {
                "replicate": replicate,
                "null_estimate": _rat_equal_mean(event_frame, "pseudo_excess"),
                "null_control": "within_split_boundary_control_label_exchange",
            }
        )
    return pd.DataFrame(rows)


def run_tests(
    split_scores: pd.DataFrame,
    events: pd.DataFrame,
    *,
    permutations: int = 2000,
    bootstraps: int = 2000,
    seed: int = 20260804,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    estimate = _rat_equal_mean(events)
    low, high, completed = _rat_bootstrap(
        events, column=PRIMARY, replicates=bootstraps, seed=seed
    )
    null = exchangeability_null(
        split_scores, permutations=permutations, seed=seed + 1
    )
    null_values = pd.to_numeric(null["null_estimate"], errors="coerce").dropna()
    primary_p = float(
        (1 + int((null_values >= estimate).sum())) / (1 + len(null_values))
    )
    by_rat = (
        events.groupby("rat", as_index=False)
        .agg(
            events=("event_index", "size"),
            mean_turnover_excess=(PRIMARY, "mean"),
            median_turnover_excess=(PRIMARY, "median"),
        )
        .sort_values("rat")
    )
    by_rat["positive_direction"] = by_rat["mean_turnover_excess"] > 0.0
    loo_rows = []
    for omitted in sorted(events["rat"].astype(str).unique()):
        retained = events[~events["rat"].astype(str).eq(omitted)]
        value = _rat_equal_mean(retained)
        loo_rows.append(
            {
                "omitted_rat": omitted,
                "events": int(len(retained)),
                "effect": value,
                "positive_direction": bool(value > 0.0),
            }
        )
    loo = pd.DataFrame(loo_rows)
    primary_status = "inconclusive"
    if low > 0.0 and primary_p <= 0.05:
        primary_status = "directional_pass_unadjusted"
    elif high < 0.0:
        primary_status = "contradicted"

    companion, _, _, _ = partial_spearman(
        events,
        PRIMARY,
        HELDOUT_DELTA,
        list(COMPANION_CONTROLS),
        rat_fixed_effects=True,
    )
    companion_low, companion_high, companion_positive_fraction, companion_completed = (
        rat_cluster_bootstrap_partial(
            events,
            PRIMARY,
            HELDOUT_DELTA,
            list(COMPANION_CONTROLS),
            replicates=bootstraps,
            seed=seed + 2,
        )
    )
    tests = pd.DataFrame(
        [
            {
                "hypothesis": "H6",
                "test": "heldout_assembly_turnover_at_training_defined_switch",
                "role": "primary",
                "events": int(len(events)),
                "rats": int(events["rat"].nunique()),
                "sessions": int(events["session"].nunique()),
                "estimate": estimate,
                "rat_bootstrap_ci_low": low,
                "rat_bootstrap_ci_high": high,
                "bootstrap_replicates_completed": completed,
                "permutation_p_value": primary_p,
                "positive_rats": int(by_rat["positive_direction"].sum()),
                "leave_one_rat_out_positive": bool(loo["positive_direction"].all()),
                "status_before_campaign_fdr": primary_status,
                "null_control": "within_split_boundary_control_label_exchange",
            },
            {
                "hypothesis": "H6",
                "test": "turnover_excess_predicts_heldout_imm_advantage",
                "role": "companion",
                "events": int(len(events)),
                "rats": int(events["rat"].nunique()),
                "sessions": int(events["session"].nunique()),
                "estimate": companion,
                "rat_bootstrap_ci_low": companion_low,
                "rat_bootstrap_ci_high": companion_high,
                "bootstrap_replicates_completed": companion_completed,
                "bootstrap_positive_fraction": companion_positive_fraction,
                "permutation_p_value": np.nan,
                "positive_rats": np.nan,
                "leave_one_rat_out_positive": np.nan,
                "status_before_campaign_fdr": (
                    "directional_pass_unadjusted"
                    if np.isfinite(companion_low) and companion_low > 0.0
                    else "inconclusive"
                ),
                "null_control": "quality_adjusted_rat_fixed_effect_partial_spearman",
            },
        ]
    )
    return tests, by_rat, loo, null


def build_gates(
    split_scores: pd.DataFrame,
    events: pd.DataFrame,
    tests: pd.DataFrame,
    *,
    minimum_events: int = 80,
    minimum_splits_per_event: int = 5,
) -> pd.DataFrame:
    primary = tests[tests["role"].eq("primary")]
    no_leakage = (
        not split_scores["heldout_replay_spikes_used_for_latent_inference"].astype(bool).any()
        if "heldout_replay_spikes_used_for_latent_inference" in split_scores
        else False
    )
    rows = [
        ("h6_instrumentation_present", "assembly_turnover_evaluable" in split_scores, int("assembly_turnover_evaluable" in split_scores), 1),
        ("heldout_never_used_for_latent_inference", no_leakage, int(no_leakage), 1),
        ("evaluable_events", len(events) >= minimum_events, len(events), f">={minimum_events}"),
        ("minimum_repeated_splits", events["completed_evaluable_splits"].ge(minimum_splits_per_event).all(), int(events["completed_evaluable_splits"].min()) if len(events) else 0, f">={minimum_splits_per_event}"),
        ("all_four_rats", events["rat"].nunique() == 4, events["rat"].nunique(), 4),
        ("all_eight_sessions", events["session"].nunique() == 8, events["session"].nunique(), 8),
        ("primary_test_computed", len(primary) == 1 and primary["estimate"].notna().all(), int(primary["estimate"].notna().sum()), 1),
        ("exchangeability_null_complete", primary["permutation_p_value"].notna().all(), int(primary["permutation_p_value"].notna().sum()), 1),
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
    split_scores_csv: str | Path,
    output_dir: str | Path,
    permutations: int = 2000,
    bootstraps: int = 2000,
    seed: int = 20260804,
) -> dict[str, Path]:
    input_path = Path(split_scores_csv)
    splits = pd.read_csv(input_path)
    events = build_event_medians(splits)
    tests, by_rat, loo, null = run_tests(
        splits,
        events,
        permutations=permutations,
        bootstraps=bootstraps,
        seed=seed,
    )
    gates = build_gates(splits, events, tests)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    tables = {
        EVENT_OUTPUT: events,
        TEST_OUTPUT: tests,
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
    primary = tests[tests["role"].eq("primary")].iloc[0]
    companion = tests[tests["role"].eq("companion")].iloc[0]
    report = "\n".join(
        [
            "# PF neural ensemble turnover at IMM switches",
            "",
            f"Technical status: **{'pass' if gates.iloc[-1]['passed'] else 'fail'}**.",
            "",
            "Switch boundaries and matched controls were selected using training cells only. ",
            "Held-out replay spikes were used only to measure population-vector turnover and ",
            "frozen-posterior predictive score.",
            "",
            (
                f"Primary held-out Hellinger-turnover excess: **{primary['estimate']:+.3f}** "
                f"(rat-bootstrap CI {primary['rat_bootstrap_ci_low']:+.3f} to "
                f"{primary['rat_bootstrap_ci_high']:+.3f}; exchangeability p="
                f"{primary['permutation_p_value']:.4f}; positive rats "
                f"{int(primary['positive_rats'])}/4)."
            ),
            (
                "Companion adjusted association with held-out IMM-minus-fragmented score: "
                f"{companion['estimate']:+.3f} (CI "
                f"{companion['rat_bootstrap_ci_low']:+.3f} to "
                f"{companion['rat_bootstrap_ci_high']:+.3f})."
            ),
            "",
            "Campaign-wide FDR has not yet been applied.",
        ]
    )
    report_path = output / REPORT_OUTPUT
    report_path.write_text(report + "\n", encoding="utf-8")
    paths[REPORT_OUTPUT] = report_path
    manifest = {
        "analysis": "pf_neural_ensemble_turnover_h6",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "boundary_definition": "training_cells_only_strongest_nonfragmented_sc_probability_ge_0p5",
        "control_definition": "same_event_training_spike_support_matched_sc_probability_le_0p25",
        "heldout_turnover_metric": "hellinger_distance_three_bin_population_vectors",
        "permutations": int(permutations),
        "bootstraps": int(bootstraps),
        "seed": int(seed),
        "campaign_fdr_applied": False,
        "outputs": {name: str(path) for name, path in paths.items()},
        "provenance": build_script_provenance(
            input_paths={"split_scores_csv": input_path}, cwd=ROOT
        ),
    }
    manifest_path = output / MANIFEST_OUTPUT
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    paths[MANIFEST_OUTPUT] = manifest_path
    return paths


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-scores", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--permutations", type=int, default=2000)
    parser.add_argument("--bootstraps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260804)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_analysis(
        split_scores_csv=args.split_scores,
        output_dir=args.output_dir,
        permutations=args.permutations,
        bootstraps=args.bootstraps,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
