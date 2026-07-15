#!/usr/bin/env python3
"""Audit whether Pfeiffer/Foster IMM gates converge beyond decoder quality.

This script is intentionally non-rescoring. It joins the event-level Gate 4
held-out predictions, Gate 2 order-by-map pilot, Gate 3 posterior content, and
map-permutation decisions. Associations are exploratory: they were proposed
after the gate results were inspected and do not replace any predeclared gate.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


KEYS = ["session", "rat", "event_index"]
CORE_CONTROLS = [
    "log1p_train_cell_count",
    "log1p_heldout_spikes",
    "train_imm_posterior_entropy",
]
EXTENDED_CONTROLS = [*CORE_CONTROLS, "log1p_time_bins"]

PAIR_SPECS = (
    {
        "analysis_id": "heldout_vs_order_advantage",
        "x": "gate2_real_order_advantage",
        "y": "heldout_delta_imm_minus_fragmented",
        "selection_scope": "frozen_clean_imm_gate2_pilot",
        "expected_events": 20,
        "note": "Exploratory and range-restricted to the frozen Gate 2 clean-IMM pilot.",
    },
    {
        "analysis_id": "heldout_vs_nonstationary_mass",
        "x": "gate3_mean_nonstationary_mode_probability",
        "y": "heldout_delta_imm_minus_fragmented",
        "selection_scope": "frozen_clean_imm_108",
        "expected_events": 108,
        "note": "Links posterior nonstationary content to held-out prediction.",
    },
    {
        "analysis_id": "heldout_vs_net_displacement",
        "x": "gate3_posterior_net_displacement_cm",
        "y": "heldout_delta_imm_minus_fragmented",
        "selection_scope": "frozen_clean_imm_108",
        "expected_events": 108,
        "note": "Links posterior displacement to held-out prediction.",
    },
    {
        "analysis_id": "heldout_vs_expected_switch_count",
        "x": "gate3_expected_switch_count",
        "y": "heldout_delta_imm_minus_fragmented",
        "selection_scope": "frozen_clean_imm_108",
        "expected_events": 108,
        "note": "Tests whether held-out prediction scales with expected IMM switching.",
    },
    {
        "analysis_id": "heldout_vs_map_specific_nonstationary_mass",
        "x": "gate3_nonstationary_mass_map_excess",
        "y": "heldout_delta_imm_minus_fragmented",
        "selection_scope": "events_with_posterior_content_108",
        "expected_events": 108,
        "note": "Tests whether map-specific mode content predicts held-out generalization.",
    },
    {
        "analysis_id": "heldout_vs_map_specific_displacement",
        "x": "gate3_displacement_map_excess_cm",
        "y": "heldout_delta_imm_minus_fragmented",
        "selection_scope": "events_with_posterior_content_108",
        "expected_events": 108,
        "note": "Tests whether map-specific displacement predicts held-out generalization.",
    },
    {
        "analysis_id": "margin_map_excess_vs_nonstationary_map_excess",
        "x": "gate3_nonstationary_mass_map_excess",
        "y": "gate1b_margin_map_excess",
        "selection_scope": "events_with_posterior_content_108",
        "expected_events": 108,
        "note": "Tests whether the small map-specific margin excess follows map-specific content.",
    },
    {
        "analysis_id": "margin_map_excess_vs_displacement_map_excess",
        "x": "gate3_displacement_map_excess_cm",
        "y": "gate1b_margin_map_excess",
        "selection_scope": "events_with_posterior_content_108",
        "expected_events": 108,
        "note": "Tests whether the small map-specific margin excess follows map-specific displacement.",
    },
)


def _require(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def _unique(frame: pd.DataFrame, label: str) -> None:
    duplicated = frame.duplicated(KEYS, keep=False)
    if duplicated.any():
        rows = frame.loc[duplicated, KEYS].head().to_dict("records")
        raise ValueError(f"{label} has duplicate event keys: {rows}")


def build_event_table(
    gate4: pd.DataFrame,
    gate2: pd.DataFrame,
    gate3: pd.DataFrame,
    map_permutation: pd.DataFrame,
) -> pd.DataFrame:
    """Join gate artifacts onto the primary all-event Gate 4 table."""
    _require(
        gate4,
        [
            *KEYS,
            "scope",
            "heldout_delta_event_median",
            "median_train_cell_count",
            "median_test_spikes",
            "median_n_time",
            "median_train_imm_posterior_entropy",
        ],
        "Gate 4",
    )
    gate4 = gate4[gate4["scope"].astype(str).eq("all_events")].copy()
    _unique(gate4, "Gate 4 all-events")

    _require(gate2, [*KEYS, "real_order_advantage", "order_by_map_interaction"], "Gate 2")
    _unique(gate2, "Gate 2")
    gate2 = gate2[
        [*KEYS, "real_order_advantage", "wrong_order_advantage", "order_by_map_interaction"]
    ].rename(
        columns={
            "real_order_advantage": "gate2_real_order_advantage",
            "wrong_order_advantage": "gate2_wrong_order_advantage",
            "order_by_map_interaction": "gate2_order_by_map_interaction",
        }
    )

    _require(
        gate3,
        [
            *KEYS,
            "mean_nonstationary_mode_probability",
            "fraction_time_map_nonstationary",
            "expected_switch_count",
            "map_mode_switch_count",
            "posterior_expected_path_length_cm",
            "posterior_net_displacement_cm",
        ],
        "Gate 3",
    )
    _unique(gate3, "Gate 3")
    gate3 = gate3[
        [
            *KEYS,
            "mean_nonstationary_mode_probability",
            "fraction_time_map_nonstationary",
            "expected_switch_count",
            "map_mode_switch_count",
            "posterior_expected_path_length_cm",
            "posterior_net_displacement_cm",
        ]
    ].rename(
        columns={
            "mean_nonstationary_mode_probability": "gate3_mean_nonstationary_mode_probability",
            "fraction_time_map_nonstationary": "gate3_fraction_time_map_nonstationary",
            "expected_switch_count": "gate3_expected_switch_count",
            "map_mode_switch_count": "gate3_map_mode_switch_count",
            "posterior_expected_path_length_cm": "gate3_posterior_expected_path_length_cm",
            "posterior_net_displacement_cm": "gate3_posterior_net_displacement_cm",
        }
    )

    _require(
        map_permutation,
        [
            *KEYS,
            "real_minus_null_median_delta_imm_minus_fragmented",
            "real_minus_null_median_mean_nonstationary_mode_probability",
            "real_minus_null_median_posterior_expected_path_length_cm",
            "real_minus_null_median_posterior_net_displacement_cm",
        ],
        "map permutation",
    )
    _unique(map_permutation, "map permutation")
    map_permutation = map_permutation[
        [
            *KEYS,
            "real_minus_null_median_delta_imm_minus_fragmented",
            "real_minus_null_median_mean_nonstationary_mode_probability",
            "real_minus_null_median_posterior_expected_path_length_cm",
            "real_minus_null_median_posterior_net_displacement_cm",
        ]
    ].rename(
        columns={
            "real_minus_null_median_delta_imm_minus_fragmented": "gate1b_margin_map_excess",
            "real_minus_null_median_mean_nonstationary_mode_probability": "gate3_nonstationary_mass_map_excess",
            "real_minus_null_median_posterior_expected_path_length_cm": "gate3_path_length_map_excess_cm",
            "real_minus_null_median_posterior_net_displacement_cm": "gate3_displacement_map_excess_cm",
        }
    )

    out = gate4.merge(gate2, on=KEYS, how="left", validate="one_to_one")
    out = out.merge(gate3, on=KEYS, how="left", validate="one_to_one")
    out = out.merge(map_permutation, on=KEYS, how="left", validate="one_to_one")
    out = out.rename(
        columns={
            "heldout_delta_event_median": "heldout_delta_imm_minus_fragmented",
            "median_train_imm_posterior_entropy": "train_imm_posterior_entropy",
        }
    )
    out["log1p_train_cell_count"] = np.log1p(
        pd.to_numeric(out["median_train_cell_count"], errors="coerce")
    )
    out["log1p_heldout_spikes"] = np.log1p(
        pd.to_numeric(out["median_test_spikes"], errors="coerce")
    )
    out["log1p_time_bins"] = np.log1p(pd.to_numeric(out["median_n_time"], errors="coerce"))
    out["gate2_available"] = out["gate2_real_order_advantage"].notna()
    out["gate3_available"] = out["gate3_mean_nonstationary_mode_probability"].notna()
    out["map_content_available"] = out["gate3_nonstationary_mass_map_excess"].notna()
    return out.sort_values(KEYS).reset_index(drop=True)


def _finite(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    keep = np.isfinite(out[columns].to_numpy(dtype=float)).all(axis=1)
    return out.loc[keep].copy()


def raw_spearman(frame: pd.DataFrame, x: str, y: str) -> tuple[float, float, int, int]:
    sub = _finite(frame, [x, y])
    if len(sub) < 3 or sub[x].nunique() < 2 or sub[y].nunique() < 2:
        return np.nan, np.nan, len(sub), sub["rat"].nunique()
    result = spearmanr(sub[x], sub[y])
    return float(result.statistic), float(result.pvalue), len(sub), sub["rat"].nunique()


def _rank(values: pd.Series) -> np.ndarray:
    return values.rank(method="average").to_numpy(dtype=float)


def partial_spearman(
    frame: pd.DataFrame,
    x: str,
    y: str,
    controls: list[str],
    *,
    rat_fixed_effects: bool = True,
) -> tuple[float, float, int, int]:
    """Return rank-partial correlation after OLS residualization."""
    columns = [x, y, *controls]
    sub = _finite(frame, columns)
    if len(sub) < max(8, len(controls) + 4):
        return np.nan, np.nan, len(sub), sub["rat"].nunique()
    ranked_x = _rank(sub[x])
    ranked_y = _rank(sub[y])
    design_parts = [np.ones((len(sub), 1), dtype=float)]
    if controls:
        ranked_controls = np.column_stack([_rank(sub[column]) for column in controls])
        design_parts.append(ranked_controls)
    if rat_fixed_effects and sub["rat"].nunique() > 1:
        rat_dummies = pd.get_dummies(sub["rat"].astype(str), drop_first=True, dtype=float)
        if not rat_dummies.empty:
            design_parts.append(rat_dummies.to_numpy(dtype=float))
    design = np.column_stack(design_parts)
    resid_x = ranked_x - design @ np.linalg.lstsq(design, ranked_x, rcond=None)[0]
    resid_y = ranked_y - design @ np.linalg.lstsq(design, ranked_y, rcond=None)[0]
    if np.std(resid_x) <= 0 or np.std(resid_y) <= 0:
        return np.nan, np.nan, len(sub), sub["rat"].nunique()
    result = pearsonr(resid_x, resid_y)
    return float(result.statistic), float(result.pvalue), len(sub), sub["rat"].nunique()


def rat_cluster_bootstrap_partial(
    frame: pd.DataFrame,
    x: str,
    y: str,
    controls: list[str],
    *,
    replicates: int,
    seed: int,
) -> tuple[float, float, float, int]:
    sub = _finite(frame, [x, y, *controls])
    rats = sorted(sub["rat"].astype(str).unique())
    if len(rats) < 2 or replicates <= 0:
        return np.nan, np.nan, np.nan, 0
    groups = {rat: sub[sub["rat"].astype(str).eq(rat)].copy() for rat in rats}
    rng = np.random.default_rng(seed)
    estimates: list[float] = []
    for _ in range(int(replicates)):
        sampled = rng.choice(rats, size=len(rats), replace=True)
        pieces = []
        for sample_index, rat in enumerate(sampled):
            piece = groups[str(rat)].copy()
            piece["rat"] = f"bootstrap_cluster_{sample_index}"
            pieces.append(piece)
        estimate, _, _, _ = partial_spearman(
            pd.concat(pieces, ignore_index=True),
            x,
            y,
            controls,
            rat_fixed_effects=True,
        )
        if np.isfinite(estimate):
            estimates.append(float(estimate))
    if not estimates:
        return np.nan, np.nan, np.nan, 0
    values = np.asarray(estimates, dtype=float)
    return (
        float(np.quantile(values, 0.025)),
        float(np.quantile(values, 0.975)),
        float(np.mean(values > 0)),
        len(values),
    )


def correlation_tables(
    events: pd.DataFrame,
    *,
    bootstrap_replicates: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw_rows: list[dict[str, object]] = []
    partial_rows: list[dict[str, object]] = []
    by_rat_rows: list[dict[str, object]] = []
    control_sets = {
        "decodability_core": CORE_CONTROLS,
        "decodability_plus_duration": EXTENDED_CONTROLS,
    }
    for pair_index, spec in enumerate(PAIR_SPECS):
        rho, p_value, n_events, n_rats = raw_spearman(events, spec["x"], spec["y"])
        raw_rows.append(
            {
                **spec,
                "events": n_events,
                "rats": n_rats,
                "spearman_rho": rho,
                "p_value_descriptive": p_value,
            }
        )
        for rat, group in events.groupby("rat", sort=True):
            rat_rho, rat_p, rat_events, _ = raw_spearman(group, spec["x"], spec["y"])
            by_rat_rows.append(
                {
                    "analysis_id": spec["analysis_id"],
                    "rat": rat,
                    "events": rat_events,
                    "spearman_rho": rat_rho,
                    "p_value_descriptive": rat_p,
                }
            )
        for control_index, (control_id, controls) in enumerate(control_sets.items()):
            estimate, p_partial, partial_events, partial_rats = partial_spearman(
                events,
                spec["x"],
                spec["y"],
                controls,
                rat_fixed_effects=True,
            )
            ci_low, ci_high, positive_fraction, finite_replicates = rat_cluster_bootstrap_partial(
                events,
                spec["x"],
                spec["y"],
                controls,
                replicates=bootstrap_replicates,
                seed=seed + pair_index * 17 + control_index,
            )
            if np.isfinite(ci_low) and ci_low > 0:
                classification = "positive_ci_excludes_zero_exploratory"
            elif np.isfinite(estimate) and estimate > 0:
                classification = "positive_estimate_ci_includes_zero_exploratory"
            elif np.isfinite(estimate):
                classification = "nonpositive_exploratory"
            else:
                classification = "not_estimable"
            partial_rows.append(
                {
                    "analysis_id": spec["analysis_id"],
                    "selection_scope": spec["selection_scope"],
                    "x_metric": spec["x"],
                    "y_metric": spec["y"],
                    "control_set": control_id,
                    "controls": ";".join(controls),
                    "rat_fixed_effects": True,
                    "events": partial_events,
                    "rats": partial_rats,
                    "partial_spearman_rho": estimate,
                    "p_value_descriptive": p_partial,
                    "rat_cluster_bootstrap_ci_low": ci_low,
                    "rat_cluster_bootstrap_ci_high": ci_high,
                    "rat_cluster_bootstrap_positive_fraction": positive_fraction,
                    "finite_bootstrap_replicates": finite_replicates,
                    "classification": classification,
                    "post_hoc_status": "exploratory_not_confirmatory",
                    "note": spec["note"],
                }
            )
    return pd.DataFrame(raw_rows), pd.DataFrame(partial_rows), pd.DataFrame(by_rat_rows)


def gate_summary(events: pd.DataFrame, raw: pd.DataFrame, partial: pd.DataFrame) -> pd.DataFrame:
    overlap = {
        "gate4_all_events_present": int(events["heldout_delta_imm_minus_fragmented"].notna().sum()),
        "gate2_pilot_overlap_present": int(events["gate2_available"].sum()),
        "gate3_clean_overlap_present": int(events["gate3_available"].sum()),
        "map_content_overlap_present": int(events["map_content_available"].sum()),
    }
    criteria = {
        "gate4_all_events_present": (160, "160 primary Gate 4 events"),
        "gate2_pilot_overlap_present": (20, "20 frozen Gate 2 pilot events"),
        "gate3_clean_overlap_present": (108, "108 frozen clean-IMM Gate 3 events"),
        "map_content_overlap_present": (108, "108 events with map-specific posterior content"),
    }
    rows = []
    for gate, observed in overlap.items():
        expected, criterion = criteria[gate]
        rows.append(
            {
                "gate_type": "technical",
                "gate": gate,
                "passed": observed == expected,
                "observed": observed,
                "criterion": criterion,
            }
        )
    rows.extend(
        [
            {
                "gate_type": "technical",
                "gate": "all_pair_specs_reported",
                "passed": len(raw) == len(PAIR_SPECS),
                "observed": f"{len(raw)}/{len(PAIR_SPECS)}",
                "criterion": "all prelisted exploratory associations reported",
            },
            {
                "gate_type": "technical",
                "gate": "all_partial_control_sets_reported",
                "passed": len(partial) == 2 * len(PAIR_SPECS),
                "observed": f"{len(partial)}/{2 * len(PAIR_SPECS)}",
                "criterion": "core and duration-adjusted partial associations reported",
            },
        ]
    )
    technical_pass = all(bool(row["passed"]) for row in rows)
    rows.append(
        {
            "gate_type": "summary",
            "gate": "overall_technical",
            "passed": technical_pass,
            "observed": "complete" if technical_pass else "incomplete",
            "criterion": "all joins and exploratory outputs complete",
        }
    )
    rows.append(
        {
            "gate_type": "interpretation",
            "gate": "biological_confirmation",
            "passed": False,
            "observed": "not_applicable_post_hoc",
            "criterion": "this post-hoc audit is descriptive and cannot create a new confirmatory gate",
        }
    )
    return pd.DataFrame(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_value(args: list[str]) -> str:
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def write_report(
    output_dir: Path,
    raw: pd.DataFrame,
    partial: pd.DataFrame,
    gates: pd.DataFrame,
) -> None:
    core = partial[partial["control_set"].eq("decodability_core")].copy()
    lines = [
        "# Pfeiffer/Foster IMM Gate Convergence and Decodability Audit",
        "",
        "This is a non-rescoring, post-hoc exploratory audit. Event counts differ because",
        "Gate 4 covers 160 events, Gate 3 covers 108 frozen clean-IMM events, and the",
        "completed order-by-map factorial covers a frozen 20-event pilot.",
        "",
        "## Core decodability-adjusted associations",
        "",
        "| analysis | events | raw rho | partial rho | rat-bootstrap 95% CI | classification |",
        "| --- | ---: | ---: | ---: | --- | --- |",
    ]
    raw_index = raw.set_index("analysis_id")
    for row in core.itertuples(index=False):
        raw_rho = raw_index.loc[row.analysis_id, "spearman_rho"]
        lines.append(
            f"| {row.analysis_id} | {int(row.events)} | {raw_rho:.3f} | "
            f"{row.partial_spearman_rho:.3f} | "
            f"[{row.rat_cluster_bootstrap_ci_low:.3f}, {row.rat_cluster_bootstrap_ci_high:.3f}] | "
            f"{row.classification} |"
        )
    technical = gates.loc[gates["gate"].eq("overall_technical"), "passed"].iloc[0]
    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            f"Technical completion: `{bool(technical)}`.",
            "",
            "These associations diagnose whether the existing gates share a decodability axis.",
            "They do not replace the independently defined Gate 1b/2/3/4 results and should not",
            "be described as preregistered or confirmatory.",
        ]
    )
    (output_dir / "pfeiffer_imm_gate_convergence_report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate4-event-medians", type=Path, required=True)
    parser.add_argument("--gate2-factorial-decisions", type=Path, required=True)
    parser.add_argument("--gate3-posterior-content", type=Path, required=True)
    parser.add_argument("--map-permutation-decisions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inputs = {
        "gate4_event_medians": args.gate4_event_medians,
        "gate2_factorial_decisions": args.gate2_factorial_decisions,
        "gate3_posterior_content": args.gate3_posterior_content,
        "map_permutation_decisions": args.map_permutation_decisions,
    }
    events = build_event_table(
        pd.read_csv(args.gate4_event_medians),
        pd.read_csv(args.gate2_factorial_decisions),
        pd.read_csv(args.gate3_posterior_content),
        pd.read_csv(args.map_permutation_decisions),
    )
    raw, partial, by_rat = correlation_tables(
        events,
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    gates = gate_summary(events, raw, partial)
    outputs = {
        "pfeiffer_imm_gate_convergence_event_table.csv": events,
        "pfeiffer_imm_gate_convergence_correlations.csv": raw,
        "pfeiffer_imm_gate_convergence_partial_correlations.csv": partial,
        "pfeiffer_imm_gate_convergence_by_rat.csv": by_rat,
        "pfeiffer_imm_gate_convergence_gate_summary.csv": gates,
    }
    for name, frame in outputs.items():
        frame.to_csv(args.output_dir / name, index=False)
    manifest = {
        "analysis": "pfeiffer_imm_gate_convergence_and_decodability_audit",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "code_commit": _git_value(["git", "rev-parse", "HEAD"]),
        "git_branch": _git_value(["git", "branch", "--show-current"]),
        "command_line": " ".join(sys.argv),
        "post_hoc_status": "exploratory_not_confirmatory",
        "event_aggregation": "Gate 4 repeated splits collapsed to one median per event",
        "partial_correlation": "rank residualization with rat fixed effects",
        "bootstrap": "rat-cluster bootstrap",
        "bootstrap_replicates": int(args.bootstrap_replicates),
        "seed": int(args.seed),
        "inputs": {name: str(path.resolve()) for name, path in inputs.items()},
        "input_sha256": {name: _sha256(path) for name, path in inputs.items()},
    }
    (args.output_dir / "pfeiffer_imm_gate_convergence_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_report(args.output_dir, raw, partial, gates)


if __name__ == "__main__":
    main()
