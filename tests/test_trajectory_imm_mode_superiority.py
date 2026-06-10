from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path("scripts").resolve()))
from trajectory_imm_mode_superiority import (
    DEFAULT_DIFFUSION_MODEL,
    DEFAULT_FIRST_ORDER_IMM_MODEL,
    DEFAULT_FRAGMENTED_MODEL,
    DEFAULT_MOMENTUM_MODEL,
    DEFAULT_STATIONARY_MODEL,
    DEFAULT_TRAJECTORY_IMM_MODEL,
    rat_bootstrap_trajectory_imm_superiority,
    trajectory_imm_event_pairs,
    trajectory_imm_mode_readiness,
    trajectory_imm_promotion_gate_summary,
    trajectory_imm_superiority_summary,
    write_trajectory_imm_superiority_outputs,
)


def test_trajectory_imm_superiority_gate_passes_for_strong_toy_rows(tmp_path: Path):
    scores = pd.DataFrame(
        [
            row("Rat1/Open1", 0, DEFAULT_STATIONARY_MODEL, 0.0),
            row("Rat1/Open1", 0, DEFAULT_DIFFUSION_MODEL, 5.0),
            row("Rat1/Open1", 0, DEFAULT_FRAGMENTED_MODEL, 7.0),
            row("Rat1/Open1", 0, DEFAULT_MOMENTUM_MODEL, 11.0),
            row("Rat1/Open1", 0, DEFAULT_FIRST_ORDER_IMM_MODEL, 20.0),
            row("Rat1/Open1", 0, DEFAULT_TRAJECTORY_IMM_MODEL, 32.0, mode_mass_momentum=0.7, mode_entropy=0.4),
            row("Rat1/Open1", 1, DEFAULT_STATIONARY_MODEL, 0.0),
            row("Rat1/Open1", 1, DEFAULT_DIFFUSION_MODEL, 5.0),
            row("Rat1/Open1", 1, DEFAULT_FRAGMENTED_MODEL, 7.0),
            row("Rat1/Open1", 1, DEFAULT_MOMENTUM_MODEL, 10.0),
            row("Rat1/Open1", 1, DEFAULT_FIRST_ORDER_IMM_MODEL, 18.0),
            row("Rat1/Open1", 1, DEFAULT_TRAJECTORY_IMM_MODEL, 30.0, mode_mass_momentum=0.6, mode_entropy=0.5),
            row("Rat2/Open1", 2, DEFAULT_STATIONARY_MODEL, 0.0),
            row("Rat2/Open1", 2, DEFAULT_DIFFUSION_MODEL, 5.0),
            row("Rat2/Open1", 2, DEFAULT_FRAGMENTED_MODEL, 7.0),
            row("Rat2/Open1", 2, DEFAULT_MOMENTUM_MODEL, 8.0),
            row("Rat2/Open1", 2, DEFAULT_FIRST_ORDER_IMM_MODEL, 14.0),
            row("Rat2/Open1", 2, DEFAULT_TRAJECTORY_IMM_MODEL, 24.0, mode_mass_momentum=0.5, mode_entropy=0.6),
            row("Rat2/Open1", 3, DEFAULT_STATIONARY_MODEL, 0.0),
            row("Rat2/Open1", 3, DEFAULT_DIFFUSION_MODEL, 5.0),
            row("Rat2/Open1", 3, DEFAULT_FRAGMENTED_MODEL, 7.0),
            row("Rat2/Open1", 3, DEFAULT_MOMENTUM_MODEL, 9.0),
            row("Rat2/Open1", 3, DEFAULT_FIRST_ORDER_IMM_MODEL, 15.0),
            row("Rat2/Open1", 3, DEFAULT_TRAJECTORY_IMM_MODEL, 23.0, mode_mass_momentum=0.4, mode_entropy=0.7),
        ]
    )

    pairs = trajectory_imm_event_pairs(scores, margin_threshold=5.5)
    summary = trajectory_imm_superiority_summary(pairs)
    mode = trajectory_imm_mode_readiness(scores)
    gate = trajectory_imm_promotion_gate_summary(pairs, mode, n_bootstrap=100, random_seed=1)

    assert len(pairs) == 4
    assert pairs["required_core_complete"].all()
    assert int(summary.iloc[0]["trajectory_imm_wins_vs_first_order_imm"]) == 4
    assert float(summary.iloc[0]["median_delta_vs_first_order_imm"]) > 0.0
    assert bool(mode.iloc[0]["interpretability_ready"])
    assert gate.loc[gate["gate"] == "overall", "passed"].iloc[0]

    write_trajectory_imm_superiority_outputs(scores, tmp_path, n_bootstrap=100, random_seed=1)
    for name in (
        "trajectory_imm_superiority_event_pairs.csv",
        "trajectory_imm_superiority_summary.csv",
        "rat_trajectory_imm_superiority_summary.csv",
        "leave_one_rat_out_trajectory_imm_superiority_summary.csv",
        "rat_bootstrap_trajectory_imm_superiority.csv",
        "trajectory_imm_mode_readiness.csv",
        "trajectory_imm_promotion_gate_summary.csv",
    ):
        assert (tmp_path / name).is_file()


def test_trajectory_imm_gate_blocks_model_that_does_not_beat_first_order_imm():
    scores = pd.DataFrame(
        [
            row("Rat1/Open1", 0, DEFAULT_STATIONARY_MODEL, 0.0),
            row("Rat1/Open1", 0, DEFAULT_DIFFUSION_MODEL, 3.0),
            row("Rat1/Open1", 0, DEFAULT_FRAGMENTED_MODEL, 4.0),
            row("Rat1/Open1", 0, DEFAULT_MOMENTUM_MODEL, 5.0),
            row("Rat1/Open1", 0, DEFAULT_FIRST_ORDER_IMM_MODEL, 20.0),
            row("Rat1/Open1", 0, DEFAULT_TRAJECTORY_IMM_MODEL, 19.5, mode_mass_momentum=0.7),
            row("Rat2/Open1", 1, DEFAULT_STATIONARY_MODEL, 0.0),
            row("Rat2/Open1", 1, DEFAULT_DIFFUSION_MODEL, 3.0),
            row("Rat2/Open1", 1, DEFAULT_FRAGMENTED_MODEL, 4.0),
            row("Rat2/Open1", 1, DEFAULT_MOMENTUM_MODEL, 5.0),
            row("Rat2/Open1", 1, DEFAULT_FIRST_ORDER_IMM_MODEL, 21.0),
            row("Rat2/Open1", 1, DEFAULT_TRAJECTORY_IMM_MODEL, 20.5, mode_mass_momentum=0.6),
        ]
    )

    pairs = trajectory_imm_event_pairs(scores)
    mode = trajectory_imm_mode_readiness(scores)
    gate = trajectory_imm_promotion_gate_summary(pairs, mode, n_bootstrap=100, random_seed=1)

    assert not gate.loc[gate["gate"] == "trajectory_imm_raw_win_majority_vs_first_order_imm", "passed"].iloc[0]
    assert not gate.loc[gate["gate"] == "trajectory_imm_median_delta_vs_first_order_imm_positive", "passed"].iloc[0]
    assert not gate.loc[gate["gate"] == "overall", "passed"].iloc[0]


def test_mode_readiness_requires_mode_diagnostic_columns():
    scores = pd.DataFrame(
        [
            row("Rat1/Open1", 0, DEFAULT_STATIONARY_MODEL, 0.0),
            row("Rat1/Open1", 0, DEFAULT_FIRST_ORDER_IMM_MODEL, 2.0),
            row("Rat1/Open1", 0, DEFAULT_TRAJECTORY_IMM_MODEL, 3.0),
        ]
    )

    readiness = trajectory_imm_mode_readiness(scores)

    assert int(readiness.iloc[0]["mode_diagnostic_columns"]) == 0
    assert not bool(readiness.iloc[0]["mode_diagnostics_present"])
    assert not bool(readiness.iloc[0]["interpretability_ready"])


def test_promotion_gate_parses_string_false_mode_readiness_flags():
    mode = pd.DataFrame(
        [
            {
                "mode_diagnostics_present": "False",
                "interpretability_ready": "False",
                "mode_diagnostic_columns": 0,
                "mode_diagnostics_complete_fraction": 0.0,
            }
        ]
    )

    gate = trajectory_imm_promotion_gate_summary(
        pd.DataFrame(),
        mode,
        n_bootstrap=10,
        random_seed=1,
    ).set_index("gate")

    assert not bool(gate.loc["trajectory_imm_mode_diagnostics_present", "passed"])
    assert not bool(
        gate.loc["trajectory_imm_mode_diagnostics_interpretability_ready", "passed"]
    )


def test_trajectory_imm_summary_parses_csv_bool_flags():
    pairs = pd.DataFrame(
        [
            {
                "rat": "Rat1",
                "session": "Rat1/Open1",
                "event_index": 0,
                "required_core_complete": "True",
                "delta_trajectory_imm_minus_first_order_imm": 6.0,
                "delta_trajectory_imm_minus_momentum": 7.0,
                "trajectory_imm_confident_vs_first_order_imm": "True",
                "first_order_imm_confident_vs_trajectory_imm": "False",
                "trajectory_imm_ambiguous_vs_first_order_imm": "False",
                "trajectory_imm_confident_vs_momentum": "1.0",
                "momentum_confident_vs_trajectory_imm": "0.0",
                "trajectory_imm_ambiguous_vs_momentum": "False",
                "trajectory_imm_raw_best_exact_core": "True",
                "trajectory_imm_confident_exact_core_claim": "True",
                "first_order_imm_raw_best_exact_core": "False",
                "momentum_raw_best_exact_core": "False",
                "trajectory_imm_rank_in_required_core": 1,
            },
            {
                "rat": "Rat1",
                "session": "Rat1/Open1",
                "event_index": 1,
                "required_core_complete": "1.0",
                "delta_trajectory_imm_minus_first_order_imm": -6.0,
                "delta_trajectory_imm_minus_momentum": 1.0,
                "trajectory_imm_confident_vs_first_order_imm": "False",
                "first_order_imm_confident_vs_trajectory_imm": "True",
                "trajectory_imm_ambiguous_vs_first_order_imm": "False",
                "trajectory_imm_confident_vs_momentum": "False",
                "momentum_confident_vs_trajectory_imm": "False",
                "trajectory_imm_ambiguous_vs_momentum": "True",
                "trajectory_imm_raw_best_exact_core": "False",
                "trajectory_imm_confident_exact_core_claim": "False",
                "first_order_imm_raw_best_exact_core": "True",
                "momentum_raw_best_exact_core": "False",
                "trajectory_imm_rank_in_required_core": 2,
            },
        ]
    )

    summary = trajectory_imm_superiority_summary(pairs).iloc[0]

    assert summary["complete_core_events"] == 2
    assert summary["trajectory_imm_confident_vs_first_order_imm"] == 1
    assert summary["first_order_imm_confident_vs_trajectory_imm"] == 1
    assert summary["trajectory_imm_confident_vs_momentum"] == 1
    assert summary["momentum_confident_vs_trajectory_imm"] == 0
    assert summary["ambiguous_vs_momentum"] == 1
    assert summary["trajectory_imm_raw_best_exact_core"] == 1
    assert summary["trajectory_imm_raw_best_exact_core_fraction"] == 0.5
    assert summary["trajectory_imm_confident_exact_core_claims"] == 1
    assert summary["first_order_imm_raw_best_exact_core"] == 1


def test_noncomparable_trajectory_imm_rows_cannot_satisfy_promotion_or_readiness():
    scores = pd.DataFrame(
        [
            row("Rat1/Open1", 0, DEFAULT_STATIONARY_MODEL, 0.0),
            row("Rat1/Open1", 0, DEFAULT_DIFFUSION_MODEL, 3.0),
            row("Rat1/Open1", 0, DEFAULT_FRAGMENTED_MODEL, 4.0),
            row("Rat1/Open1", 0, DEFAULT_MOMENTUM_MODEL, 5.0),
            row("Rat1/Open1", 0, DEFAULT_FIRST_ORDER_IMM_MODEL, 10.0),
            row(
                "Rat1/Open1",
                0,
                DEFAULT_TRAJECTORY_IMM_MODEL,
                100.0,
                evidence_comparable="False",
                evidence_support="truncated_full_grid",
                mode_mass_momentum=0.7,
            ),
        ]
    )

    pairs = trajectory_imm_event_pairs(scores)
    summary = trajectory_imm_superiority_summary(pairs).iloc[0]
    readiness = trajectory_imm_mode_readiness(scores).iloc[0]

    assert not bool(pairs.iloc[0]["required_core_complete"])
    assert DEFAULT_TRAJECTORY_IMM_MODEL in pairs.iloc[0]["missing_required_core_models"]
    assert pd.isna(pairs.iloc[0]["trajectory_imm_log_evidence"])
    assert summary["complete_core_events"] == 0
    assert readiness["trajectory_imm_rows"] == 0
    assert not bool(readiness["interpretability_ready"])


def test_rat_bootstrap_reports_positive_intervals_for_strong_rows():
    scores = pd.DataFrame(
        [
            row("Rat1/Open1", 0, DEFAULT_STATIONARY_MODEL, 0.0),
            row("Rat1/Open1", 0, DEFAULT_DIFFUSION_MODEL, 1.0),
            row("Rat1/Open1", 0, DEFAULT_FRAGMENTED_MODEL, 2.0),
            row("Rat1/Open1", 0, DEFAULT_MOMENTUM_MODEL, 3.0),
            row("Rat1/Open1", 0, DEFAULT_FIRST_ORDER_IMM_MODEL, 5.0),
            row("Rat1/Open1", 0, DEFAULT_TRAJECTORY_IMM_MODEL, 15.0),
            row("Rat2/Open1", 1, DEFAULT_STATIONARY_MODEL, 0.0),
            row("Rat2/Open1", 1, DEFAULT_DIFFUSION_MODEL, 1.0),
            row("Rat2/Open1", 1, DEFAULT_FRAGMENTED_MODEL, 2.0),
            row("Rat2/Open1", 1, DEFAULT_MOMENTUM_MODEL, 3.0),
            row("Rat2/Open1", 1, DEFAULT_FIRST_ORDER_IMM_MODEL, 5.0),
            row("Rat2/Open1", 1, DEFAULT_TRAJECTORY_IMM_MODEL, 13.0),
        ]
    )
    pairs = trajectory_imm_event_pairs(scores)
    boot = rat_bootstrap_trajectory_imm_superiority(pairs, n_bootstrap=100, random_seed=1)

    assert float(boot.iloc[0]["mean_delta_vs_first_order_imm_ci95_low"]) > 0.0
    assert float(boot.iloc[0]["median_delta_vs_first_order_imm_ci95_low"]) > 0.0


def row(
    session: str,
    event_index: int,
    model: str,
    log_evidence: float,
    evidence_comparable: object = True,
    evidence_support: str = "exact_full_grid",
    **diagnostics: float,
) -> dict[str, object]:
    out: dict[str, object] = {
        "status": "success",
        "session": session,
        "event_index": event_index,
        "model": model,
        "requested_model": model,
        "model_family": "trajectory" if model != DEFAULT_STATIONARY_MODEL else "stationary",
        "log_evidence": log_evidence,
        "evidence_comparable": evidence_comparable,
        "evidence_support": evidence_support,
    }
    out.update(diagnostics)
    return out
