from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.test_replay_dynamics_behavior_hypotheses import (
    GATE_OUTPUT,
    PRIMARY_OUTPUT,
    build_gate_summary,
    run_analysis,
    run_primary_tests,
    summarize_matched_sensitivity,
)


def _synthetic_events(seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for rat_index in range(4):
        rat = f"Rat{rat_index + 1}"
        for event_index in range(24):
            predictor = float(rng.normal())
            rows.append(
                {
                    "session": f"{rat}/Open{1 + event_index % 2}",
                    "rat": rat,
                    "event_index": event_index,
                    "delta_momentum_minus_imm": predictor,
                    "composition_index": -1.8 * predictor + rng.normal(scale=0.15),
                    "composition_evaluable": True,
                    "future_commitment_index": 1.8 * predictor + rng.normal(scale=0.15),
                    "event_duration_ms": 100.0 + rng.normal(scale=10.0),
                    "n_spikes": int(rng.integers(20, 60)),
                    "active_cell_count": int(rng.integers(8, 20)),
                    "posterior_entropy": float(rng.uniform(0.5, 2.0)),
                    "trajectory_minus_stationary_log_evidence": float(rng.normal(20.0, 3.0)),
                    "posterior_path_length_cm": float(rng.uniform(20.0, 100.0)),
                    "current_animal_x_cm": float(rng.uniform(0.0, 100.0)),
                    "current_animal_y_cm": float(rng.uniform(0.0, 100.0)),
                    "next_well_x_cm": float(rng.uniform(0.0, 100.0)),
                    "next_well_y_cm": float(rng.uniform(0.0, 100.0)),
                    "route_frequency": int(rng.integers(1, 20)),
                    "time_to_departure_s": float(rng.uniform(0.5, 10.0)),
                    "elapsed_time_since_reward_s": float(rng.uniform(0.0, 5.0)),
                    "run_decoder_error_cm": 12.0 + rat_index,
                    "current_well": int(event_index % 3),
                    "next_well": int((event_index + 1) % 3),
                    "raw_momentum_win": predictor > 0.8,
                    "clean_imm": predictor < -0.8,
                }
            )
    return pd.DataFrame(rows)


def test_primary_tests_recover_predeclared_directions() -> None:
    primary, by_rat, leave_one_out = run_primary_tests(
        _synthetic_events(),
        bootstrap_replicates=200,
        seed=13,
    )
    indexed = primary.set_index("test")

    composition = indexed.loc["composition_decreases_with_momentum_axis"]
    commitment = indexed.loc["future_commitment_increases_with_momentum_axis"]
    assert composition["adjusted_standardized_coefficient"] < -0.8
    assert composition["rat_cluster_bootstrap_ci_high"] < 0.0
    assert commitment["adjusted_standardized_coefficient"] > 0.8
    assert commitment["rat_cluster_bootstrap_ci_low"] > 0.0
    assert set(primary["status"]) == {"supported"}
    assert by_rat["adjusted_standardized_coefficient"].notna().all()
    assert leave_one_out["expected_direction_retained"].all()


def test_gate_does_not_pass_an_underpowered_composition_cohort() -> None:
    primary, by_rat, leave_one_out = run_primary_tests(
        _synthetic_events().assign(
            composition_index=lambda frame: frame["composition_index"].where(frame.index < 12)
        ),
        bootstrap_replicates=20,
        seed=3,
    )

    gates = build_gate_summary(
        primary,
        by_rat,
        leave_one_out,
        events=_synthetic_events(),
    ).set_index("gate")

    assert not bool(
        gates.loc["composition_decreases_with_momentum_axis_cohort_size", "passed"]
    )
    assert not bool(gates.loc["overall_strong_primary_support", "passed"])


def test_adjusted_coefficient_rejects_saturated_small_group() -> None:
    events = _synthetic_events().head(8)

    primary, by_rat, _ = run_primary_tests(
        events,
        bootstrap_replicates=5,
        seed=4,
    )

    assert primary["adjusted_standardized_coefficient"].isna().all()
    assert by_rat["adjusted_standardized_coefficient"].isna().all()


def test_run_analysis_writes_primary_and_nonvacuous_gate_outputs(tmp_path) -> None:
    event_path = tmp_path / "event_metrics.csv"
    _synthetic_events().to_csv(event_path, index=False)

    outputs = run_analysis(
        event_metrics_csv=event_path,
        output_dir=tmp_path / "out",
        bootstrap_replicates=50,
        seed=7,
    )

    assert outputs[PRIMARY_OUTPUT].is_file()
    gates = pd.read_csv(outputs[GATE_OUTPUT]).set_index("gate")
    assert bool(gates.loc["overall_strong_primary_support", "passed"])


def test_strong_gate_fails_without_run_decoder_error_control() -> None:
    events = _synthetic_events()
    events["run_decoder_error_cm"] = np.nan
    primary, by_rat, leave_one_out = run_primary_tests(
        events,
        bootstrap_replicates=20,
        seed=2,
    )

    gates = build_gate_summary(
        primary,
        by_rat,
        leave_one_out,
        events=events,
    ).set_index("gate")

    assert not bool(gates.loc["run_decoder_error_control_available", "passed"])
    assert not bool(gates.loc["overall_strong_primary_support", "passed"])


def test_matched_summary_keeps_categorical_comparison_secondary() -> None:
    matched = pd.DataFrame(
        {
            "composition_difference_treated_minus_control": [-2.0, -1.0],
            "commitment_difference_treated_minus_control": [3.0, 5.0],
        }
    )

    summary = summarize_matched_sensitivity(matched).set_index("test")

    assert (
        summary.loc["composition_momentum_minus_clean_imm", "median_treated_minus_control"]
        == -1.5
    )
    assert (
        summary.loc["commitment_momentum_minus_clean_imm", "median_treated_minus_control"]
        == 4.0
    )
    assert set(summary["role"]) == {"secondary_descriptive_only"}
