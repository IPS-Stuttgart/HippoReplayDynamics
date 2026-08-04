from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from scripts import test_pfeiffer_ripple_cross_cell_coordination as analysis


def _event_row(session: str, rat: str, event_index: int) -> dict[str, object]:
    return {
        "session": session,
        "rat": rat,
        "event_index": event_index,
        "completed_splits": 20,
        analysis.PRIMARY_Y: 2.0 + event_index,
        analysis.MAP_SPECIFIC_Y: 0.5 + event_index,
        analysis.CONTENT_Y: 0.2 + event_index / 10,
        analysis.ABSOLUTE_CONTENT_Y: 0.7,
        "log1p_median_train_cell_count": 4.0,
        "log1p_median_test_spikes": 3.0,
        "median_real_imm_train_posterior_entropy": 2.0,
        "log1p_median_n_time": 3.5,
        "log1p_median_train_spikes": 4.0,
    }


def test_native_ripple_join_checks_event_identity() -> None:
    events = pd.DataFrame(
        [
            _event_row("Rat1/Open1", "Rat1", 0),
            _event_row("Rat1/Open1", "Rat1", 1),
        ]
    )
    splits = pd.DataFrame(
        [
            {
                "session": "Rat1/Open1",
                "event_index": event_index,
                "status": "success",
                "event_start_s": 10.0 + event_index,
                "event_end_s": 10.2 + event_index,
            }
            for event_index in (0, 1)
            for _ in range(2)
        ]
    )
    ripples = [
        SimpleNamespace(
            start=10.0,
            end=10.2,
            peak=10.1,
            raw_power=50.0,
            z_power_session=4.0,
            z_power_epoch=3.5,
        ),
        SimpleNamespace(
            start=11.0,
            end=11.2,
            peak=11.1,
            raw_power=60.0,
            z_power_session=5.0,
            z_power_epoch=4.5,
        ),
    ]
    session = SimpleNamespace(
        ripple_count=2,
        ripple=lambda index: ripples[index],
    )

    joined = analysis.join_native_ripple_metrics(
        events,
        splits,
        dataset_root=Path("unused"),
        session_loader=lambda path: session,
    )

    assert joined[analysis.PRIMARY_X].tolist() == [3.5, 4.5]
    assert joined["ripple_power_raw"].tolist() == [50.0, 60.0]
    assert joined["ripple_event_start_abs_error_s"].eq(0.0).all()
    assert joined["ripple_event_end_abs_error_s"].eq(0.0).all()
    assert joined["native_detected_swr"].all()


def _synthetic_events(seed: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for rat_index in range(4):
        rat = f"Rat{rat_index + 1}"
        for session_index in range(2):
            session = f"{rat}/Open{session_index + 1}"
            for event_index in range(20):
                ripple = rng.normal()
                content = 0.25 * ripple + rng.normal(scale=0.8)
                entropy = rng.normal()
                test_spikes = rng.normal()
                outcome = (
                    1.6 * ripple
                    + 0.8 * content
                    - 0.2 * entropy
                    + 0.1 * test_spikes
                    + rng.normal(scale=0.5)
                )
                rows.append(
                    {
                        "session": session,
                        "rat": rat,
                        "event_index": event_index,
                        analysis.PRIMARY_X: ripple,
                        analysis.PRIMARY_Y: outcome,
                        analysis.MAP_SPECIFIC_Y: outcome + rng.normal(scale=0.3),
                        analysis.CONTENT_Y: content,
                        analysis.ABSOLUTE_CONTENT_Y: 0.7 + 0.1 * content,
                        "log1p_median_train_cell_count": rng.normal(),
                        "log1p_median_test_spikes": test_spikes,
                        "median_real_imm_train_posterior_entropy": entropy,
                        "log1p_median_n_time": rng.normal(),
                        "log1p_median_train_spikes": rng.normal(),
                    }
                )
    return pd.DataFrame(rows)


def test_partial_association_recovers_coordination_beyond_content() -> None:
    events = _synthetic_events()

    rho, p_value, n_events, n_rats = analysis.partial_spearman(
        events,
        analysis.PRIMARY_X,
        analysis.PRIMARY_Y,
        analysis.PRIMARY_CONTROLS,
    )

    assert n_events == 160
    assert n_rats == 4
    assert rho > 0.7
    assert p_value < 1e-10


def test_within_session_permutation_detects_strong_positive_effect() -> None:
    events = _synthetic_events()

    null, p_value = analysis.within_session_permutation(
        events,
        analysis_id="primary",
        x=analysis.PRIMARY_X,
        y=analysis.PRIMARY_Y,
        controls=analysis.PRIMARY_CONTROLS,
        replicates=99,
        seed=8,
    )

    assert len(null) == 99
    assert p_value <= 0.02


def _gate_events() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for rat_index in range(4):
        rat = f"Rat{rat_index + 1}"
        for session_index in range(2):
            session = f"{rat}/Open{session_index + 1}"
            for event_index in range(20):
                row = _event_row(session, rat, event_index)
                row.update(
                    {
                        "ripple_power_raw": 50.0,
                        "ripple_power_z_session": 4.0,
                        analysis.PRIMARY_X: 3.5,
                        "ripple_event_start_abs_error_s": 0.0,
                        "ripple_event_end_abs_error_s": 0.0,
                    }
                )
                rows.append(row)
    return pd.DataFrame(rows)


def _association_row(analysis_id: str, rho: float = 0.3) -> dict[str, object]:
    return {
        "analysis_id": analysis_id,
        "adjusted_partial_spearman_rho": rho,
        "extended_adjusted_partial_spearman_rho": rho - 0.02,
        "rat_cluster_bootstrap_ci_low": 0.1,
        "rat_cluster_bootstrap_ci_high": 0.5,
        "within_session_permutation_p_one_sided": 0.01,
    }


def test_gate_classifies_selective_map_specific_coordination() -> None:
    events = _gate_events()
    associations = pd.DataFrame(
        [
            _association_row(
                "primary_ripple_strength_predicts_coordination_beyond_content"
            ),
            _association_row(
                "sensitivity_ripple_strength_predicts_map_specific_coordination"
            ),
            _association_row(
                "secondary_ripple_strength_predicts_map_specific_content",
                rho=0.05,
            ),
        ]
    )
    by_rat = pd.DataFrame(
        {
            "rat": [f"Rat{index}" for index in range(1, 5)],
            "adjusted_partial_spearman_rho": [0.1, 0.2, 0.3, 0.4],
        }
    )
    leave_one_out = pd.DataFrame(
        {
            "omitted_rat": [f"Rat{index}" for index in range(1, 5)],
            "adjusted_partial_spearman_rho": [0.2, 0.2, 0.2, 0.2],
        }
    )
    dissociation = pd.DataFrame(
        [
            {
                "estimate": 0.25,
                "rat_cluster_bootstrap_ci_low": 0.05,
                "rat_cluster_bootstrap_ci_high": 0.4,
            }
        ]
    )

    gates, decision = analysis.build_gate_summary(
        events,
        associations,
        by_rat,
        leave_one_out,
        dissociation,
        expected_events=160,
        expected_splits=20,
        event_time_tolerance_s=1e-9,
    )

    assert decision == "ripple_selectively_indexes_map_specific_cross_cell_coordination"
    indexed = gates.set_index("gate")
    assert bool(indexed.loc["overall_technical", "passed"])
    assert bool(indexed.loc["ripple_coordination_hypothesis_supported", "passed"])
    assert bool(indexed.loc["map_specific_coordination_sensitivity_supported", "passed"])


def test_primary_failure_is_not_rescued_by_content_association() -> None:
    events = _gate_events()
    associations = pd.DataFrame(
        [
            _association_row(
                "primary_ripple_strength_predicts_coordination_beyond_content",
                rho=-0.1,
            ),
            _association_row(
                "sensitivity_ripple_strength_predicts_map_specific_coordination",
                rho=-0.1,
            ),
            _association_row(
                "secondary_ripple_strength_predicts_map_specific_content",
                rho=0.3,
            ),
        ]
    )
    associations.loc[
        associations["analysis_id"].str.startswith("primary_"),
        "rat_cluster_bootstrap_ci_low",
    ] = -0.3
    associations.loc[
        associations["analysis_id"].str.startswith("primary_"),
        "within_session_permutation_p_one_sided",
    ] = 0.9
    by_rat = pd.DataFrame(
        {"rat": [f"Rat{index}" for index in range(1, 5)], "adjusted_partial_spearman_rho": [-0.1] * 4}
    )
    leave_one_out = pd.DataFrame(
        {"omitted_rat": [f"Rat{index}" for index in range(1, 5)], "adjusted_partial_spearman_rho": [-0.1] * 4}
    )
    dissociation = pd.DataFrame(
        [{"estimate": -0.4, "rat_cluster_bootstrap_ci_low": -0.6, "rat_cluster_bootstrap_ci_high": -0.2}]
    )

    _, decision = analysis.build_gate_summary(
        events,
        associations,
        by_rat,
        leave_one_out,
        dissociation,
        expected_events=160,
        expected_splits=20,
        event_time_tolerance_s=1e-9,
    )

    assert decision == "ripple_strength_tracks_content_not_heldout_coordination"
