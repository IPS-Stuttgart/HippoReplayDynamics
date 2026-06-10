import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))

from build_off_swr_promotion_funnel import (  # noqa: E402
    build_funnel_summary,
    build_gate_summary,
    write_off_swr_promotion_funnel_outputs,
)


def test_off_swr_promotion_funnel_joins_discovery_and_exact_validation(tmp_path):
    discovery = tmp_path / "discovery"
    validation = tmp_path / "validation"
    output = tmp_path / "funnel"
    discovery.mkdir()
    validation.mkdir()

    candidate_table = pd.DataFrame(
        [
            _candidate("Rat1/Open1", 1, 0, margin=8.0, state="run", label="ordinary_movement_or_spiking_like", tier="weak"),
            _candidate("Rat1/Open1", 2, 0, margin=25.0, state="run", label="ordinary_movement_or_spiking_like", tier="moderate"),
            _candidate("Rat2/Open1", 3, 0, margin=55.0, state="immobile", label="ordinary_movement_or_spiking_like", tier="strong"),
            _candidate("Rat2/Open1", 4, 0, margin=120.0, state="immobile", label="interesting_off_swr_trajectory_candidate", tier="extreme"),
        ]
    )
    high_specificity = candidate_table[candidate_table["trajectory_family_margin"].ge(50.0)].copy()
    high_specificity["high_specificity_label"] = [
        "tier_distance_candidate_movement_spiking_or_low_information",
        "promotion_ready_high_specificity_candidate",
    ]
    high_specificity["passes_high_specificity_promotion_filter"] = [False, True]
    high_specificity["passes_strong_tier"] = True
    high_specificity["passes_1s_swr_exclusion"] = True
    high_specificity["passes_immobility_filter"] = True
    high_specificity["passes_specificity_label_filter"] = [False, True]
    validation_decisions = pd.DataFrame(
        [
            {
                **_key("Rat2/Open1", 4, 0),
                "required_models_complete": True,
                "trajectory_confident_claim": True,
                "nontrajectory_confident_claim": False,
                "best_trajectory_model": "sorted-spike-state-space-first-order-imm",
                "trajectory_minus_nontrajectory_log_evidence": 101.0,
            }
        ]
    )

    candidate_table.to_csv(discovery / "off_swr_candidate_table.csv", index=False)
    high_specificity.to_csv(discovery / "off_swr_high_specificity_candidate_table.csv", index=False)
    _tier_summary().to_csv(discovery / "off_swr_candidate_tier_threshold_summary.csv", index=False)
    _tier_group_summary(group="session").to_csv(discovery / "off_swr_candidate_tier_session_summary.csv", index=False)
    _tier_group_summary(group="rat").to_csv(discovery / "off_swr_candidate_tier_rat_summary.csv", index=False)
    pd.DataFrame(
        [
            {"stratum": "off_swr_immobile_windows", "windows": 3},
            {"stratum": "off_swr_running_windows", "windows": 7},
        ]
    ).to_csv(discovery / "off_swr_run_state_stratified_summary.csv", index=False)
    validation_decisions.to_csv(validation / "promoted_off_swr_candidate_exact_core_decisions.csv", index=False)

    outputs = write_off_swr_promotion_funnel_outputs(
        discovery_dir=discovery,
        validation_dir=validation,
        output=output,
    )

    funnel = outputs["off_swr_promotion_funnel_summary.csv"].set_index("stage")
    assert int(funnel.loc["screened_off_swr_windows", "windows"]) == 10
    assert int(funnel.loc["weak_trajectory_candidates", "windows"]) == 4
    assert int(funnel.loc["moderate_trajectory_candidates", "windows"]) == 3
    assert int(funnel.loc["strong_trajectory_candidates", "windows"]) == 2
    assert int(funnel.loc["extreme_trajectory_candidates", "windows"]) == 1
    assert int(funnel.loc["strong_candidates_after_1s_swr_exclusion", "windows"]) == 2
    assert int(funnel.loc["promotion_ready_candidates", "windows"]) == 1
    assert int(funnel.loc["exact_core_validated_candidates", "windows"]) == 1
    assert int(funnel.loc["exact_core_trajectory_confident_candidates", "windows"]) == 1
    assert float(funnel.loc["promotion_ready_candidates", "fraction_of_screened_off_swr_windows"]) == 0.1
    assert float(funnel.loc["exact_core_trajectory_confident_candidates", "median_exact_margin"]) == 101.0

    rejection = outputs["off_swr_promotion_funnel_rejection_summary.csv"].set_index("funnel_status")
    assert int(rejection.loc["high_specificity_filter_rejected", "candidate_windows"]) == 1
    assert int(rejection.loc["high_specificity_filter_rejected", "movement_or_low_information_windows"]) == 1
    assert int(rejection.loc["exact_validated_promotion_ready", "candidate_windows"]) == 1

    group = outputs["off_swr_promotion_funnel_group_summary.csv"]
    rat1 = group[group["group_type"].eq("rat") & group["rat"].eq("Rat1")].iloc[0]
    rat2 = group[group["group_type"].eq("rat") & group["rat"].eq("Rat2")].iloc[0]
    assert int(rat1["exact_trajectory_confident_candidates"]) == 0
    assert int(rat2["promotion_ready_candidates"]) == 1
    assert int(rat2["exact_trajectory_confident_candidates"]) == 1

    gates = outputs["off_swr_promotion_funnel_gate_summary.csv"].set_index("gate")
    assert bool(gates.loc["overall", "passed"])

    for filename in outputs:
        assert (output / filename).exists()


def test_off_swr_promotion_funnel_gate_checks_validation_keys():
    candidate_table = pd.DataFrame(
        [_candidate("Rat2/Open1", 4, 0, margin=120.0, state="immobile", label="interesting_off_swr_trajectory_candidate", tier="extreme")]
    )
    high_specificity = candidate_table.copy()
    high_specificity["passes_high_specificity_promotion_filter"] = True
    validation_decisions = pd.DataFrame(
        [
            {
                **_key("Rat2/Open1", 999, 0),
                "required_models_complete": True,
                "trajectory_confident_claim": True,
                "nontrajectory_confident_claim": False,
                "trajectory_minus_nontrajectory_log_evidence": 101.0,
            }
        ]
    )
    funnel = build_funnel_summary(
        candidate_table=candidate_table,
        tier_summary=_tier_summary(),
        run_state_summary=pd.DataFrame(),
        high_specificity=high_specificity,
        validation_decisions=validation_decisions,
    )

    gates = build_gate_summary(
        candidate_table=candidate_table,
        tier_summary=_tier_summary(),
        high_specificity=high_specificity,
        validation_decisions=validation_decisions,
        funnel=funnel,
    ).set_index("gate")

    assert not bool(gates.loc["exact_validation_matches_promotion_ready", "passed"])
    assert not bool(gates.loc["overall", "passed"])


def test_off_swr_promotion_funnel_counts_only_complete_exact_core_rows():
    candidate_table = pd.DataFrame(
        [_candidate("Rat2/Open1", 4, 0, margin=120.0, state="immobile", label="interesting_off_swr_trajectory_candidate", tier="extreme")]
    )
    high_specificity = candidate_table.copy()
    high_specificity["passes_high_specificity_promotion_filter"] = True
    validation_decisions = pd.DataFrame(
        [
            {
                **_key("Rat2/Open1", 4, 0),
                "required_models_complete": False,
                "trajectory_confident_claim": True,
                "nontrajectory_confident_claim": False,
                "trajectory_minus_nontrajectory_log_evidence": 101.0,
            }
        ]
    )
    funnel = build_funnel_summary(
        candidate_table=candidate_table,
        tier_summary=_tier_summary(),
        run_state_summary=pd.DataFrame(),
        high_specificity=high_specificity,
        validation_decisions=validation_decisions,
    ).set_index("stage")

    assert int(funnel.loc["exact_core_validated_candidates", "windows"]) == 0
    assert int(funnel.loc["exact_core_trajectory_confident_candidates", "windows"]) == 0


def test_off_swr_promotion_funnel_parses_numeric_boolean_flags():
    candidate_table = pd.DataFrame(
        [
            _candidate(
                "Rat2/Open1",
                4,
                0,
                margin=120.0,
                state="immobile",
                label="interesting_off_swr_trajectory_candidate",
                tier="extreme",
            )
        ]
    )
    high_specificity = candidate_table.copy()
    high_specificity["passes_high_specificity_promotion_filter"] = [1.0]
    validation_decisions = pd.DataFrame(
        [
            {
                **_key("Rat2/Open1", 4, 0),
                "required_models_complete": 1.0,
                "trajectory_confident_claim": 1.0,
                "nontrajectory_confident_claim": 0.0,
                "trajectory_minus_nontrajectory_log_evidence": 101.0,
            }
        ]
    )

    funnel = build_funnel_summary(
        candidate_table=candidate_table,
        tier_summary=_tier_summary(),
        run_state_summary=pd.DataFrame(),
        high_specificity=high_specificity,
        validation_decisions=validation_decisions,
    ).set_index("stage")

    assert int(funnel.loc["promotion_ready_candidates", "windows"]) == 1
    assert int(funnel.loc["exact_core_validated_candidates", "windows"]) == 1
    assert int(funnel.loc["exact_core_trajectory_confident_candidates", "windows"]) == 1


def _key(session: str, event_index: int, null_index: int) -> dict[str, object]:
    return {"session": session, "event_index": event_index, "null_index": null_index}


def _candidate(
    session: str,
    event_index: int,
    null_index: int,
    *,
    margin: float,
    state: str,
    label: str,
    tier: str,
) -> dict[str, object]:
    rat = session.split("/")[0]
    return {
        **_key(session, event_index, null_index),
        "rat": rat,
        "candidate_rank": event_index,
        "candidate_specificity_label": label,
        "candidate_tier": tier,
        "trajectory_family_margin": margin,
        "best_trajectory_model": "sorted-spike-state-space-first-order-imm",
        "run_or_immobility_state": state,
        "animal_speed_mean": 1.0 if state == "immobile" else 20.0,
        "n_spikes": 20 + event_index,
        "active_cell_count": 5 + event_index,
        "distance_to_nearest_swr_s": 2.0,
    }


def _tier_summary() -> pd.DataFrame:
    rows = []
    for tier, threshold, count in [
        ("weak", 5.5, 4),
        ("moderate", 20.0, 3),
        ("strong", 50.0, 2),
        ("extreme", 100.0, 1),
    ]:
        rows.append(
            {
                "candidate_tier": tier,
                "tier_margin_threshold": threshold,
                "off_swr_windows": 10,
                "candidate_windows": count,
                "immobile_windows": 3,
                "running_windows": 7,
                "unknown_speed_windows": 0,
                "candidate_windows_after_1s_swr_exclusion": count if tier != "weak" else 4,
            }
        )
    return pd.DataFrame(rows)


def _tier_group_summary(*, group: str) -> pd.DataFrame:
    rows = []
    groups = [
        {"rat": "Rat1", "session": "Rat1/Open1", "off_swr_windows": 5, "counts": {"weak": 2, "moderate": 2, "strong": 0, "extreme": 0}},
        {"rat": "Rat2", "session": "Rat2/Open1", "off_swr_windows": 5, "counts": {"weak": 2, "moderate": 1, "strong": 2, "extreme": 1}},
    ]
    for item in groups:
        for tier, count in item["counts"].items():
            row = {
                "rat": item["rat"],
                "candidate_tier": tier,
                "off_swr_windows": item["off_swr_windows"],
                "candidate_windows": count,
            }
            if group == "session":
                row["session"] = item["session"]
            rows.append(row)
    return pd.DataFrame(rows)
