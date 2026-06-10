import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))

from calibrate_off_swr_promotion_fdr import (  # noqa: E402
    build_null_calibration,
    build_threshold_sensitivity,
    write_off_swr_promotion_fdr_outputs,
)


def test_off_swr_promotion_fdr_calibration_detects_specific_alignment(tmp_path):
    high = pd.DataFrame(
        [
            _high_candidate(0, margin=120.0, state="immobile", interesting=True, promoted=True),
            _high_candidate(1, margin=110.0, state="immobile", interesting=True, promoted=True),
            _high_candidate(2, margin=105.0, state="immobile", interesting=True, promoted=True),
            _high_candidate(3, margin=90.0, state="immobile", interesting=True, promoted=True),
            _high_candidate(4, margin=88.0, state="immobile", interesting=True, promoted=True),
            _high_candidate(5, margin=80.0, state="immobile", interesting=True, promoted=True),
            _high_candidate(6, margin=79.0, state="run", interesting=True, promoted=False),
            _high_candidate(7, margin=78.0, state="run", interesting=True, promoted=False),
            _high_candidate(8, margin=70.0, state="run", interesting=False, promoted=False),
            _high_candidate(9, margin=65.0, state="run", interesting=False, promoted=False),
            _high_candidate(10, margin=60.0, state="run", interesting=False, promoted=False),
            _high_candidate(11, margin=55.0, state="run", interesting=False, promoted=False),
        ]
    )

    calibration = build_null_calibration(
        high_specificity=high,
        n_permutations=500,
        random_seed=3,
    ).set_index("control_source")

    assert int(calibration.loc["running_high_specificity_controls", "control_promotion_ready_windows"]) == 0
    assert int(calibration.loc["ordinary_movement_spiking_high_specificity_controls", "control_promotion_ready_windows"]) == 0
    assert bool(calibration.loc["joint_label_immobility_shuffle_null", "observed_exceeds_null_p95"])


def test_off_swr_promotion_fdr_outputs_and_gates_pass(tmp_path):
    discovery = tmp_path / "discovery"
    validation = tmp_path / "validation"
    output = tmp_path / "fdr"
    discovery.mkdir()
    validation.mkdir()

    high = pd.DataFrame(
        [
            _high_candidate(0, margin=120.0, state="immobile", interesting=True, promoted=True),
            _high_candidate(1, margin=110.0, state="immobile", interesting=True, promoted=True),
            _high_candidate(2, margin=105.0, state="immobile", interesting=True, promoted=True),
            _high_candidate(3, margin=90.0, state="immobile", interesting=True, promoted=True),
            _high_candidate(4, margin=88.0, state="immobile", interesting=True, promoted=True),
            _high_candidate(5, margin=80.0, state="immobile", interesting=True, promoted=True),
            _high_candidate(6, margin=79.0, state="run", interesting=True, promoted=False),
            _high_candidate(7, margin=78.0, state="run", interesting=True, promoted=False),
            _high_candidate(8, margin=70.0, state="run", interesting=False, promoted=False),
            _high_candidate(9, margin=65.0, state="run", interesting=False, promoted=False),
            _high_candidate(10, margin=60.0, state="run", interesting=False, promoted=False),
            _high_candidate(11, margin=55.0, state="run", interesting=False, promoted=False),
        ]
    )
    weak = pd.DataFrame(
        [
            *high.to_dict("records"),
            _candidate(12, margin=30.0, state="run", interesting=False),
            _candidate(13, margin=10.0, state="run", interesting=False),
        ]
    )
    validation_decisions = high[high["passes_high_specificity_promotion_filter"]].copy()
    validation_decisions["required_models_complete"] = True
    validation_decisions["trajectory_confident_claim"] = True
    validation_decisions["nontrajectory_confident_claim"] = False
    validation_decisions["trajectory_minus_nontrajectory_log_evidence"] = validation_decisions[
        "trajectory_family_margin"
    ]

    weak.to_csv(discovery / "off_swr_candidate_table.csv", index=False)
    high.to_csv(discovery / "off_swr_high_specificity_candidate_table.csv", index=False)
    _tier_summary().to_csv(discovery / "off_swr_candidate_tier_threshold_summary.csv", index=False)
    validation_decisions.to_csv(validation / "promoted_off_swr_candidate_exact_core_decisions.csv", index=False)

    outputs = write_off_swr_promotion_fdr_outputs(
        discovery_dir=discovery,
        validation_dir=validation,
        output=output,
        n_permutations=500,
        random_seed=3,
    )

    summary = outputs["off_swr_promotion_empirical_fdr_summary.csv"].iloc[0]
    assert summary["fdr_calibration_status"] == "empirical_controls_support_promotion_specificity"
    assert int(summary["promotion_ready_windows"]) == 6
    assert int(summary["direct_control_windows"]) == 6
    assert int(summary["direct_control_promotion_ready_windows"]) == 0

    gates = outputs["off_swr_promotion_null_gate_summary.csv"].set_index("gate")
    assert int(gates.loc["direct_control_pools_present", "observed"]) == 6
    assert bool(gates.loc["overall", "passed"])

    threshold = build_threshold_sensitivity(
        candidate_table=weak,
        high_specificity=high,
        tier_summary=_tier_summary(),
        screened=20,
        n_permutations=500,
        random_seed=3,
    ).set_index("candidate_tier")
    assert int(threshold.loc["strong", "promotion_ready_windows"]) == 6
    assert int(threshold.loc["extreme", "promotion_ready_windows"]) == 3

    for filename in outputs:
        assert (output / filename).exists()


def _candidate(index: int, *, margin: float, state: str, interesting: bool) -> dict[str, object]:
    return {
        "session": "Rat1/Open1",
        "rat": "Rat1",
        "event_index": index,
        "null_index": index,
        "candidate_rank": index + 1,
        "candidate_specificity_label": "interesting_off_swr_trajectory_candidate"
        if interesting
        else "ordinary_movement_or_spiking_like",
        "candidate_tier": "extreme" if margin >= 100.0 else "strong" if margin >= 50.0 else "moderate",
        "trajectory_family_margin": margin,
        "best_trajectory_model": "sorted-spike-state-space-first-order-imm",
        "distance_to_nearest_swr_s": 2.0,
        "run_or_immobility_state": state,
        "animal_speed_mean": 1.0 if state == "immobile" else 20.0,
    }


def _high_candidate(
    index: int,
    *,
    margin: float,
    state: str,
    interesting: bool,
    promoted: bool,
) -> dict[str, object]:
    row = _candidate(index, margin=margin, state=state, interesting=interesting)
    row.update(
        {
            "high_specificity_label": "promotion_ready_high_specificity_candidate"
            if promoted
            else "tier_distance_candidate_movement_spiking_or_low_information",
            "passes_strong_tier": True,
            "passes_1s_swr_exclusion": True,
            "speed_available": True,
            "passes_immobility_filter": state == "immobile",
            "passes_specificity_label_filter": interesting,
            "passes_high_specificity_promotion_filter": promoted,
        }
    )
    return row


def _tier_summary() -> pd.DataFrame:
    rows = []
    for tier, threshold, count in [
        ("weak", 5.5, 14),
        ("moderate", 20.0, 13),
        ("strong", 50.0, 12),
        ("extreme", 100.0, 3),
    ]:
        rows.append(
            {
                "candidate_tier": tier,
                "tier_margin_threshold": threshold,
                "off_swr_windows": 20,
                "candidate_windows": count,
            }
        )
    return pd.DataFrame(rows)
