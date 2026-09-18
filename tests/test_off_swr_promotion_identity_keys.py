from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))

from build_off_swr_promotion_funnel import _read_required_csv as read_funnel_csv  # noqa: E402
from calibrate_off_swr_promotion_fdr import (  # noqa: E402
    _read_required_csv as read_fdr_csv,
    build_gate_summary,
)


def test_promotion_csv_readers_preserve_adjacent_large_decimal_keys(tmp_path: Path) -> None:
    first = 2**53
    second = first + 1
    path = tmp_path / "promotion_keys.csv"
    path.write_text(
        "session,event_index,null_index,trajectory_family_margin\n"
        f"Rat1/Open1,{first}.0,0.0,10.0\n"
        f"Rat1/Open1,{second}.0,1.0,20.0\n",
        encoding="utf-8",
    )

    for reader in (read_funnel_csv, read_fdr_csv):
        loaded = reader(path)
        assert loaded["event_index"].tolist() == [first, second]
        assert loaded["null_index"].tolist() == [0, 1]


def test_fdr_gate_rejects_equal_count_validation_for_wrong_candidate() -> None:
    high_specificity = pd.DataFrame(
        [
            {
                "session": "Rat1/Open1",
                "event_index": 4,
                "null_index": 0,
                "passes_high_specificity_promotion_filter": True,
            }
        ]
    )
    validation_decisions = pd.DataFrame(
        [
            {
                "session": "Rat1/Open1",
                "event_index": 999,
                "null_index": 0,
                "required_models_complete": True,
                "trajectory_confident_claim": True,
            }
        ]
    )
    summary = pd.DataFrame(
        [
            {
                "screened_off_swr_windows": 20,
                "promotion_ready_windows": 1,
                "exact_validated_windows": 1,
                "exact_trajectory_confident_windows": 1,
                "direct_control_windows": 2,
                "direct_control_promotion_ready_windows": 0,
                "p95_permutation_fdr_bound": 0.5,
                "min_permutation_empirical_p_value": 0.001,
            }
        ]
    )
    calibration = pd.DataFrame(
        [
            {
                "null_type": "permutation_null",
                "permutations": 100,
                "observed_exceeds_null_p95": True,
                "observed_exceeds_null_p99": True,
            }
        ]
    )
    threshold = pd.DataFrame(
        [
            {
                "candidate_tier": "strong",
                "promotion_ready_windows": 1,
                "observed_exceeds_joint_shuffle_p95": True,
            },
            {
                "candidate_tier": "extreme",
                "promotion_ready_windows": 1,
                "observed_exceeds_joint_shuffle_p95": True,
            },
        ]
    )

    gates = build_gate_summary(
        summary=summary,
        calibration=calibration,
        threshold_sensitivity=threshold,
        validation_decisions=validation_decisions,
        high_specificity=high_specificity,
        n_permutations=100,
    ).set_index("gate")

    identity_gate = gates.loc["exact_validation_matches_promotion_ready"]
    assert not bool(identity_gate["passed"])
    assert "matched_keys=0/1" in str(identity_gate["observed"])
    assert not bool(gates.loc["overall", "passed"])
