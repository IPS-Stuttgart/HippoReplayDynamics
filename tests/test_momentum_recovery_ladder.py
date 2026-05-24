import numpy as np
import pandas as pd

from hipporeplayimm.momentum_recovery_ladder import (
    DIFFUSION_MODEL,
    FINITE_VELOCITY_MODEL,
    PAIRWISE_MOMENTUM_MODEL,
    interpret_ladder_summary,
    ladder_event_recovery,
    summarize_ladder_tiers,
)


def test_ladder_summary_identifies_native_candidate_support_loss():
    scores = pd.DataFrame(
        [
            *_tier_rows(
                "full_grid_pairwise_momentum",
                0,
                PAIRWISE_MOMENTUM_MODEL,
                expected_log_evidence=5.0,
                diffusion_log_evidence=1.0,
                expected_support="exact_full_grid",
            ),
            *_tier_rows(
                "exact_finite_velocity_momentum",
                1,
                FINITE_VELOCITY_MODEL,
                expected_log_evidence=4.0,
                diffusion_log_evidence=1.0,
                expected_support="exact_full_grid",
            ),
            *_tier_rows(
                "oracle_candidate_pairwise_momentum",
                2,
                PAIRWISE_MOMENTUM_MODEL,
                expected_log_evidence=3.0,
                diffusion_log_evidence=1.0,
                expected_support="truncated_full_grid",
                oracle=True,
            ),
            *_tier_rows(
                "native_candidate_pairwise_momentum",
                3,
                PAIRWISE_MOMENTUM_MODEL,
                expected_log_evidence=0.0,
                diffusion_log_evidence=1.0,
                expected_support="truncated_full_grid",
            ),
        ]
    )

    events = ladder_event_recovery(scores)
    summary = summarize_ladder_tiers(events)
    interpretation = interpret_ladder_summary(summary)

    native = summary[summary["ladder_tier"] == "native_candidate_pairwise_momentum"].iloc[0]
    assert native["certified_or_strict_recovered_events"] == 0
    assert interpretation.iloc[0]["diagnosis"] == "native_candidate_support_loss"


def test_ladder_summary_identifies_full_grid_failure_first():
    summary = pd.DataFrame(
        [
            {"ladder_tier": "full_grid_pairwise_momentum", "certified_or_strict_recovery_fraction": 0.0},
            {"ladder_tier": "exact_finite_velocity_momentum", "certified_or_strict_recovery_fraction": 1.0},
            {"ladder_tier": "oracle_candidate_pairwise_momentum", "certified_or_strict_recovery_fraction": 1.0},
            {"ladder_tier": "native_candidate_pairwise_momentum", "certified_or_strict_recovery_fraction": 1.0},
        ]
    )

    interpretation = interpret_ladder_summary(summary)

    assert interpretation.iloc[0]["diagnosis"] == "full_grid_pairwise_momentum_fails"


def _tier_rows(
    tier: str,
    tier_index: int,
    expected_model: str,
    *,
    expected_log_evidence: float,
    diffusion_log_evidence: float,
    expected_support: str,
    oracle: bool = False,
) -> list[dict[str, object]]:
    return [
        _row(
            tier,
            tier_index,
            expected_model,
            expected_model,
            expected_log_evidence,
            expected_support,
            oracle=oracle,
        ),
        _row(
            tier,
            tier_index,
            expected_model,
            DIFFUSION_MODEL,
            diffusion_log_evidence,
            "exact_full_grid",
            oracle=oracle,
        ),
    ]


def _row(
    tier: str,
    tier_index: int,
    expected_model: str,
    model: str,
    log_evidence: float,
    evidence_support: str,
    *,
    oracle: bool = False,
) -> dict[str, object]:
    return {
        "status": "success",
        "session": "RatX/OpenY",
        "event_index": int(tier_index),
        "simulation_event_index": int(tier_index),
        "true_model": "momentum",
        "expected_model": expected_model,
        "ladder_expected_model": expected_model,
        "model": model,
        "log_evidence": log_evidence,
        "n_time": 5,
        "n_spikes": 20,
        "ladder_tier": tier,
        "ladder_tier_index": tier_index,
        "ladder_oracle_candidate_support": oracle,
        "evidence_support": evidence_support,
        "evidence_comparable": evidence_support == "exact_full_grid",
        "diagnostic_state_space_momentum_evidence_support": (
            evidence_support if model == PAIRWISE_MOMENTUM_MODEL else np.nan
        ),
    }
