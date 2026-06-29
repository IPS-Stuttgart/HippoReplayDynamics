from __future__ import annotations

import pandas as pd

from hipporeplayimm import advanced_result_diagnostics as diagnostics

GROUP_COLS = ("session", "event_index", "event_window_variant")
STATIONARY = "sorted-spike-state-space-stationary"
DIFFUSION = "sorted-spike-state-space-diffusion"
FRAGMENTED = "sorted-spike-state-space-fragmented"
FIRST_ORDER_IMM = "sorted-spike-state-space-first-order-imm"
MOMENTUM = "sorted-spike-state-space-momentum-exact-sparse"


def _wrong_map_scores() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = [
        (STATIONARY, 10.0, 7.0),
        (DIFFUSION, 15.0, 9.0),
        (FRAGMENTED, 12.0, 8.0),
        (FIRST_ORDER_IMM, 18.0, 10.0),
        (MOMENTUM, 17.0, 11.0),
    ]
    base = {
        "session": "Rat1/Open1",
        "event_index": 0,
        "event_window_variant": pd.NA,
        "status": "success",
    }
    current = pd.DataFrame(
        [{**base, "model": model, "log_evidence": real} for model, real, _ in rows]
    )
    wrong = pd.DataFrame(
        [
            {
                **base,
                "model": model,
                "log_evidence": wrong_value,
                "map_session": "Rat2/Open1",
            }
            for model, _, wrong_value in rows
        ]
    )
    return current, wrong


def test_wrong_map_selected_rows_keep_missing_optional_group_metadata() -> None:
    current, wrong = _wrong_map_scores()

    deltas = diagnostics.wrong_map_absolute_evidence_deltas(
        current,
        wrong,
        group_cols=GROUP_COLS,
    )

    selected = deltas[deltas["statistic_type"].eq("real_map_selected_model")]
    assert selected["statistic"].tolist() == [
        "best_exact_core_model_real_map",
        "best_exact_trajectory_model_real_map",
    ]
    assert selected["event_window_variant"].isna().all()
    assert selected["selected_model"].tolist() == [FIRST_ORDER_IMM, FIRST_ORDER_IMM]


def test_wrong_map_summary_keeps_missing_optional_group_metadata() -> None:
    current, wrong = _wrong_map_scores()
    deltas = diagnostics.wrong_map_absolute_evidence_deltas(
        current,
        wrong,
        group_cols=GROUP_COLS,
    )

    summary = diagnostics.wrong_map_absolute_evidence_summary(
        deltas,
        group_cols=("event_window_variant",),
    )

    assert not summary.empty
    assert summary["event_window_variant"].isna().all()
    assert "best_exact_core_model_real_map" in set(summary["statistic"])


def test_wrong_map_difference_in_differences_keeps_missing_optional_group_metadata() -> None:
    current, wrong = _wrong_map_scores()

    did = diagnostics.wrong_map_family_margin_difference_in_differences(
        current,
        wrong,
        group_cols=GROUP_COLS,
    )

    assert len(did) == 1
    assert did["event_window_variant"].isna().all()
    assert did.loc[0, "real_best_trajectory_model"] == FIRST_ORDER_IMM
    assert did.loc[0, "wrong_best_trajectory_model"] == MOMENTUM
