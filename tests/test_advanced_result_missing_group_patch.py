from __future__ import annotations

import numpy as np
import pandas as pd

import hipporeplayimm
from hipporeplayimm import advanced_result_diagnostics as diagnostics
from hipporeplayimm.advanced_result_diagnostics import (
    add_evidence_margin_columns,
    evidence_margin_table,
)


def _scores_with_missing_window_metadata() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "session": ["Rat1/Open1", "Rat1/Open1"],
            "event_index": [0, 0],
            "event_window_variant": [pd.NA, pd.NA],
            "model": ["stationary", "diffusion"],
            "log_evidence": [10.0, 12.0],
            "status": ["success", "success"],
            "evidence_comparable": [True, True],
        }
    )


def _legacy_dropna_evidence_margin_table(
    scores: pd.DataFrame,
    *,
    group_cols=("session", "event_index"),
    evidence_col: str = "log_evidence",
    model_col: str = "model",
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    ok = diagnostics._comparable_rows(scores)
    for key, group in ok.groupby(list(group_cols), sort=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        group = group.dropna(subset=[evidence_col]).sort_values(evidence_col, ascending=False)
        if group.empty:
            continue
        best = group.iloc[0]
        second = group.iloc[1] if len(group) > 1 else None
        second_value = float(second[evidence_col]) if second is not None else np.nan
        margin = float(best[evidence_col]) - second_value if second is not None else np.inf
        row = dict(zip(group_cols, key_tuple, strict=True))
        row.update(
            {
                "best_model_by_evidence": str(best[model_col]),
                "second_best_model_by_evidence": "" if second is None else str(second[model_col]),
                "best_log_evidence": float(best[evidence_col]),
                "second_best_log_evidence": second_value,
                "evidence_margin_to_second_best": margin,
                "evidence_margin_category": diagnostics.classify_evidence_margin(margin),
                "models_compared": int(len(group)),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _legacy_dropna_add_evidence_margin_columns(
    scores: pd.DataFrame,
    *,
    group_cols=("session", "event_index"),
) -> pd.DataFrame:
    margins = _legacy_dropna_evidence_margin_table(scores, group_cols=group_cols)
    if margins.empty:
        out = scores.copy()
        out["evidence_margin_to_second_best"] = np.nan
        out["evidence_margin_category"] = "missing"
        return out
    return scores.merge(margins, on=list(group_cols), how="left")


def test_evidence_margin_table_keeps_missing_optional_group_metadata() -> None:
    scores = _scores_with_missing_window_metadata()

    margins = evidence_margin_table(
        scores,
        group_cols=("session", "event_index", "event_window_variant"),
    )

    assert len(margins) == 1
    assert margins["event_window_variant"].isna().all()
    assert margins.loc[0, "best_model_by_evidence"] == "diffusion"
    assert margins.loc[0, "second_best_model_by_evidence"] == "stationary"
    assert np.isclose(margins.loc[0, "evidence_margin_to_second_best"], 2.0)


def test_evidence_margin_columns_merge_back_missing_optional_group_metadata() -> None:
    scores = _scores_with_missing_window_metadata()

    merged = add_evidence_margin_columns(
        scores,
        group_cols=("session", "event_index", "event_window_variant"),
    )

    assert len(merged) == len(scores)
    assert merged["best_model_by_evidence"].tolist() == ["diffusion", "diffusion"]
    assert np.allclose(merged["evidence_margin_to_second_best"].to_numpy(dtype=float), [2.0, 2.0])


def test_runtime_patches_restore_stale_missing_group_evidence_margin_aliases() -> None:
    scores = pd.DataFrame(
        {
            "session": ["Rat1/Open1", "Rat1/Open1"],
            "event_index": [3, 3],
            "window_index": [np.nan, np.nan],
            "status": ["success", "success"],
            "model": ["stationary", "diffusion"],
            "log_evidence": [1.0, 4.0],
            "evidence_comparable": [True, True],
        }
    )
    group_cols = ("session", "event_index", "window_index")

    diagnostics.evidence_margin_table = _legacy_dropna_evidence_margin_table
    diagnostics.add_evidence_margin_columns = _legacy_dropna_add_evidence_margin_columns

    hipporeplayimm.apply_runtime_patches()

    margins = diagnostics.evidence_margin_table(scores, group_cols=group_cols)
    assert margins.shape[0] == 1
    assert pd.isna(margins.loc[0, "window_index"])
    assert margins.loc[0, "best_model_by_evidence"] == "diffusion"
    assert float(margins.loc[0, "evidence_margin_to_second_best"]) == 3.0

    annotated = diagnostics.add_evidence_margin_columns(scores, group_cols=group_cols)
    assert annotated["evidence_margin_category"].tolist() == ["weak", "weak"]
