from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def test_threshold_validation_patch_reinstalls_after_overwrite(monkeypatch) -> None:
    import hipporeplayimm.advanced_result_diagnostics as diagnostics
    from hipporeplayimm.advanced_result_threshold_validation import apply_advanced_result_threshold_validation_patch

    def stale_decisions(*args, **kwargs) -> pd.DataFrame:
        del args, kwargs
        return pd.DataFrame()

    def stale_sweep(*args, **kwargs) -> pd.DataFrame:
        del args, kwargs
        return pd.DataFrame()

    monkeypatch.setattr(diagnostics, "_advanced_result_threshold_validation_patch_applied", True, raising=False)
    monkeypatch.setattr(diagnostics, "paired_model_margin_decisions", stale_decisions)
    monkeypatch.setattr(diagnostics, "paired_model_margin_threshold_sweep", stale_sweep)

    apply_advanced_result_threshold_validation_patch()

    with pytest.raises(ValueError, match="finite nonnegative"):
        diagnostics.paired_model_margin_decisions(
            pd.DataFrame(),
            positive_model="p",
            reference_model="r",
            margin_threshold=np.inf,
        )
    with pytest.raises(ValueError, match="thresholds"):
        diagnostics.paired_model_margin_threshold_sweep(
            pd.DataFrame(),
            positive_model="p",
            reference_model="r",
            thresholds=(0.0, np.nan),
            group_cols=("session", "event_index"),
        )


def test_runtime_margin_decisions_keep_scalar_threshold_validation_after_missing_group_patch() -> None:
    import hipporeplayimm
    import hipporeplayimm.advanced_result_diagnostics as diagnostics

    hipporeplayimm.apply_runtime_patches()

    for threshold in (True, np.bool_(False), np.array([0.0]), np.nan, np.inf):
        with pytest.raises(ValueError, match="finite nonnegative"):
            diagnostics.paired_model_margin_decisions(
                _paired_scores(),
                positive_model="momentum",
                reference_model="diffusion",
                margin_threshold=threshold,
            )


def test_runtime_margin_threshold_sweep_rejects_array_thresholds() -> None:
    import hipporeplayimm
    import hipporeplayimm.advanced_result_diagnostics as diagnostics

    hipporeplayimm.apply_runtime_patches()

    with pytest.raises(ValueError, match="thresholds"):
        diagnostics.paired_model_margin_threshold_sweep(
            _paired_scores(),
            positive_model="momentum",
            reference_model="diffusion",
            thresholds=(0.0, np.array([1.0])),
            group_cols=("session", "event_index"),
        )


def _paired_scores() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "session": ["Rat1/Open1", "Rat1/Open1"],
            "event_index": [0, 0],
            "model": ["momentum", "diffusion"],
            "log_evidence": [2.0, 0.0],
            "status": ["success", "success"],
            "evidence_comparable": [True, True],
        }
    )
