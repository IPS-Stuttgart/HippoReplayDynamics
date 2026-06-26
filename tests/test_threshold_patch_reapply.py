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
