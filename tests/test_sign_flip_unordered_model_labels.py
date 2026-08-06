from __future__ import annotations

import pandas as pd
import pytest

from hipporeplayimm.sign_flip_report import (
    _normalize_model_label,
    score_table_sign_flip_summary,
)


def test_sign_flip_summary_groups_equivalent_mapping_labels() -> None:
    primary = {
        "family": "state-space",
        "mode": "imm",
        "members": {"diffusion", "momentum"},
    }
    reordered = {
        "members": {"momentum", "diffusion"},
        "mode": "imm",
        "family": "state-space",
    }
    frame = pd.DataFrame(
        {
            "model": [primary, reordered, {"family": "other"}],
            "delta_vs_best_static": [1.0, 2.0, -1.0],
        }
    )

    summary = score_table_sign_flip_summary(frame)

    assert len(summary) == 2
    assert summary.loc[0, "n_observations"] == 2
    assert summary.loc[0, "observed_mean"] == pytest.approx(1.5)
    assert summary.loc[0, "p_value"] == pytest.approx(0.5)


def test_sign_flip_model_labels_canonicalize_unordered_sets() -> None:
    assert _normalize_model_label({"stationary", "diffusion"}) == _normalize_model_label(
        {"diffusion", "stationary"}
    )
