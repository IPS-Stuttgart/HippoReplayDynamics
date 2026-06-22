from __future__ import annotations

import pandas as pd
import pytest

from hipporeplayimm.ground_truth import (
    _unique_float_from_columns,
    _unique_optional_float_from_column,
)


def test_ground_truth_float_metadata_rejects_nonfinite_values() -> None:
    scores = pd.DataFrame({"state_space_diffusion_sigma_cm_sqrt_s": ["inf"]})

    with pytest.raises(ValueError, match="finite numeric"):
        _unique_float_from_columns(
            scores,
            ("state_space_diffusion_sigma_cm_sqrt_s",),
            85.0,
        )


def test_ground_truth_optional_float_metadata_rejects_nonfinite_values() -> None:
    scores = pd.DataFrame(
        {"state_space_trajectory_imm_mode_stickiness": ["-inf"]}
    )

    with pytest.raises(ValueError, match="finite numeric"):
        _unique_optional_float_from_column(
            scores,
            "state_space_trajectory_imm_mode_stickiness",
            None,
        )


def test_ground_truth_float_metadata_still_ignores_missing_sentinels() -> None:
    scores = pd.DataFrame(
        {"state_space_diffusion_sigma_cm_sqrt_s": ["", "nan", None]}
    )

    assert _unique_float_from_columns(
        scores,
        ("state_space_diffusion_sigma_cm_sqrt_s",),
        85.0,
    ) == pytest.approx(85.0)
