from __future__ import annotations

import pandas as pd
import pytest

from hipporeplayimm.ground_truth import (
    _parse_bool,
    _unique_bool_from_column,
    _unique_float_from_columns,
    _unique_optional_float_from_column,
)


class _OverflowingNumeric:
    def __float__(self) -> float:
        raise OverflowError("too large to convert to float")


def test_ground_truth_float_metadata_rejects_nonfinite_values() -> None:
    scores = pd.DataFrame({"state_space_diffusion_sigma_cm_sqrt_s": ["inf"]})

    with pytest.raises(ValueError, match="finite numeric"):
        _unique_float_from_columns(
            scores,
            ("state_space_diffusion_sigma_cm_sqrt_s",),
            85.0,
        )


def test_ground_truth_float_metadata_overflow_is_value_error() -> None:
    scores = pd.DataFrame({"state_space_diffusion_sigma_cm_sqrt_s": [_OverflowingNumeric()]})

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


def test_ground_truth_bool_metadata_rejects_invalid_numeric_values() -> None:
    scores = pd.DataFrame({"encoding_use_excitatory": [2]})

    with pytest.raises(ValueError, match="boolean values"):
        _unique_bool_from_column(scores, "encoding_use_excitatory", True)

    with pytest.raises(ValueError, match="boolean values"):
        _parse_bool(2)


def test_ground_truth_bool_metadata_overflow_is_value_error() -> None:
    scores = pd.DataFrame({"encoding_use_excitatory": [_OverflowingNumeric()]})

    with pytest.raises(ValueError, match="boolean values"):
        _unique_bool_from_column(scores, "encoding_use_excitatory", True)

    with pytest.raises(ValueError, match="boolean values"):
        _parse_bool(_OverflowingNumeric())


def test_ground_truth_bool_metadata_accepts_boolean_like_values() -> None:
    true_scores = pd.DataFrame({"encoding_use_excitatory": ["yes", "1"]})
    false_scores = pd.DataFrame({"encoding_use_excitatory": ["no", 0]})

    assert _unique_bool_from_column(true_scores, "encoding_use_excitatory", False) is True
    assert _unique_bool_from_column(false_scores, "encoding_use_excitatory", True) is False
