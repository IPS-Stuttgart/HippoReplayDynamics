from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hipporeplayimm.bma_options_patch import (
    _apply_bma_output_options,
    _coerce_bool_option,
    _coerce_text_option,
    _option_text,
)


@pytest.mark.parametrize(
    "value",
    [b"true", bytearray(b"true"), memoryview(b"true"), np.bytes_(b"true")],
)
def test_bma_bool_option_accepts_byte_backed_true_values(value: object) -> None:
    assert _coerce_bool_option(value) is True


@pytest.mark.parametrize(
    "value",
    [b"false", bytearray(b"0"), memoryview(b"off"), np.bytes_(b"false")],
)
def test_bma_bool_option_accepts_byte_backed_false_values(value: object) -> None:
    assert _coerce_bool_option(value) is False


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (b"custom_log_evidence", "custom_log_evidence"),
        (bytearray(b"ensemble"), "ensemble"),
        (memoryview(b"bayesian-model-average"), "bayesian-model-average"),
        (np.bytes_(b"auto"), "auto"),
    ],
)
def test_bma_text_options_decode_byte_backed_scalars(
    value: object,
    expected: str,
) -> None:
    assert _option_text(value) == expected


@pytest.mark.parametrize(
    "value",
    ["auto", np.str_("auto"), np.array("auto"), b"auto", bytearray(b"auto"), memoryview(b"auto"), np.bytes_(b"auto")],
)
def test_bma_text_option_accepts_scalar_values(value: object) -> None:
    assert _coerce_text_option(value) == "auto"


@pytest.mark.parametrize(
    "value",
    [["auto"], ("auto",), np.array(["auto"]), np.array([b"auto"])],
)
def test_bma_text_option_rejects_non_scalar_values(value: object) -> None:
    with pytest.raises(ValueError, match="text option must be a scalar"):
        _coerce_text_option(value)


def test_bma_output_options_match_and_rename_byte_backed_model_labels() -> None:
    comparison = pd.DataFrame(
        {
            "model": pd.Series(
                [np.bytes_(b"bayesian-model-average"), "diffusion"],
                dtype=object,
            ),
            "requested_model": pd.Series(
                [memoryview(b"bayesian-model-average"), "diffusion"],
                dtype=object,
            ),
        }
    )

    removed = _apply_bma_output_options(
        comparison,
        include_bma=False,
        model_name="bayesian-model-average",
    )
    assert removed["model"].tolist() == ["diffusion"]

    renamed = _apply_bma_output_options(
        comparison,
        include_bma=True,
        model_name=bytearray(b"ensemble"),
    )
    assert renamed["model"].tolist() == ["ensemble", "diffusion"]
    assert renamed["requested_model"].tolist() == ["ensemble", "diffusion"]
