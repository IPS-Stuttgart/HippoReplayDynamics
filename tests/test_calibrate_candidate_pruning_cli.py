from __future__ import annotations

import pytest

from scripts.calibrate_candidate_pruning import _parse_models


def test_parse_models_accepts_comma_and_whitespace_separators() -> None:
    assert _parse_models("momentum, imm") == ("momentum", "imm")
    assert _parse_models("momentum imm") == ("momentum", "imm")
    assert _parse_models("stationary, diffusion momentum") == (
        "stationary",
        "diffusion",
        "momentum",
    )


@pytest.mark.parametrize("value", ["", "   ", "momentum,,imm", "momentum,", ",imm"])
def test_parse_models_rejects_empty_model_entries(value: str) -> None:
    with pytest.raises(ValueError):
        _parse_models(value)
