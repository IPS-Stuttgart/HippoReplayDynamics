from __future__ import annotations

import pytest

import hipporeplayimm.models as models
import hipporeplayimm.state_space_utils as state_space_utils


@pytest.mark.parametrize(
    "value",
    [bytearray(b"0.5"), memoryview(b"0.5")],
    ids=["bytearray", "memoryview"],
)
def test_model_parameter_validators_reject_text_byte_buffers(value: object) -> None:
    with pytest.raises(TypeError, match=r"diffusion_scale.*string"):
        models._validate_positive_parameter("diffusion_scale", value)


@pytest.mark.parametrize(
    "value",
    [bytearray(b"2"), memoryview(b"2")],
    ids=["bytearray", "memoryview"],
)
def test_state_space_count_validators_reject_text_byte_buffers(value: object) -> None:
    with pytest.raises(TypeError, match=r"top_k.*string"):
        state_space_utils._coerce_integer_count("top_k", value)
