from __future__ import annotations

import pytest

from hipporeplayimm.position_validation import (
    PositionDecodingConfig,
    validate_session_position_decoding,
)


def test_position_decoding_normalizes_decode_bin_numeric_overflow() -> None:
    config = PositionDecodingConfig(decode_bin_s=10**400)

    with pytest.raises(
        ValueError,
        match="decode_bin_s must be finite and positive",
    ):
        validate_session_position_decoding(object(), config)
